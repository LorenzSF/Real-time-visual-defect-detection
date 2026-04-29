"""Streaming inference loop for real-time visual defect detection.

Iterates over an image source (folder or simulated camera), runs per-frame
inference, and writes the streaming-session outputs:
``benchmark_summary.json`` + ``predictions_<model>.json`` +
``live_status_<model>.json`` (+ ``heatmaps/`` overlays) per PLAN.md §1.3.

Folds the previous input_handler.py + decision_engine.py + live_metrics.py
+ report_generator.py into one module so the package matches the four-file
layout.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA

from benchmark_AD.data import (
    LabeledSample,
    list_labeled_images,
    normalize_0_1,
    read_image_bgr,
    resize as resize_bgr,
)

from .dashboard import LiveDashboardServer
from .inference import FrameInference
from .settings import resolve_runtime_settings


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def _score_ratio(score: float, threshold: float) -> float:
    denom = max(float(threshold), 1e-6)
    return float(np.clip(float(score) / denom, 0.0, 1.0))


def _green_red_overlay(
    image_bgr: np.ndarray,
    anomaly_map: Optional[np.ndarray],
) -> np.ndarray:
    if anomaly_map is None:
        return image_bgr.copy()

    heatmap = np.asarray(anomaly_map, dtype=np.float32)
    if heatmap.ndim != 2:
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]
    resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    clipped = np.clip(resized, 0.0, 1.0)[..., None]
    green = np.asarray([0.0, 255.0, 0.0], dtype=np.float32)
    red = np.asarray([0.0, 0.0, 255.0], dtype=np.float32)
    colored = ((1.0 - clipped) * green + clipped * red).astype(np.uint8)
    return cv2.addWeighted(image_bgr, 0.62, colored, 0.38, 0.0)


@dataclass
class _Frame:
    frame_id: int
    timestamp_utc: str
    path: Path
    raw_image_bgr: np.ndarray
    model_input: np.ndarray
    label: int = -1
    defect_type: Optional[str] = None


class _SessionWriter:
    """Streams JSON session artifacts and heatmap overlays to disk."""

    def __init__(
        self,
        session_dir: Path,
        model_name: str,
        benchmark_summary: Dict[str, Any],
    ) -> None:
        self.session_dir = session_dir
        self.heatmaps_dir = session_dir / "heatmaps"
        self.dashboard_dir = session_dir / "dashboard"
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = _safe_name(model_name)
        self.predictions_path = session_dir / f"predictions_{safe_model_name}.json"
        self.status_path = session_dir / f"live_status_{safe_model_name}.json"
        self.summary_path = session_dir / "benchmark_summary.json"
        self.current_frame_path = self.dashboard_dir / "current_frame.png"
        self.predictions_path.write_text("[]", encoding="utf-8")
        self.summary_path.write_text(
            json.dumps(benchmark_summary, indent=2),
            encoding="utf-8",
        )
        self._rows: List[Dict[str, Any]] = []
        self._lock = Lock()

    def append_decision(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self._rows.append(row)
            self.predictions_path.write_text(
                json.dumps(self._rows, indent=2), encoding="utf-8"
            )

    def write_status(self, status: Dict[str, Any]) -> None:
        with self._lock:
            self.status_path.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )

    def save_heatmap(
        self,
        frame: _Frame,
        anomaly_map: Optional[np.ndarray],
        pred_is_anomaly: int,
    ) -> Optional[str]:
        if pred_is_anomaly != 1 or anomaly_map is None:
            return None
        overlay = _green_red_overlay(frame.raw_image_bgr, anomaly_map)
        relative = Path("heatmaps") / f"frame_{frame.frame_id:06d}.png"
        cv2.imwrite(str(self.session_dir / relative), overlay)
        return relative.as_posix()

    def save_current_frame(
        self,
        frame: _Frame,
        anomaly_map: Optional[np.ndarray],
    ) -> str:
        overlay = _green_red_overlay(frame.raw_image_bgr, anomaly_map)
        cv2.imwrite(str(self.current_frame_path), overlay)
        return Path("dashboard").joinpath("current_frame.png").as_posix()


class _LiveMetrics:
    """Rolling latency + decision counters that feed the live_status snapshot."""

    def __init__(
        self,
        session_dir: Path,
        input_fps: float,
        latency_sla_ms: float,
        score_history_size: int,
    ) -> None:
        self.session_dir = session_dir
        self.input_fps = float(input_fps)
        self.latency_sla_ms = float(latency_sla_ms)
        self.started_at = time.perf_counter()
        self.frames_seen = 0
        self.decisions_emitted = 0
        self.fail_count = 0
        self.latencies_ms: deque[float] = deque(maxlen=2048)
        self.completed_at_s: deque[float] = deque(maxlen=10)
        self.score_history: deque[float] = deque(maxlen=max(1, int(score_history_size)))
        self.recent_decisions: deque[Dict[str, Any]] = deque(maxlen=12)
        self.recent_fails: deque[Dict[str, Any]] = deque(maxlen=8)
        self.latest_score = 0.0
        self.current_frame_path: Optional[str] = None
        self.current_frame_version = 0
        self._lock = Lock()

    def record_frame(self) -> None:
        with self._lock:
            self.frames_seen += 1

    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latencies_ms.append(float(latency_ms))

    def record_decision(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self.decisions_emitted += 1
            self.completed_at_s.append(time.perf_counter())
            self.score_history.append(float(row.get("score", 0.0)))
            self.latest_score = float(row.get("score", 0.0))
            self.current_frame_path = (
                str(row["current_frame_path"]) if row.get("current_frame_path") is not None else None
            )
            self.current_frame_version = int(row.get("current_frame_version", 0))
            self.recent_decisions.appendleft(row)
            if int(row.get("pred_is_anomaly", 0)) == 1:
                self.fail_count += 1
                self.recent_fails.appendleft(row)

    def snapshot(
        self,
        model_name: str,
        threshold: float,
        extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._lock:
            uptime = max(time.perf_counter() - self.started_at, 1e-6)
            latencies = np.asarray(self.latencies_ms, dtype=np.float32)
            mean_latency = float(np.mean(latencies)) if latencies.size else 0.0
            p95_latency = float(np.percentile(latencies, 95)) if latencies.size else 0.0
            rolling_fps = 0.0
            if len(self.completed_at_s) >= 2:
                span = float(self.completed_at_s[-1] - self.completed_at_s[0])
                if span > 0.0:
                    rolling_fps = float((len(self.completed_at_s) - 1) / span)
            anomaly_rate = 0.0
            if self.decisions_emitted > 0:
                anomaly_rate = float(self.fail_count / self.decisions_emitted)
            axis_max = max(
                float(threshold) * 1.5,
                self.latest_score * 1.2,
                max(self.score_history, default=0.0) * 1.1,
                1e-6,
            )
            status = {
                "session_dir": str(self.session_dir),
                "active_model": model_name,
                "frames_seen": self.frames_seen,
                "decisions_emitted": self.decisions_emitted,
                "fail_count": self.fail_count,
                "input_fps": self.input_fps,
                "processed_fps": float(self.frames_seen / uptime),
                "decision_fps": float(self.decisions_emitted / uptime),
                "mean_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "latency_sla_ms": self.latency_sla_ms,
                "threshold": float(threshold),
                "latest_score": self.latest_score,
                "score_axis_max": axis_max,
                "score_history": list(self.score_history),
                "rolling_fps_10": rolling_fps,
                "anomaly_rate": anomaly_rate,
                "current_frame_path": self.current_frame_path,
                "current_frame_version": self.current_frame_version,
                "recent_decisions": list(self.recent_decisions),
                "recent_fails": list(self.recent_fails),
            }
            status.update(extras)
            return status


def _interleave_by_label(samples: List[LabeledSample]) -> List[LabeledSample]:
    buckets = {
        0: deque(s for s in samples if s.label == 0),
        1: deque(s for s in samples if s.label == 1),
        -1: deque(s for s in samples if s.label == -1),
    }
    ordered: List[LabeledSample] = []
    label_cycle = [0, 1, -1]
    while any(buckets[label] for label in label_cycle):
        for label in label_cycle:
            if buckets[label]:
                ordered.append(buckets[label].popleft())
    return ordered


def _build_corruption_fn(
    cfg: Dict[str, Any],
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Resolve the configured corruption to a callable, or ``None`` when disabled."""
    if not bool(cfg.get("enabled", False)):
        return None
    # Local import keeps the streaming module decoupled from the corruption
    # package at import time; only loaded when actually needed.
    from corruptions.corruption_registry import get_corruption

    return get_corruption(str(cfg["type"]), int(cfg["severity"]))


def _iter_folder_frames(
    root_dir: Path,
    resize_wh: Optional[Tuple[int, int]],
    normalize: bool,
    loop: bool,
    max_frames: Optional[int],
    sequence_mode: str,
    corruption_fn: Optional[Callable[[np.ndarray], np.ndarray]],
) -> Iterator[_Frame]:
    samples = list_labeled_images(root_dir)
    if str(sequence_mode).lower() == "interleaved_labels":
        samples = _interleave_by_label(samples)
    if not samples:
        raise ValueError(f"No image files found in runtime input folder: {root_dir}")

    emitted = 0
    frame_id = 0
    while True:
        for sample in samples:
            if max_frames is not None and emitted >= max_frames:
                return

            raw = read_image_bgr(str(sample.path))
            if resize_wh is not None:
                raw = resize_bgr(raw, resize_wh)
            if corruption_fn is not None:
                raw = corruption_fn(raw)
            model_input = normalize_0_1(raw) if normalize else raw

            yield _Frame(
                frame_id=frame_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                path=Path(sample.path),
                raw_image_bgr=raw,
                model_input=model_input,
                label=sample.label,
                defect_type=sample.defect_type,
            )
            frame_id += 1
            emitted += 1

        if not loop:
            return


class _Projection2D:
    def __init__(self, method: str) -> None:
        self.method = method
        self._transformer: Any = None

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        method = str(self.method).lower()
        if method == "umap":
            import umap

            self._transformer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=42,
            )
            return np.asarray(self._transformer.fit_transform(matrix), dtype=np.float32)

        self._transformer = PCA(n_components=2, svd_solver="full")
        return np.asarray(self._transformer.fit_transform(matrix), dtype=np.float32)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self._transformer is None:
            raise RuntimeError("Projection is not fitted.")
        return np.asarray(self._transformer.transform(matrix), dtype=np.float32)


class _EmbeddingProjector:
    def __init__(
        self,
        projection_method: str,
        live_points_limit: int = 250,
        reference_display_limit: int = 1200,
    ) -> None:
        self.projection_method = str(projection_method).lower()
        if self.projection_method == "auto":
            self.projection_method = "pca"
        self.live_points_limit = max(1, int(live_points_limit))
        self.reference_display_limit = max(1, int(reference_display_limit))
        self.enabled = False
        self.embedding_source = "unavailable"
        self.reference_points: List[Dict[str, Any]] = []
        self.live_points: deque[Dict[str, Any]] = deque(maxlen=self.live_points_limit)
        self.axis = {"min_x": -1.0, "max_x": 1.0, "min_y": -1.0, "max_y": 1.0}
        self._projection: Optional[_Projection2D] = None
        self._feature_dim: Optional[int] = None
        self._lock = Lock()

    def fit(
        self,
        inference: FrameInference,
        resize_wh: Optional[Sequence[int]],
        normalize: bool,
    ) -> None:
        train_paths = inference.training_samples()
        embeddings: List[np.ndarray] = []
        paths: List[str] = []

        for path in train_paths:
            image = inference.read_and_preprocess_image(
                Path(path),
                resize_wh=resize_wh,
                normalize=normalize,
            )
            embedding, source = inference.extract_embedding(image)
            arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            self.embedding_source = source
            embeddings.append(arr)
            paths.append(str(path))

        if len(embeddings) < 2:
            return

        matrix = np.stack(embeddings, axis=0).astype(np.float32)
        self._feature_dim = int(matrix.shape[1])
        self._projection = _Projection2D(self.projection_method)
        coords = self._projection.fit_transform(matrix)
        sampled_indices = self._sample_indices(len(paths), self.reference_display_limit)
        self.reference_points = [
            {
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "path": paths[idx],
            }
            for idx in sampled_indices
        ]
        self._expand_axis(coords)
        self.enabled = True

    def project_live(
        self,
        embedding: Optional[np.ndarray],
        score: float,
        threshold: float,
        path: str,
        frame_id: int,
    ) -> None:
        if not self.enabled or self._projection is None or embedding is None:
            return

        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._feature_dim:
            return

        coords = self._projection.transform(vec)
        point = {
            "x": float(coords[0, 0]),
            "y": float(coords[0, 1]),
            "score": float(score),
            "score_ratio": _score_ratio(score, threshold),
            "path": str(path),
            "frame_id": int(frame_id),
        }
        with self._lock:
            self.live_points.append(point)
            self._expand_axis(coords)

    def bootstrap_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "projection": self.projection_method,
                "source": self.embedding_source,
                "reference_points": list(self.reference_points),
                "axis": dict(self.axis),
            }

    def dynamic_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "embedding_enabled": self.enabled,
                "embedding_projection": self.projection_method,
                "embedding_source": self.embedding_source,
                "embedding_live_points": list(self.live_points),
                "embedding_axis": dict(self.axis),
            }

    @staticmethod
    def _sample_indices(count: int, limit: int) -> List[int]:
        if count <= limit:
            return list(range(count))
        return list(np.linspace(0, count - 1, num=limit, dtype=np.int32))

    def _expand_axis(self, coords: np.ndarray) -> None:
        coords = np.asarray(coords, dtype=np.float32)
        min_x = float(np.min(coords[:, 0]))
        max_x = float(np.max(coords[:, 0]))
        min_y = float(np.min(coords[:, 1]))
        max_y = float(np.max(coords[:, 1]))
        if not self.enabled and len(self.live_points) == 0:
            pad_x = max((max_x - min_x) * 0.1, 1e-3)
            pad_y = max((max_y - min_y) * 0.1, 1e-3)
            self.axis = {
                "min_x": min_x - pad_x,
                "max_x": max_x + pad_x,
                "min_y": min_y - pad_y,
                "max_y": max_y + pad_y,
            }
            return

        self.axis["min_x"] = min(self.axis["min_x"], min_x)
        self.axis["max_x"] = max(self.axis["max_x"], max_x)
        self.axis["min_y"] = min(self.axis["min_y"], min_y)
        self.axis["max_y"] = max(self.axis["max_y"], max_y)


class StreamingInputApp:
    """Iterate an image source, run inference per frame, write session outputs."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = resolve_runtime_settings(cfg)
        self.inference = FrameInference(
            run_dir=Path(self.cfg["artifact"]["resolved_run_dir"]),
            model_name=str(self.cfg["artifact"]["model_name"]),
            fit_policy=str(self.cfg["artifact"]["fit_policy"]),
            dataset_path_override=self.cfg["artifact"].get("dataset_path_override"),
            extract_dir_override=self.cfg["artifact"].get("extract_dir_override"),
        )
        self.corruption_cfg = dict(self.cfg.get("corruption", {}))
        self.dataset_name = Path(self.cfg["input"]["root_dir"]).resolve().name
        self.dashboard_cfg = dict(self.cfg.get("dashboard", {}))

    def _artifact_extras(self) -> Dict[str, Any]:
        corr_enabled = bool(self.corruption_cfg.get("enabled", False))
        return {
            "corruption_type": str(self.corruption_cfg.get("type", "")) if corr_enabled else "",
            "severity": int(self.corruption_cfg.get("severity", 0)) if corr_enabled else 0,
            "dataset": self.dataset_name,
        }

    def run(self) -> Path:
        session_dir = self._create_session_dir()
        extras = self._artifact_extras()
        resize_wh: Optional[Tuple[int, int]] = None
        if bool(self.cfg["preprocessing"]["resize"]["enabled"]):
            resize_wh = (
                int(self.cfg["preprocessing"]["resize"]["width"]),
                int(self.cfg["preprocessing"]["resize"]["height"]),
            )
        writer = _SessionWriter(
            session_dir=session_dir,
            model_name=self.inference.model_name,
            benchmark_summary=self.inference.summary,
        )
        metrics = _LiveMetrics(
            session_dir=session_dir,
            input_fps=float(self.cfg["run"]["target_fps"]),
            latency_sla_ms=float(self.cfg["run"]["latency_sla_ms"]),
            score_history_size=int(self.dashboard_cfg["score_history_size"]),
        )
        corruption_fn = _build_corruption_fn(self.corruption_cfg)
        projector = _EmbeddingProjector(
            projection_method=str(self.dashboard_cfg["embedding_projection"]),
            live_points_limit=int(self.dashboard_cfg["embedding_live_points"]),
            reference_display_limit=int(self.dashboard_cfg["embedding_reference_limit"]),
        )
        try:
            projector.fit(
                inference=self.inference,
                resize_wh=resize_wh,
                normalize=bool(self.cfg["preprocessing"]["normalize"]["enabled"]),
            )
        except Exception:
            projector = _EmbeddingProjector(
                projection_method=str(self.dashboard_cfg["embedding_projection"]),
                live_points_limit=int(self.dashboard_cfg["embedding_live_points"]),
                reference_display_limit=int(self.dashboard_cfg["embedding_reference_limit"]),
            )

        status_provider = lambda: self._compose_status(
            metrics=metrics,
            projector=projector,
            extras=extras,
        )
        bootstrap_provider = lambda: self._compose_bootstrap(projector)
        web_server = self._start_web_server(session_dir, status_provider, bootstrap_provider)
        writer.write_status(status_provider())

        target_interval = 0.0
        target_fps = float(self.cfg["run"]["target_fps"])
        if target_fps > 0.0:
            target_interval = 1.0 / target_fps

        try:
            for frame in _iter_folder_frames(
                root_dir=Path(self.cfg["input"]["root_dir"]),
                resize_wh=resize_wh,
                normalize=bool(self.cfg["preprocessing"]["normalize"]["enabled"]),
                loop=bool(self.cfg["input"]["loop"]),
                max_frames=self.cfg["run"]["max_frames"],
                sequence_mode=str(self.cfg["input"]["sequence_mode"]),
                corruption_fn=corruption_fn,
            ):
                started = time.perf_counter()
                metrics.record_frame()

                t0 = time.perf_counter()
                result = self.inference.predict(frame.model_input)
                metrics.record_latency((time.perf_counter() - t0) * 1000.0)

                pred_is_anomaly = int(result.anomaly_score >= self.inference.threshold)
                current_frame_path = writer.save_current_frame(
                    frame=frame,
                    anomaly_map=result.anomaly_map,
                )
                heatmap_path = writer.save_heatmap(
                    frame=frame,
                    anomaly_map=result.anomaly_map,
                    pred_is_anomaly=pred_is_anomaly,
                )
                projector.project_live(
                    embedding=result.embedding,
                    score=float(result.anomaly_score),
                    threshold=self.inference.threshold,
                    path=str(frame.path),
                    frame_id=frame.frame_id,
                )
                row = {
                    "model": self.inference.model_name,
                    "path": str(frame.path),
                    "label": int(frame.label),
                    "defect_type": frame.defect_type,
                    "score": float(result.anomaly_score),
                    "pred_is_anomaly": pred_is_anomaly,
                    "heatmap_path": heatmap_path,
                    "current_frame_path": current_frame_path,
                    "current_frame_version": int(frame.frame_id),
                }
                row.update(extras)
                writer.append_decision(row)
                metrics.record_decision(row)
                writer.write_status(status_provider())
                self._sleep_to_target(started, target_interval)
        finally:
            if web_server is not None:
                web_server.stop()

        return session_dir

    def _create_session_dir(self) -> Path:
        out_dir = Path(self.cfg["run"]["output_dir"])
        session_name = str(self.cfg["run"]["session_name"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{session_name}_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _start_web_server(
        self,
        session_dir: Path,
        status_provider: Callable[[], Dict[str, Any]],
        bootstrap_provider: Callable[[], Dict[str, Any]],
    ) -> Optional[LiveDashboardServer]:
        if not bool(self.cfg["web"]["enabled"]):
            return None
        server = LiveDashboardServer(
            host=str(self.cfg["web"]["host"]),
            port=int(self.cfg["web"]["port"]),
            session_dir=session_dir,
            status_provider=status_provider,
            bootstrap_provider=bootstrap_provider,
        )
        server.start()
        return server

    def _compose_status(
        self,
        metrics: _LiveMetrics,
        projector: _EmbeddingProjector,
        extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        status = metrics.snapshot(
            model_name=self.inference.model_name,
            threshold=self.inference.threshold,
            extras=extras,
        )
        status.update(projector.dynamic_payload())
        return status

    def _compose_bootstrap(self, projector: _EmbeddingProjector) -> Dict[str, Any]:
        payload = projector.bootstrap_payload()
        payload["refresh_ms"] = int(self.cfg["web"]["refresh_ms"])
        return payload

    @staticmethod
    def _sleep_to_target(started: float, target_interval: float) -> None:
        if target_interval <= 0.0:
            return
        elapsed = time.perf_counter() - started
        remaining = target_interval - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
