"""Streaming inference loop for real-time visual defect detection.

Iterates over an image source (folder or simulated camera), runs per-frame
inference, and writes the streaming-session outputs:
``predictions.json`` + ``live_status.json`` (+ ``heatmaps/`` overlays) per
PLAN.md §1.3.

Folds the previous input_handler.py + decision_engine.py + live_metrics.py
+ report_generator.py into one module so the package matches the four-file
layout prescribed by PLAN.md.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

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
    """Streams predictions.json, live_status.json, and heatmap overlays to disk."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.heatmaps_dir = session_dir / "heatmaps"
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = session_dir / "predictions.json"
        self.status_path = session_dir / "live_status.json"
        self.predictions_path.write_text("[]", encoding="utf-8")
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
        heatmap = np.asarray(anomaly_map, dtype=np.float32)
        if heatmap.ndim != 2:
            return None

        h, w = frame.raw_image_bgr.shape[:2]
        resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        clipped = np.clip(resized, 0.0, 1.0)
        colored = cv2.applyColorMap(
            (clipped * 255.0).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(frame.raw_image_bgr, 0.55, colored, 0.45, 0.0)
        relative = Path("heatmaps") / f"frame_{frame.frame_id:06d}.png"
        cv2.imwrite(str(self.session_dir / relative), overlay)
        return relative.as_posix()


class _LiveMetrics:
    """Rolling latency + decision counters that feed the live_status snapshot."""

    def __init__(
        self,
        session_dir: Path,
        input_fps: float,
        latency_sla_ms: float,
    ) -> None:
        self.session_dir = session_dir
        self.input_fps = float(input_fps)
        self.latency_sla_ms = float(latency_sla_ms)
        self.started_at = time.perf_counter()
        self.frames_seen = 0
        self.decisions_emitted = 0
        self.fail_count = 0
        self.latencies_ms: deque[float] = deque(maxlen=2048)
        self.recent_decisions: deque[Dict[str, Any]] = deque(maxlen=12)
        self.recent_fails: deque[Dict[str, Any]] = deque(maxlen=8)
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
            self.recent_decisions.appendleft(row)
            if int(row.get("pred_is_anomaly", 0)) == 1:
                self.fail_count += 1
                self.recent_fails.appendleft(row)

    def snapshot(self, model_name: str, threshold: float) -> Dict[str, Any]:
        with self._lock:
            uptime = max(time.perf_counter() - self.started_at, 1e-6)
            latencies = np.asarray(self.latencies_ms, dtype=np.float32)
            mean_latency = float(np.mean(latencies)) if latencies.size else 0.0
            p95_latency = float(np.percentile(latencies, 95)) if latencies.size else 0.0
            return {
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
                "recent_decisions": list(self.recent_decisions),
                "recent_fails": list(self.recent_fails),
            }


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


class StreamingInputApp:
    """Iterate an image source, run inference per frame, write session outputs."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = resolve_runtime_settings(cfg)
        self.inference = FrameInference(
            run_dir=Path(self.cfg["artifact"]["resolved_run_dir"]),
            model_name=str(self.cfg["artifact"]["model_name"]),
            fit_policy=str(self.cfg["artifact"]["fit_policy"]),
        )

    def run(self) -> Path:
        session_dir = self._create_session_dir()
        writer = _SessionWriter(session_dir=session_dir)
        metrics = _LiveMetrics(
            session_dir=session_dir,
            input_fps=float(self.cfg["run"]["target_fps"]),
            latency_sla_ms=float(self.cfg["run"]["latency_sla_ms"]),
        )

        resize_wh: Optional[Tuple[int, int]] = None
        if bool(self.cfg["preprocessing"]["resize"]["enabled"]):
            resize_wh = (
                int(self.cfg["preprocessing"]["resize"]["width"]),
                int(self.cfg["preprocessing"]["resize"]["height"]),
            )
        corruption_fn = _build_corruption_fn(dict(self.cfg.get("corruption", {})))

        status_provider = lambda: metrics.snapshot(
            model_name=self.inference.model_name,
            threshold=self.inference.threshold,
        )
        web_server = self._start_web_server(session_dir, status_provider)
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
                heatmap_path = writer.save_heatmap(
                    frame=frame,
                    anomaly_map=result.anomaly_map,
                    pred_is_anomaly=pred_is_anomaly,
                )
                row = {
                    "model": self.inference.model_name,
                    "path": str(frame.path),
                    "label": int(frame.label),
                    "defect_type": frame.defect_type,
                    "score": float(result.anomaly_score),
                    "pred_is_anomaly": pred_is_anomaly,
                    "heatmap_path": heatmap_path,
                }
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
    ) -> Optional[LiveDashboardServer]:
        if not bool(self.cfg["web"]["enabled"]):
            return None
        server = LiveDashboardServer(
            host=str(self.cfg["web"]["host"]),
            port=int(self.cfg["web"]["port"]),
            session_dir=session_dir,
            status_provider=status_provider,
        )
        server.start()
        return server

    @staticmethod
    def _sleep_to_target(started: float, target_interval: float) -> None:
        if target_interval <= 0.0:
            return
        elapsed = time.perf_counter() - started
        remaining = target_interval - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
