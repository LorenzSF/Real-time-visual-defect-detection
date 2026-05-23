from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, List

import numpy as np

from scipy.stats import genpareto

from src.corruption import apply_corruption
from src.metrics import (
    FrameLogger,
    OnlineMetrics,
    calibrate_offline_threshold,
    offline_metrics,
)
from src.models import build_model
from src.schemas import DatasetEntry, Frame, RunConfig
from src.stream import (
    build_offline_split,
    build_stream,
    build_warmup_stream,
    frames_from_entries,
    warmup,
)
from src.visualization import StreamVisualizer, prediction_projection_vector


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_run_dir(output_dir: str, experiment_name: str) -> Path:
    run_dir = Path(output_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_report(report: dict[str, Any], run_dir: Path) -> Path:
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(_jsonify(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def _build_model_for_run(cfg: RunConfig) -> tuple[Any, Any | None]:
    set_seeds(cfg.seed)
    model = build_model(cfg.model, cfg.warmup.fit_epochs)
    torch = _reset_peak_vram(cfg.model.device)
    return model, torch


def _reset_peak_vram(model_device: str) -> Any | None:
    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available() and model_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    return torch


def _peak_vram_mb(torch: Any | None, model_device: str) -> float:
    if torch is None:
        return 0.0
    if torch.cuda.is_available() and model_device.startswith("cuda"):
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    return 0.0


def _finalize_report(
    *,
    report: dict[str, Any],
    cfg: RunConfig,
    run_dir: Path,
    experiment_name: str,
    mode: str,
    runtime: dict[str, Any],
    threshold: dict[str, Any],
    evaluation: dict[str, Any],
) -> Path:
    report["runtime"] = runtime
    report["threshold"] = threshold
    report["evaluation"] = evaluation
    report["run"] = {
        "mode": mode,
        "experiment_name": experiment_name,
        "seed": cfg.seed,
    }
    report["hardware"] = _collect_hardware_info(cfg.model.device)
    report["stream"] = dataclasses.asdict(cfg.stream)
    report["warmup"] = dataclasses.asdict(cfg.warmup)
    report["model"] = dataclasses.asdict(cfg.model)
    report["corruption"] = dataclasses.asdict(cfg.corruption)
    if mode == "offline":
        report["offline"] = dataclasses.asdict(cfg.offline)
    return save_report(report, run_dir)


def _derive_experiment_name(cfg: RunConfig) -> str:
    parts = [cfg.model.name, cfg.stream.dataset]
    input_name = Path(cfg.stream.input_path).name
    if input_name:
        parts.append(input_name)
    if cfg.corruption.enabled and cfg.corruption.specs:
        for spec in cfg.corruption.specs:
            parts.append(f"{spec.kind}_s{spec.severity}")
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)


def _jsonify(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonify(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _collect_hardware_info(model_device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
        },
        "accelerator": {
            "requested_device": model_device,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
        },
    }

    try:
        import torch
    except ImportError:
        info["accelerator"]["torch_available"] = False
        return info

    info["accelerator"]["torch_available"] = True
    info["accelerator"]["torch_version"] = torch.__version__

    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_devices: list[dict[str, Any]] = []
        if cuda_available:
            for device_index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_index)
                cuda_devices.append(
                    {
                        "index": device_index,
                        "name": props.name,
                        "total_memory_mb": float(
                            props.total_memory / (1024.0 * 1024.0)
                        ),
                        "compute_capability": [
                            int(props.major),
                            int(props.minor),
                        ],
                        "multiprocessor_count": int(props.multi_processor_count),
                    }
                )
        info["accelerator"].update(
            {
                "cuda_available": cuda_available,
                "cuda_device_count": len(cuda_devices),
                "cuda_devices": cuda_devices,
            }
        )
    except (RuntimeError, AttributeError) as exc:
        info["accelerator"]["probe_error"] = f"{type(exc).__name__}: {exc}"

    return info


def _calibrate_threshold(
    cfg: RunConfig,
    scores: List[float],
) -> tuple[float, dict[str, Any]]:
    mode = cfg.metrics.threshold_mode
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.size == 0:
        raise RuntimeError(
            f"{mode} calibration requires at least one finite calibration score"
        )

    if mode == "max_score_ok":
        threshold = float(np.max(scores_arr))
        return threshold, {
            "mode": mode,
            "threshold": threshold,
            "n_calibration_scores": int(scores_arr.size),
        }

    if mode == "pot":
        threshold, report = _pot_threshold(scores_arr, cfg.metrics.pot_risk)
        report.update({"mode": mode, "threshold": threshold})
        return threshold, report

    raise ValueError(f"unknown threshold_mode {mode!r}")


def _pot_threshold(
    scores: np.ndarray, pot_risk: float
) -> tuple[float, dict[str, Any]]:
    """Siffer et al. 2017, KDD, §3.2. Fit a Generalized Pareto to the upper 
    tail of the calibration scores and derive the threshold at target risk
    `pot_risk` (false positive rate)."""
    init_q = 0.98
    u = float(np.quantile(scores, init_q))
    tail = scores[scores > u] - u
    if tail.size < 10:
        raise RuntimeError(
            f"pot calibration: only {tail.size} exceedances above q={init_q} "
            f"(need >= 10); increase metrics.calibration_steps"
        )
    ksi, _, sigma = genpareto.fit(tail, floc=0.0)
    n, n_u = int(scores.size), int(tail.size)
    ratio = (n / n_u) * pot_risk
    if abs(ksi) < 1e-9:
        threshold = u - float(sigma) * math.log(ratio)
    else:
        threshold = u + (float(sigma) / float(ksi)) * (ratio ** (-float(ksi)) - 1.0)
    return float(threshold), {
        "pot_risk": float(pot_risk),
        "pot_init_quantile": init_q,
        "pot_u": u,
        "pot_ksi": float(ksi),
        "pot_sigma": float(sigma),
        "pot_n_tail": n_u,
        "n_calibration_scores": n,
    }


def _collect_warmup_projection_vectors(
    model, warmup_frames: List[Frame], enabled: bool
) -> "np.ndarray | None":
    """Re-score warmup frames to harvest vectors for dashboard PCA.

    The dashboard's reference cloud is the StandardScaler + PCA(2)
    projection of these vectors. The model is already fitted at this point;
    this second pass exists because vectors depend on `predict` outputs.
    Returns None when the dashboard is disabled or vector dimensions are
    inconsistent across frames.
    """
    if not enabled:
        return None
    vecs: list[np.ndarray] = []
    for frame in warmup_frames:
        pred = model.predict(frame)
        vec = prediction_projection_vector(pred)
        if vec is None:
            continue
        vecs.append(vec)
    if len(vecs) < 2:
        return None
    sizes = {v.size for v in vecs}
    if len(sizes) != 1:
        return None
    return np.stack(vecs, axis=0)


def run_streaming(cfg: RunConfig) -> None:
    experiment_name = _derive_experiment_name(cfg)
    run_dir = build_run_dir(cfg.output_dir, experiment_name)
    print(f"[main] experiment={experiment_name} seed={cfg.seed}")

    model, torch = _build_model_for_run(cfg)

    print("[main] warming up model...")
    cold_start_t0 = time.perf_counter()
    set_seeds(cfg.seed)
    warmup_stream = build_warmup_stream(cfg.stream, cfg.warmup.warmup_steps)
    warmup_frames = warmup(model, warmup_stream, cfg.warmup)
    cold_start_s = time.perf_counter() - cold_start_t0

    active_threshold = float(cfg.metrics.initial_threshold)
    threshold_report: dict[str, Any] = {
        "mode": cfg.metrics.threshold_mode,
        "initial_threshold": active_threshold,
        "status": "calibrating",
        "calibration_steps": cfg.metrics.calibration_steps,
        "switch_frame": None,
    }
    print(
        "[main] threshold initializing: "
        f"mode={cfg.metrics.threshold_mode} initial={active_threshold:.6f} "
        f"calibration_steps={cfg.metrics.calibration_steps}"
    )

    resolved_metrics_cfg = dataclasses.replace(
        cfg.metrics, threshold_value=active_threshold
    )
    set_seeds(cfg.seed)
    stream = build_stream(
        cfg.stream,
        cfg.warmup.warmup_steps,
        cfg.metrics.calibration_steps,
    )

    metrics = OnlineMetrics(resolved_metrics_cfg)
    warmup_projection_vectors = _collect_warmup_projection_vectors(
        model, warmup_frames, cfg.visualization.dashboard_enabled
    )
    viz = StreamVisualizer(
        cfg.visualization,
        run_dir,
        active_threshold,
        cfg.model.name,
        warmup_projection_vectors,
    )

    corrupted = apply_corruption(stream, cfg.corruption)
    calibration_scores: list[float] = []
    calibration_seen = 0
    threshold_ready = False
    evaluation_start_frame: int | None = None

    print("[main] starting streaming inference loop")
    with FrameLogger(run_dir / "frames.jsonl") as frames_log:
        for frame in warmup_frames:
            frames_log.write_warmup(frame)

        for frame in corrupted:
            pred = model.predict(frame)

            if not threshold_ready:
                calibration_seen += 1
                score = float(pred.score)
                if math.isfinite(score):
                    calibration_scores.append(score)
                frames_log.write_threshold_calibration(frame, pred)
                if calibration_seen >= cfg.metrics.calibration_steps:
                    active_threshold, threshold_report = _calibrate_threshold(
                        cfg, calibration_scores
                    )
                    threshold_report.update(
                        {
                            "status": "calibrated",
                            "initial_threshold": float(cfg.metrics.initial_threshold),
                            "calibration_steps": cfg.metrics.calibration_steps,
                            "calibration_seen": calibration_seen,
                            "n_calibration_scores": len(calibration_scores),
                            "switch_frame": int(frame.index) + 1,
                        }
                    )
                    metrics.set_threshold(active_threshold)
                    viz.set_threshold(active_threshold)
                    threshold_ready = True
                    evaluation_start_frame = int(frame.index) + 1
                    print(
                        "[main] threshold switched: "
                        f"mode={cfg.metrics.threshold_mode} "
                        f"value={active_threshold:.6f} "
                        f"after_frame={frame.index}"
                    )
                continue

            threshold_used = active_threshold
            metrics.update(frame, pred, threshold_used)
            viz.render(frame, pred, metrics.snapshot())
            frames_log.write(frame, pred, threshold_used)
            if frame.index % cfg.log_every == 0:
                print(f"[step {frame.index}] {metrics.snapshot()}")

    if not threshold_ready:
        threshold_report.update(
            {
                "status": "incomplete",
                "calibration_seen": calibration_seen,
                "n_calibration_scores": len(calibration_scores),
                "threshold": active_threshold,
            }
        )

    report = metrics.finalize()
    report_path = _finalize_report(
        report=report,
        cfg=cfg,
        run_dir=run_dir,
        experiment_name=experiment_name,
        mode="streaming",
        runtime={
            "cold_start_s": cold_start_s,
            "peak_vram_mb": _peak_vram_mb(torch, cfg.model.device),
        },
        threshold=threshold_report,
        evaluation={
            "metrics_start": "after_threshold_calibration",
            "warmup_frames_excluded": len(warmup_frames),
            "threshold_calibration_frames_excluded": calibration_seen,
            "starts_at_stream_frame": evaluation_start_frame,
            "n_evaluation_frames": report["n_seen"],
        },
    )
    viz.close()
    print(f"[main] done: {report_path}")


_OFFLINE_MODELS = {
    "patchcore",
    "anomalib_patchcore",
    "padim",
    "anomalib_padim",
    "subspacead",
    "stfpm",
    "anomalib_stfpm",
    "csflow",
    "anomalib_csflow",
    "draem",
    "anomalib_draem",
    "rd4ad",
    "reverse_distillation",
}


def run_offline(cfg: RunConfig) -> None:
    if cfg.corruption.enabled:
        raise ValueError("offline mode does not apply corruptions; disable corruption.enabled")

    model_name = cfg.model.name.lower()
    if model_name not in _OFFLINE_MODELS:
        raise ValueError(
            f"offline mode supports only experiment models, got {cfg.model.name!r}"
        )

    input_name = Path(cfg.stream.input_path).name
    experiment_name = "_".join(
        [
            cfg.offline.experiment_name,
            cfg.model.name,
            cfg.stream.dataset,
            input_name,
            time.strftime("%Y%m%d-%H%M%S"),
        ]
    )
    run_dir = build_run_dir(cfg.output_dir, experiment_name)
    print(f"[main] mode=offline experiment={experiment_name} seed={cfg.seed}")

    set_seeds(cfg.seed)
    split = build_offline_split(cfg.stream, cfg.offline.split, cfg.seed)
    print(
        "[main] offline split: "
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)}"
    )

    model, torch = _build_model_for_run(cfg)

    fit_t0 = time.perf_counter()
    train_frames = list(frames_from_entries(split.train))
    if not hasattr(model, "fit_warmup"):
        raise TypeError(f"model {type(model).__name__} has no fit_warmup() method")
    model.fit_warmup(train_frames)
    fit_seconds = time.perf_counter() - fit_t0

    with FrameLogger(run_dir / "frames.jsonl") as frames_log:
        for frame in train_frames:
            frames_log.write_warmup(frame)

        val_rows, val_predict_seconds = _predict_entries(model, split.val)
        val_scores, val_labels = _score_arrays(val_rows)
        threshold, threshold_report = calibrate_offline_threshold(
            val_scores,
            val_labels,
            cfg.offline.threshold.mode,
            cfg.offline.threshold.target_fpr,
        )
        _add_threshold_to_rows(val_rows, threshold)
        _write_prediction_rows(run_dir / "validation_predictions.jsonl", val_rows)

        test_rows, test_predict_seconds = _predict_entries(
            model, split.test, frames_log, threshold
        )
        _write_prediction_rows(run_dir / "predictions.jsonl", test_rows)

    test_scores, test_labels = _score_arrays(test_rows)
    report = offline_metrics(test_scores, test_labels, threshold)
    report["n_seen"] = len(test_rows)
    report["n_anomalies"] = report["n_predicted_anomalies"]
    report["labels_available"] = True
    report["threshold_mode"] = cfg.offline.threshold.mode
    report["threshold_used"] = float(threshold)
    latencies = np.asarray([row["latency_ms"] for row in test_rows], dtype=np.float64)
    report["mean_latency_ms"] = float(np.mean(latencies)) if latencies.size else 0.0
    report["p95_latency_ms"] = float(np.quantile(latencies, 0.95)) if latencies.size else 0.0
    report["throughput_fps"] = (
        len(test_rows) / test_predict_seconds if test_predict_seconds > 0 else 0.0
    )
    report_path = _finalize_report(
        report=report,
        cfg=cfg,
        run_dir=run_dir,
        experiment_name=experiment_name,
        mode="offline",
        runtime={
            "fit_seconds": fit_seconds,
            "val_predict_seconds": val_predict_seconds,
            "test_predict_seconds": test_predict_seconds,
            "peak_vram_mb": _peak_vram_mb(torch, cfg.model.device),
        },
        threshold=threshold_report,
        evaluation={
            "mode": "offline_train_val_test",
            "train_samples": len(split.train),
            "val_samples": len(split.val),
            "test_samples": len(split.test),
            "n_evaluation_frames": len(test_rows),
        },
    )
    print(f"[main] done: {report_path}")


def _predict_entries(
    model,
    entries: list[DatasetEntry],
    frames_log: FrameLogger | None = None,
    threshold: float | None = None,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for frame in frames_from_entries(entries):
        pred = model.predict(frame)
        score = float(pred.score)
        row = {
            "idx": int(frame.index),
            "image_id": frame.image_id,
            "path": frame.source_id,
            "label": int(frame.label),
            "score": score if math.isfinite(score) else None,
            "latency_ms": float(pred.latency_ms),
        }
        if threshold is not None:
            row["threshold_used"] = float(threshold)
            row["pred_label"] = int(score >= float(threshold)) if math.isfinite(score) else -1
        rows.append(row)
        if frames_log is not None and threshold is not None:
            frames_log.write_prediction(frame, pred, threshold, "evaluation")
    return rows, time.perf_counter() - t0


def _score_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    scores = [
        float(row["score"]) if row["score"] is not None else float("nan")
        for row in rows
    ]
    labels = [int(row["label"]) for row in rows]
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _add_threshold_to_rows(rows: list[dict[str, Any]], threshold: float) -> None:
    for row in rows:
        score = row["score"]
        row["threshold_used"] = float(threshold)
        row["pred_label"] = (
            int(float(score) >= float(threshold)) if score is not None else -1
        )


def _write_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonify(row), sort_keys=True) + "\n")


def main() -> None:
    cfg = RunConfig.from_yaml("config.yaml")
    if cfg.run.mode == "streaming":
        run_streaming(cfg)
    elif cfg.run.mode == "offline":
        run_offline(cfg)
    else:
        raise ValueError(f"unknown run mode {cfg.run.mode!r}")


if __name__ == "__main__":
    main()
