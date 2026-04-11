from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

@dataclass
class RuntimeArtifact:
    run_dir: Path
    model_name: str
    model_cfg: dict[str, Any]
    runtime_cfg: dict[str, Any]
    dataset_cfg: dict[str, Any]
    preprocessing_cfg: dict[str, Any]
    threshold: float
    validation_predictions_path: Optional[Path] = None


@dataclass
class FramePacket:
    frame_id: int
    timestamp_utc: str
    path: Path
    raw_image_bgr: np.ndarray
    model_input: np.ndarray
    label: int = -1
    defect_type: Optional[str] = None


@dataclass
class ModelPrediction:
    model_name: str
    score: float
    pred_is_anomaly: int
    heatmap: Optional[np.ndarray] = None


@dataclass
class DecisionRecord:
    model: str
    path: str
    label: int
    defect_type: Optional[str]
    score: float
    pred_is_anomaly: int
    heatmap_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "path": self.path,
            "label": self.label,
            "defect_type": self.defect_type,
            "score": self.score,
            "pred_is_anomaly": self.pred_is_anomaly,
            "heatmap_path": self.heatmap_path,
        }


@dataclass
class LiveSnapshot:
    session_dir: str
    active_model: str
    frames_seen: int = 0
    decisions_emitted: int = 0
    fail_count: int = 0
    input_fps: float = 0.0
    processed_fps: float = 0.0
    decision_fps: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    latency_sla_ms: float = 0.0
    threshold: float = 0.0
    recent_decisions: list[dict[str, Any]] = field(default_factory=list)
    recent_fails: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_dir": self.session_dir,
            "active_model": self.active_model,
            "frames_seen": self.frames_seen,
            "decisions_emitted": self.decisions_emitted,
            "fail_count": self.fail_count,
            "input_fps": self.input_fps,
            "processed_fps": self.processed_fps,
            "decision_fps": self.decision_fps,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "latency_sla_ms": self.latency_sla_ms,
            "threshold": self.threshold,
            "recent_decisions": self.recent_decisions,
            "recent_fails": self.recent_fails,
        }
