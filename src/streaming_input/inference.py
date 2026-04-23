"""Per-frame inference wrapper used by the streaming pipeline.

Loads a trained anomaly detection model from a previous benchmark run
directory and exposes a single ``predict(image)`` call that returns the
streaming triple ``{anomaly_score, anomaly_map, embedding}`` mandated by
PLAN.md §1.3.

Folds the responsibilities of the previous artifacts.py + model_registry.py
+ predictor.py into one self-contained module so the package matches the
four-file layout prescribed by PLAN.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmark_AD.data import apply_dataset_split, resolve_dataset_labeled
from benchmark_AD.models import build_model


@dataclass
class FrameInferenceResult:
    """Streaming-frame output: score, optional anomaly heatmap, optional embedding."""

    anomaly_score: float
    anomaly_map: Optional[np.ndarray]
    embedding: Optional[np.ndarray]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "anomaly_map": self.anomaly_map,
            "embedding": self.embedding,
        }


def _read_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark summary: {summary_path}")
    return payload


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = summary.get("models", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark_summary.json has no model rows.")
    return [row for row in rows if isinstance(row, dict)]


def _select_row(rows: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("model")) == model_name:
            return row
    raise ValueError(f"Model '{model_name}' not present in benchmark_summary.json.")


def _resolve_model_cfg(
    summary: Dict[str, Any],
    model_row: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    cfg = model_row.get("model_cfg")
    if isinstance(cfg, dict):
        return dict(cfg)
    base = summary.get("model")
    if isinstance(base, dict):
        cfg = dict(base)
        cfg["name"] = model_name
        return cfg
    raise ValueError(f"Could not resolve model config for '{model_name}'.")


class FrameInference:
    """Wrap a trained model so the streaming loop can call it one image at a time."""

    def __init__(
        self,
        run_dir: Path,
        model_name: str,
        fit_policy: str = "auto",
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.model_name = str(model_name)
        self.fit_policy = str(fit_policy).lower()

        summary = _read_summary(self.run_dir)
        row = _select_row(_summary_rows(summary), self.model_name)
        self.threshold = float(row.get("threshold_used", row.get("threshold", 0.5)))
        self.model_cfg = _resolve_model_cfg(summary, row, self.model_name)
        self.runtime_cfg = dict(summary.get("runtime", {}))
        self.dataset_cfg = dict(summary.get("dataset", {}))

        self.model = build_model(self.model_cfg, self.runtime_cfg)
        if hasattr(self.model, "threshold"):
            self.model.threshold = self.threshold
        self._prepare_model()

    def predict(self, image: np.ndarray) -> FrameInferenceResult:
        """Return the streaming triple for one preprocessed BGR image in [0, 1]."""
        out = self.model.predict(image)
        embedding: Optional[np.ndarray] = None
        if hasattr(self.model, "get_embedding"):
            embedding = self.model.get_embedding(image)
        return FrameInferenceResult(
            anomaly_score=float(out.score),
            anomaly_map=out.heatmap,
            embedding=embedding,
        )

    def _prepare_model(self) -> None:
        already_fitted = bool(getattr(self.model, "_is_fitted", True))
        wants_fit = self.fit_policy == "historical_fit" or (
            self.fit_policy != "skip_fit" and not already_fitted
        )
        if not wants_fit:
            if not already_fitted:
                raise RuntimeError(
                    "Runtime model is not ready. Re-run with fit_policy='historical_fit' "
                    "or point at a warm-started benchmark run."
                )
            return

        samples = resolve_dataset_labeled(
            self.dataset_cfg["source_type"],
            self.dataset_cfg["path"],
            self.dataset_cfg["extract_dir"],
        )
        split = apply_dataset_split(samples, self.dataset_cfg.get("split", {}), fallback_seed=42)
        train_paths = [s.path for s in split.train]
        self.model.fit(
            train_paths,
            fit_context={
                "train_samples": split.train,
                "val_samples": split.val,
                "test_samples": split.test,
            },
        )
        if hasattr(self.model, "threshold"):
            self.model.threshold = self.threshold
