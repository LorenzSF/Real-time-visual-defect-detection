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
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from benchmark_AD.data import (
    SplitResult,
    apply_dataset_split,
    normalize_0_1,
    read_image_bgr,
    resolve_dataset_labeled,
    resize as resize_bgr,
)
from benchmark_AD.models import build_model

_FALLBACK_EMBED_SIDE = 32


@dataclass
class FrameInferenceResult:
    """Streaming-frame output: score, optional anomaly heatmap, optional embedding."""

    anomaly_score: float
    anomaly_map: Optional[np.ndarray]
    embedding: Optional[np.ndarray]
    embedding_source: str = "unavailable"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "anomaly_map": self.anomaly_map,
            "embedding": self.embedding,
            "embedding_source": self.embedding_source,
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
        self.summary = dict(summary)
        row = _select_row(_summary_rows(summary), self.model_name)
        self.threshold = float(row.get("threshold_used", row.get("threshold", 0.5)))
        self.model_cfg = _resolve_model_cfg(summary, row, self.model_name)
        self.runtime_cfg = dict(summary.get("runtime", {}))
        self.dataset_cfg = dict(summary.get("dataset", {}))
        self.dataset_split = self._resolve_dataset_split()
        self.embedding_source = "unavailable"

        self.model = build_model(self.model_cfg, self.runtime_cfg)
        if hasattr(self.model, "threshold"):
            self.model.threshold = self.threshold
        self._prepare_model()

    def predict(self, image: np.ndarray) -> FrameInferenceResult:
        """Return the streaming triple for one preprocessed BGR image in [0, 1]."""
        out = self.model.predict(image)
        embedding, source = self.extract_embedding(image)
        return FrameInferenceResult(
            anomaly_score=float(out.score),
            anomaly_map=out.heatmap,
            embedding=embedding,
            embedding_source=source,
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

        train_paths = [s.path for s in self.dataset_split.train]
        self.model.fit(
            train_paths,
            fit_context={
                "train_samples": self.dataset_split.train,
                "val_samples": self.dataset_split.val,
                "test_samples": self.dataset_split.test,
            },
        )
        if hasattr(self.model, "threshold"):
            self.model.threshold = self.threshold

    def _resolve_dataset_split(self) -> SplitResult:
        samples = resolve_dataset_labeled(
            self.dataset_cfg["source_type"],
            self.dataset_cfg["path"],
            self.dataset_cfg["extract_dir"],
            dataset_format=self.dataset_cfg.get("format"),
            cameras=self.dataset_cfg.get("cameras"),
        )
        return apply_dataset_split(
            samples,
            self.dataset_cfg.get("split", {}),
            fallback_seed=42,
        )

    def training_samples(self) -> List[Path]:
        return [Path(sample.path) for sample in self.dataset_split.train]

    def preprocess_image(
        self,
        image_bgr: np.ndarray,
        resize_wh: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        out = image_bgr
        if resize_wh is not None:
            out = resize_bgr(out, (int(resize_wh[0]), int(resize_wh[1])))
        if normalize:
            out = normalize_0_1(out)
        return out

    def read_and_preprocess_image(
        self,
        path: Path,
        resize_wh: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        raw = read_image_bgr(str(path))
        return self.preprocess_image(raw, resize_wh=resize_wh, normalize=normalize)

    def extract_embedding(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        if hasattr(self.model, "get_embedding"):
            emb = self.model.get_embedding(image)
            if emb is not None:
                arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    self.embedding_source = "model"
                    return arr, self.embedding_source

        fallback = self._fallback_embedding(image)
        self.embedding_source = "image_fallback"
        return fallback, self.embedding_source

    @staticmethod
    def _fallback_embedding(image: np.ndarray) -> np.ndarray:
        rgb = np.asarray(image[..., ::-1], dtype=np.float32)
        small = cv2.resize(
            rgb,
            (_FALLBACK_EMBED_SIDE, _FALLBACK_EMBED_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        return small.reshape(-1).astype(np.float32)
