from __future__ import annotations

import numpy as np
import cv2

from .base import BaseModel, ModelOutput


class DummyDistanceModel(BaseModel):
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def predict(self, x: np.ndarray) -> ModelOutput:
        # x expected float32 in [0,1], convert to grayscale
        gray = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())  # higher = sharper
        # Convert to a bounded "anomaly score" proxy
        score = 1.0 / (1.0 + np.exp(-(sharpness - 100.0) / 50.0))
        is_anomaly = score >= self.threshold
        return ModelOutput(score=score, is_anomaly=is_anomaly)
