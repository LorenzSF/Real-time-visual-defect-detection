from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ModelOutput:
    score: float
    is_anomaly: bool


class BaseModel:
    def predict(self, x: np.ndarray) -> ModelOutput:
        raise NotImplementedError
