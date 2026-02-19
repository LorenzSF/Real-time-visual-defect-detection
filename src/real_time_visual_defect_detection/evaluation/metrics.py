from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


@dataclass
class MetricsResult:
    precision: float
    recall: float
    f1: float
    accuracy: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    return MetricsResult(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
    )
