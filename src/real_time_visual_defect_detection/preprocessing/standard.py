from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def normalize_0_1(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0
