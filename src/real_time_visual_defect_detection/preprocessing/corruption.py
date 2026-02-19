from __future__ import annotations

from typing import Any, Dict
import numpy as np
import cv2


def apply_corruption(img: np.ndarray, corruption_type: str, params: Dict[str, Any]) -> np.ndarray:
    if corruption_type == "gaussian_noise":
        sigma = float(params.get("sigma", 10.0))
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(img.dtype)

    if corruption_type == "gaussian_blur":
        ksize = int(params.get("ksize", 5))
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    if corruption_type == "resolution_reduction":
        scale = float(params.get("scale", 0.5))
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    raise ValueError(f"Unknown corruption_type: {corruption_type}")
