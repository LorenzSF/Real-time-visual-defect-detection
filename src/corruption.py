import dataclasses
import random
from typing import Callable, Iterator

import numpy as np
from scipy.ndimage import uniform_filter1d

from .schemas import CorruptionConfig, Frame


def apply_corruption(
    stream: Iterator[Frame], cfg: CorruptionConfig
) -> Iterator[Frame]:
    """Wrap a `Frame` stream and yield corrupted frames.

    If `cfg.enabled` is false the stream passes through unchanged. Otherwise
    each spec is evaluated independently per frame: with probability
    `spec.probability` the corresponding kernel is applied to the running
    image, then the next spec is considered. This lets multiple corruptions
    compose on the same frame (matching the ImageNet-C convention).
    """
    if not cfg.enabled or not cfg.specs:
        yield from stream
        return

    for spec in cfg.specs:
        if spec.kind not in _CORRUPTIONS:
            raise ValueError(
                f"unknown corruption kind '{spec.kind}' "
                f"(supported: {sorted(_CORRUPTIONS)})"
            )

    for frame in stream:
        img = frame.image
        for spec in cfg.specs:
            if random.random() < spec.probability:
                img = _CORRUPTIONS[spec.kind](img, spec.severity)
        if img is frame.image:
            yield frame
        else:
            yield dataclasses.replace(frame, image=img)


def _gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = [0.04, 0.06, 0.08][severity - 1] * 255.0
    noise = np.random.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _shot_noise(img: np.ndarray, severity: int) -> np.ndarray:
    lam = [60, 25, 12][severity - 1]
    x = img.astype(np.float32) / 255.0
    return np.clip(np.random.poisson(x * lam) / lam * 255.0, 0, 255).astype(np.uint8)


def _motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = [3, 5, 7][severity - 1]
    n = radius * 2 + 1
    blurred = uniform_filter1d(img.astype(np.float32), size=n, axis=1, mode="nearest")
    return np.clip(blurred, 0, 255).astype(np.uint8)


_CORRUPTIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "motion_blur": _motion_blur,
}
