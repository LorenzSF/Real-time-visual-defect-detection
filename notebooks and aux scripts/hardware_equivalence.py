"""Cross-GPU latency normalization for the analysis notebooks.

Run outputs in this repo come from a mix of GPUs (A100, H100, L4, T4,
RTX PRO 6000 Blackwell). The analysis notebooks must compare per-image
latency across runs, so every measurement is rescaled to what it would
have been on the RTX PRO 6000 Blackwell reference.

Factors are empirical, derived from same-model + same-image-size +
same-batch-size pairs found across Experiments 1, 1_with_corruptions,
and 2:

    f(X) = ms(W' on RTX PRO 6000 B.) / ms(W' on X)

    csflow 512/bs8 : T4=143.91, L4=94.10, A100=77.18
    draem  512/bs8 : H100=18.04, RTX PRO 6000 B.=20.01
    padim  512/bs16: A100=11.08, L4=14.00, RTX PRO 6000 B.=4.72
    stfpm  512/bs8 : H100=7.46, L4=13.27

Chains used when the direct edge to the reference is missing:
    f(A100) = padim direct
    f(H100) = draem direct
    f(L4)   = f(A100) * (ms_A100/ms_L4)|padim   (csflow chain agrees)
    f(T4)   = f(A100) * (ms_A100/ms_T4)|csflow  (L4 chain agrees)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REFERENCE_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

GPU_FACTORS: dict[str, float] = {
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1.000,
    "NVIDIA H100 80GB HBM3": 1.109,
    "NVIDIA A100-SXM4-80GB": 0.426,
    "NVIDIA L4": 0.337,
    "Tesla T4": 0.228,
}


def gpu_factor(gpu_name: str) -> float:
    """Multiplier from a measured ms_per_image to the reference GPU.

    Raises on unknown hardware so newly introduced GPUs are not silently
    treated as the reference and skew cross-run comparisons.
    """
    if not isinstance(gpu_name, str) or not gpu_name:
        raise ValueError(f"gpu_name must be a non-empty string, got {gpu_name!r}")
    try:
        return GPU_FACTORS[gpu_name]
    except KeyError as exc:
        known = ", ".join(sorted(GPU_FACTORS))
        raise KeyError(
            f"Unknown GPU {gpu_name!r}. Add a factor to "
            f"hardware_equivalence.GPU_FACTORS. Known: {known}"
        ) from exc


def gpu_from_run_dir(run_dir: Path) -> str | None:
    """Detect the GPU name from a run output folder.

    Supports both layouts produced by main.py:
    - ``runtime_info.json`` (jobA-style benchmark runs)
    - ``report_bundle/report.json`` (offline/streaming runs)
    """
    runtime_info = run_dir / "runtime_info.json"
    if runtime_info.exists():
        info = json.loads(runtime_info.read_text(encoding="utf-8"))
        name = info.get("gpu_name")
        if isinstance(name, str) and name:
            return name
    report = run_dir / "report_bundle" / "report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        devices = (
            data.get("hardware", {})
            .get("accelerator", {})
            .get("cuda_devices", [])
        )
        if devices:
            name = devices[0].get("name")
            if isinstance(name, str) and name:
                return name
    return None


def normalize_latency(
    df: pd.DataFrame,
    *,
    gpu_col: str = "gpu_name",
    lat_col: str = "mean_latency_ms",
    fps_col: str | None = "throughput_fps",
) -> None:
    """In-place: rescale latency (and FPS, if present) to the reference GPU.

    Each row must carry the GPU name in ``gpu_col``. Rows with NaN
    latency or NaN gpu_name are passed through unchanged.
    """
    if df.empty:
        return
    if gpu_col not in df.columns:
        raise KeyError(f"DataFrame is missing required column {gpu_col!r}")
    if lat_col not in df.columns:
        raise KeyError(f"DataFrame is missing required column {lat_col!r}")
    factors = df[gpu_col].map(gpu_factor, na_action="ignore")
    df[lat_col] = df[lat_col] * factors
    if fps_col and fps_col in df.columns:
        df[fps_col] = df[fps_col] / factors
