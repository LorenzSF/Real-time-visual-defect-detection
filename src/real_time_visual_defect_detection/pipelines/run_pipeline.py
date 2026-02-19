from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from real_time_visual_defect_detection.io.dataset_loader import resolve_dataset
from real_time_visual_defect_detection.preprocessing.standard import read_image_bgr, resize, normalize_0_1
from real_time_visual_defect_detection.preprocessing.corruption import apply_corruption
from real_time_visual_defect_detection.models.embedding_distance import DummyDistanceModel


def run_pipeline(cfg: Dict[str, Any]) -> Path:
    run_cfg = cfg["run"]
    out_root = Path(run_cfg["output_dir"])
    run_id = f'{run_cfg.get("run_name","run")}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}'
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(run_cfg.get("seed", 42))
    np.random.seed(seed)

    # Save config snapshot
    (out_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    ds = cfg["dataset"]
    paths = resolve_dataset(ds["source_type"], ds["path"], ds["extract_dir"])
    if len(paths) == 0:
        raise ValueError(
            f"No image files found for dataset source_type='{ds['source_type']}' at path='{ds['path']}'."
        )

    pre = cfg["preprocessing"]
    do_resize = pre["resize"]["enabled"]
    w, h = int(pre["resize"]["width"]), int(pre["resize"]["height"])

    corr = cfg["corruption"]
    do_corr = bool(corr["enabled"])
    corr_type = corr.get("type", "")
    corr_params = corr.get("params", {})

    model_cfg = cfg["model"]
    model = DummyDistanceModel(threshold=float(model_cfg["threshold"]))

    rows: List[Dict[str, Any]] = []
    for p in paths:
        img = read_image_bgr(str(p))
        if do_resize:
            img = resize(img, (w, h))

        if do_corr:
            img = apply_corruption(img, corr_type, corr_params)

        x = normalize_0_1(img)
        out = model.predict(x)

        rows.append(
            {
                "path": str(p),
                "score": out.score,
                "pred_is_anomaly": int(out.is_anomaly),
            }
        )

    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_dir
