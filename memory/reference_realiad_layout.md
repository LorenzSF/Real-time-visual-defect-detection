---
name: Real-IAD dataset layout — gotchas
description: Non-obvious structural details of Real-IAD that have broken runs in past sessions
type: reference
---

Real-IAD is ingested via the `real_iad` format in [src/benchmark_AD/data.py](../src/benchmark_AD/data.py) (`_list_real_iad_samples`). The pipeline expects **one category at a time** via `dataset.path`, not a parent folder containing all categories.

**Expected per-category layout:**
```
<category>/
├── OK/
│   └── S000X/                    # specimen ID (used for leakage-free splits)
│       └── <img>_C1_*.jpg        # camera views C1..C5 (jpg only)
└── NG/
    └── <defect_type>/
        └── S000X/
            ├── <img>_C1_*.jpg    # image
            └── <img>_C1_*.png    # segmentation mask — IGNORED by loader
```

**Three failure modes that broke previous runs:**

1. **Double-nested category folder from zips.** Each `<category>.zip` embeds a top-level `<category>/` directory, so unzipping `audiojack.zip` into `/dataset/audiojack/` produces `/dataset/audiojack/audiojack/OK/...`. The config must point `dataset.path` at the INNER path (`/dataset/audiojack/audiojack`), not the outer. See [scripts/unzip_realiad.sh](../scripts/unzip_realiad.sh) and [src/benchmark_AD/configs/realiad.yaml](../src/benchmark_AD/configs/realiad.yaml).

2. **`format: "real_iad"` must be set.** `auto` mode mis-detects `S000X/` groupings as defect classes and ingests PNG masks as images, which poisons the test set.

3. **PNG/JPG mixing in NG/.** JPGs are images, PNGs are segmentation masks — the `real_iad` loader filters the PNGs via `_REAL_IAD_IMAGE_EXTS` ([data.py](../src/benchmark_AD/data.py)). `auto` / `generic` modes do not filter.

**On Drive (as of 2026-04-24):** all 30 categories are stored as individual `.zip` files under `Real-IAD_dataset/realiad_1024/` — 1024 px variant, still zipped. The Colab pipeline extracts one category at a time to local scratch to avoid filling the Colab disk with all 30 extracted (~200+ GB).

**Source: https://realiad4ad.github.io/Real-IAD/** (official dataset page). 30 object categories × 5 camera views (C1..C5).
