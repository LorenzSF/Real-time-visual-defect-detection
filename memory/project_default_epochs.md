---
name: epochs=1 in default.yaml is intentional (smoke test)
description: Do not "fix" the tiny epoch counts in default.yaml — they are a deliberate fast-iteration config
type: project
---

In [src/benchmark_AD/default.yaml](../src/benchmark_AD/default.yaml), the trained anomaly-detection models are pinned to very low epoch counts:

- `stfpm.epochs: 1`
- `csflow.epochs: 1`
- `draem.epochs: 1`
- `rd4ad.epochs: 1`

**Why:** this is the fast smoke-test config used during pipeline development and CI-like iteration on the local laptop — not the config meant to produce thesis results. Running the full pipeline to completion in seconds matters more here than result quality.

**How to apply:**
- Do **not** raise these values silently when editing default.yaml — it will make local runs unusably slow.
- When the user asks for representative / realistic benchmark numbers, bump epochs inside a **dedicated config overlay** (e.g. `configs/hpc_realiad.yaml`, `configs/runpod_full.yaml`) that extends `default.yaml`, never by editing defaults.
- Realistic targets for the thesis: STFPM ≈ 50–100, CSFlow ≈ 100–240, DRAEM ≈ 100–700, RD4AD ≈ 100–200.
- Feature-based models (PatchCore, PaDiM, SubspaceAD) are unaffected — they do not train.
