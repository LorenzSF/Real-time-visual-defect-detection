---
name: Thesis compute strategy for Jobs A/B/C
description: How the three benchmark jobs are split across HPC, Colab, and paid GPU, and why
type: project
---

Jobs are defined in [PLAN.md](../PLAN.md):
- **Job A** — clean benchmark on Real-IAD (all 7 models × 30 categories)
- **Job B** — clean benchmark on Deceuninck (all 7 models)
- **Job C** — corruption sweep (7 models × 6 corruptions × 3 severities × 2 datasets = 252 runs)

**Current split (as of 2026-04-24):**
- Today on Colab Pro (this session's work): **feature-based preview of Job A** — PatchCore + PaDiM + SubspaceAD on all 30 Real-IAD categories, **camera C1 only**. Runs via `notebooks/run_jobA_colab.ipynb` + `scripts/run_jobA_colab.sh` + `src/benchmark_AD/configs/colab_featurebased.yaml`. Results sync to `Drive:/thesis_runs/jobA/`. Resumable via `<category>.done` markers.
- Later on KU Leuven Genius HPC (when queue clears) *or* RunPod 4090 rental: **trained models for Job A** (STFPM, CSFlow, DRAEM, RD4AD with realistic epochs, all 5 cameras) + **Jobs B and C in full**.

Why: trained models at realistic epochs on all cameras would need ~60 h on Colab A100 — doesn't fit a single session. Feature-based models (no training) finish in ~4–6 h on V100 and give defensible representative results today without waiting on HPC.

How to apply: when Lorenzo asks about running models, first check what fits the current compute path. Don't propose "run everything on Colab" — it won't finish. Don't propose "wait for HPC" when Colab can deliver a usable subset today.
