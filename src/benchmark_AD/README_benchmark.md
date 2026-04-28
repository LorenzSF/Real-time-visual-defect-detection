# benchmark_AD — Batch Benchmark Pipeline

Reproducible offline evaluation of anomaly-detection models on image datasets. Produces the headline tables and the model artifacts later consumed by the streaming module.

## How It Works

- Entry point [main.py](../../main.py) loads a YAML config, builds a single model or the full benchmark grid, and writes one timestamped run folder per launch.
- [pipeline.py](pipeline.py) orchestrates the four stages: **resolve dataset → split → fit on train → score val/test → calibrate threshold → emit metrics**.
- [data.py](data.py) discovers labels (JSON sidecar → `good/` + `bad/` convention → flat fallback) and builds leakage-free splits (per-sample-id stratification for Real-IAD).
- [models.py](models.py) wraps each AD backend behind a uniform `BaseModel.fit / predict / get_embedding` interface so the pipeline stays model-agnostic.
- [evaluation.py](evaluation.py) computes AUROC, AUPR, F1, recall@FPR=1%/5%, macro/weighted/per-defect recall, and runtime cost.
- Outputs live under `data/outputs/<run_name>_<UTC_TS>/`: `runtime_info.json`, `benchmark_summary.json`, `predictions_<model>.json`, `validation_predictions_<model>.json`, optional `plots/embedding_umap_<model>.html`.

## Configuration

- Base config: [default.yaml](default.yaml). Per-dataset overlays in [configs/](configs/) extend it via the top-level `_extends` key — only override the fields that differ.
- Key sections:
  - `run` — seed, output directory, run name.
  - `runtime` — device (`auto | cpu | cuda`), precision, workers, cudnn.
  - `dataset` — source type, path, format hint (`auto | real_iad`), camera filter, split policy.
  - `preprocessing` — resize (default 512×512), normalize (`0_1`).
  - `model` — single-model defaults + per-backbone hyperparameters.
  - `model.thresholding.mode` — `val_f1` (default), `val_quantile`, or `fixed`.
  - `corruption` — disabled by default; see [src/corruptions/](../corruptions/).
  - `benchmark.models` — list of models for multi-model runs.
- CLI overrides on `main.py`: `--model`, `--all-models`, `--dataset-path`, `--run-name`, `--seed`, `--corruption`, `--severity`.

## Considerations

- **Threshold mode is `val_f1`** because it lifts F1/recall on industrial defects without hurting AUROC; `val_quantile` is kept as a sanity-check baseline and as automatic fallback when val collapses to a single class.
- **Train on goods only** (`split.train_on_good_only: true`) so feature-based models (PaDiM, PatchCore, SubspaceAD) see a clean nominal manifold; bad samples are reserved for val/test calibration and scoring.
- **Leakage-free splits** — Real-IAD shares one physical sample across multiple camera views, so splits stratify by `sample_id`, not by file.
- **Image size 512×512** matches the resolution at which the trained models (RD4AD, STFPM, CSFlow, DRAEM, SubspaceAD) were calibrated; changing it invalidates the headline numbers.
- **Per-model dependency preflight** — unavailable models are skipped with a logged reason, never silently failing the whole run.
- **Seed exposure** — `--seed` flag plus per-`(model, seed)` markers under the run dir make multi-seed sweeps resumable.
- **Single-model vs benchmark mode** — `--model` and `--all-models` are mutually exclusive; `benchmark.models: []` falls back to the single-model block.

## Why It Is Built This Way

- A flat module layout (one file per concern) is easier to read and modify under a thesis deadline than a plugin architecture nobody else will extend.
- Config-as-source-of-truth (YAML + `_extends`) keeps clean, val_defect, and corruption runs comparable: a per-job overlay only encodes what changes, so methodology drift between jobs is visually obvious in a diff.
- Writing both `validation_predictions_*` and `predictions_*` lets downstream notebooks recompute thresholds offline without re-running inference.
- Decoupling threshold calibration from the model fit means a single trained model produces multiple metric tables (`val_f1`, `val_quantile`, `fixed`) without retraining.
