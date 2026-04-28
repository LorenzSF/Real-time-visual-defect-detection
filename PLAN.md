# PLAN.md — Thesis Project: Real-Time Visual Defect Detection

> **For Claude Code:** Read this file before making any edit.
> Every code change must serve one of the goals listed here.
> Do not refactor outside task scope. Do not add new dependencies without asking first.
> Prefer simple, readable code over clever abstractions.
>
> **Methodology decisions** (split rules, threshold calibration, metric
> choices) live in [METHOD.md](METHOD.md). Update both files when those
> change.

---

## Project Goals

1. **Benchmarking pipeline** — Evaluate SOTA anomaly detection models on a standard dataset (Real-IAD) and a real industrial dataset (Deceuninck). Produce reproducible, comparable metrics.
2. **Streaming inference + XAI dashboard** — Run the best-performing model in a simulated production environment. Show explainable outputs (heatmaps, live metrics, live embedding plot) suitable for a factory floor operator.
3. **Corruption robustness evaluation** — Apply synthetic image corruptions in both batch and streaming modes to measure how model performance degrades under real-world visual degradations.

---

## Repository Structure (current + planned)

```
.
├── main.py                        # Entry point: batch benchmark pipeline
├── runtime_main.py                # Entry point: streaming inference pipeline
├── pyproject.toml
├── PLAN.md                        # ← this file
│
├── src/
│   ├── benchmark_AD/              # [EXISTS] Batch benchmarking pipeline
│   │   ├── pipeline.py            # Main benchmark orchestrator
│   │   ├── models.py              # Model registry and wrappers
│   │   ├── data.py                # Dataset loading and splitting
│   │   ├── evaluation.py          # Metrics: AUROC, F1, inference time
│   │   └── default.yaml           # Config file for all pipeline options
│   │
│   ├── corruptions/               # [PARTIAL] Synthetic corruption module
│   │   ├── corruption_registry.py # [TO BUILD] Functions per corruption type + severity
│   │   ├── test_loader.py
│   │   ├── test_pipeline_benchmark.py
│   │   └── test_registry.py
│   │
│   └── streaming_input/           # [TO BUILD] Streaming inference module
│       ├── app.py                 # StreamingInputApp — main loop
│       ├── settings.py            # load_settings(), DEFAULT_SETTINGS_FILE
│       ├── inference.py           # Per-frame model inference wrapper
│       └── dashboard.py           # Live XAI dashboard (Streamlit or Dash)
│
├── notebooks/
│   └── benchmark_graphs and tables.ipynb  # Results analysis
│
└── data/
    └── runs/                      # Auto-generated output per run
```

---

## Coding Conventions

These apply to every file touched or created in this project:

- **Language:** Python 3.11. Type hints on all function signatures.
- **Style:** Follow PEP 8. Max line length: 100 characters.
- **Comments:** Write comments that explain *why*, not *what*. A reader who knows Python should understand the purpose of every block without needing to trace the full codebase.
- **Functions:** One responsibility per function. If a function does more than one logical thing, split it.
- **Naming:** `snake_case` for variables and functions. Descriptive names — avoid `x`, `tmp`, `data2`.
- **Error handling:** Use explicit `try/except` with meaningful messages. Never silently swallow exceptions.
- **No hidden state:** Avoid global variables. Pass config/state explicitly through function arguments.
- **Dependencies:** Only use packages already declared in `pyproject.toml`. Ask before adding new ones.
- **No over-engineering:** No abstract base classes or plugin systems unless already in place. Flat is better than nested.

---

## Status snapshot — 2026-04-29

| Block | State | Notes |
|---|---|---|
| Pipeline + metrics | ✅ done | AUROC/AUPR, Recall@FPR=1%/5%, macro/weighted/per-defect recall, runtime cost — all in `benchmark_summary.json`. |
| JobA clean (val_quantile) | ✅ 30 cats × 3 feature-based + partial trained | Headline numbers untrustworthy on F1/Recall; AUROC valid. |
| **JobA val_defect (val_f1)** | ✅ 19 / 30 cats × 3 feature-based | Cleared §9 gates; default switched to `val_f1` on 2026-04-27. See [data/outputs/jobA_val_defect_V1/_analysis/REPORT.md](data/outputs/jobA_val_defect_V1/_analysis/REPORT.md). |
| JobA val_f1 missing 11 cats | 🟡 todo | porcelain_doll, regulator, tape, toy, toy_brick, u_block, usb, usb_adaptor, vcpill, wooden_beads, woodstick. |
| JobA trained (CSFlow/DRAEM/STFPM/RD4AD on HPC) | 🟡 partial | csflow audiojack only. See [docs/HPC_KU_LEUVEN_RUNBOOK.md §5.5](docs/HPC_KU_LEUVEN_RUNBOOK.md). |
| JobB Deceuninck clean (val_quantile) | ✅ 1 cell × 3 models | AUROC ≥ 0.995 across the 3 models. |
| **JobB Deceuninck val_f1** | ✅ feature-based / 🟡 trained | seed=42 baseline for all 6 models in `data/outputs/jobB_val_defect_V1/`. Seed sweep complete for padim+subspacead (s={7,17,42,123}) and partial for patchcore (s={7,42}) in `data/outputs/jobB_val_defect_and_seed/`. draem/stfpm/rd4ad still single-seed (s=42). |
| **Multi-seed sweep infra** | ✅ done | `--seed` flag on `main.py`; both jobB drivers accept `SEED`; val_defect driver also accepts `MODELS=` (per-model split, e.g. patchcore alone) and `SKIP_CLEANUP=1` (chain cells without disconnecting Colab). Per-`(model, seed)` markers under RESULTS_DIR make sweeps resumable. |
| Corruption module | 🟡 partial | See `src/corruptions/` — 3-corruption registry, pipeline wiring, smoke driver, and tests done. Job C has started locally on a Real-IAD subset; full sweep + Deceuninck + aggregation still pending. |
| Streaming module + dashboard | 🟡 partial | `src/streaming_input/` is flattened, `runtime_main.py` is wired, model-suffixed JSON artifacts are emitted under `streaming_output_*`, and the §2.1 dashboard panels are implemented on the existing HTTP server. Targeted tests + compile checks pass; browser-level run verification and thesis screenshots are still pending. |
| Notebook tables/plots | 🟡 partial | Tables A–D (TSV) generated under `_analysis/`; not yet imported into the headline notebook. **`notebooks/analyze jobB.ipynb`** built (37 cells) with multi-seed coherence section — confirms that 9 seed=42 metrics on patchcore/padim were single-split artifacts (e.g. patchcore f1: 1.0→0.989, recall: 1.0→0.979 at s=7); 23 metrics — including all `val_*` — saturate across every seed (val set too easy on Deceuninck). |

---

## Work Plan

### 1 — Benchmark + Corruptions + GPU Runs

#### 1.1 — Close the headline benchmark (val_f1 default, full grid)
**Goal:** every cell of the headline table reports AUROC, F1, **Recall@FPR=1%**, **macro/per-defect recall** and inference cost under the same `val_f1` calibration policy.

- ✅ Pipeline runs end-to-end on all supported models.
- ✅ `default.yaml` now uses `val_f1`. All Job A/B configs inherit it; the `*_val_defect.yaml` overlays are kept as historical aliases.
- 🟡 **JobA val_f1 — finish the remaining 11 cats** (porcelain_doll, regulator, tape, toy, toy_brick, u_block, usb, usb_adaptor, vcpill, wooden_beads, woodstick). Reuse [scripts/run_jobA_val_defect_colab.sh](scripts/run_jobA_val_defect_colab.sh) (or its successor) and append outputs into `data/outputs/jobA_val_defect_V1/`.
- 🟡 **JobA trained — wICE re-run.** `git pull` on the HPC clone (so `colab_trained.yaml` picks up `val_f1` and `image_size: 512` from `default.yaml`), extend to the 30 categories × 4 trained models matrix described in [HPC_KU_LEUVEN_RUNBOOK.md §6–7](docs/HPC_KU_LEUVEN_RUNBOOK.md).
- 🟡 **JobB Deceuninck val_f1.** Feature-based done with seed sweep (padim/subspacead at s={7,17,42,123}; patchcore at s={7,42}); trained models (draem/stfpm/rd4ad) still single-seed. See [notebooks/analyze jobB.ipynb](notebooks/analyze%20jobB.ipynb) §7 for the multi-seed coherence assessment.
- 🟡 **Re-run regression cell.** `plastic_plug × PaDiM` showed ΔAUROC = −0.096 in the val_defect rerun; re-launch with a fresh seed to confirm whether the drop is noise or real before publishing the table.
- After results: regenerate Tables A–D via [scripts/compare_clean_vd.py](scripts/compare_clean_vd.py) and identify the **best-performing model** by inspecting the consolidated `benchmark_summary.json` set manually → this name is then passed to `runtime_main.py --model <name>` for streaming. Never auto-select.

#### 1.2 — Build corruption module
**Goal:** `src/corruptions/corruption_registry.py` integrated into the batch pipeline.

- ✅ Implemented 3 corruption types in [src/corruptions/corruption_registry.py](src/corruptions/corruption_registry.py), each `(image: np.ndarray, severity: int) -> np.ndarray` with severity ∈ {1..5}:
  - `gaussian_blur` — spatial blurring
  - `motion_blur` — directional blur simulating camera motion
  - `jpeg_compression` — lossy compression artifact
- ✅ `get_corruption(name, severity)` factory exposed as the public API.
- ✅ Wired into `default.yaml` `corruption:` section (`enabled / type / severity`); `--corruption` and `--severity` CLI flags on `main.py` override the YAML.
- ✅ Pipeline applies corruption to **test images only** (after resize, before normalize); rows are stamped with `corruption_type / severity / dataset`; `live_status_<model>.json` mirrors the streaming session schema; corrupted TP/FP/FN/TN samples are exported per run.
- ✅ Smoke driver: [scripts/run_smoke_corruption.sh](scripts/run_smoke_corruption.sh) — `anomalib_padim × gaussian_blur × severity 3` on the local Deceuninck dataset.

#### 1.3 — Launch corruption benchmark + start streaming module
**Goal:** Job C running on GPU, JSON corruption outputs saved, `streaming_input/` ready.

- 🟡 Configure Job C configs:
  - Dedicated Job C overlays exist in `src/benchmark_AD/configs/colab_jobC_realiad.yaml` and `src/benchmark_AD/configs/colab_jobC_deceuninck.yaml`, with manual `--model` selection and per-model overrides matching Jobs A/B.
  - `scripts/run_jobC.sh` drives the corruption grid via CLI (`--corruption`, `--severity`) and currently defaults to the 5-category Real-IAD fallback subset, not the full 30-category grid.
  - `src/benchmark_AD/default.yaml`, `configs/realiad.yaml`, and `configs/industrial.yaml` remain clean-benchmark oriented rather than carrying the full Job C grid directly.
- 🟡 Launch on GPU cluster:
  - **Job C:** local outputs exist under `data/outputs/jobC_corruption/` (116 run folders checked on 2026-04-28), showing the sweep is underway.
  - Current checked outputs are Real-IAD only; Deceuninck corruption cells are not present yet.
- 🟡 Save Job C outputs:
  - The pipeline emits `predictions_<model>.json` and `live_status_<model>.json` with `corruption_type`, `severity`, `dataset`, and summary metrics in the streaming-shaped JSON schema.
  - `benchmark_summary.json` remains part of the run contract. This is the accepted §1.3 artifact set.
- 🟡 Start building `src/streaming_input/`:
  - ✅ `settings.py`: `DEFAULT_SETTINGS_FILE` and `load_settings(path) -> dict` are present.
  - ✅ `inference.py`: wraps the selected model and returns `{anomaly_score, anomaly_map, embedding}`.
  - ✅ `app.py`: `StreamingInputApp` loads the model, iterates folder input, and writes `benchmark_summary.json`, `predictions_<model>.json`, and `live_status_<model>.json`.
  - 🟡 `dashboard.py`: no longer empty; a minimal HTTP dashboard placeholder exists, while the full §2.1 XAI dashboard is still pending.
- 🟡 Clean `src/streaming_input/`:
  - ✅ Folder input handling, model loading, inference contracts, and session-output writing were folded into the flat module structure.
  - ✅ Dashboard/report/web-app expansion is still deferred to §2.1.
  - ✅ `runtime_main.py` already supports manual `--model <name>` and `--run-dir <path>` selection, and also already exposes `--corruption` / `--severity`.
  - 🟡 `src/streaming_input/settings.yaml` is kept as runtime config data, while `__init__.py` now exports only `StreamingInputApp`, `DEFAULT_SETTINGS_FILE`, and `load_settings`.
- 🟡 Verify `StreamingInputApp().run()`:
  - Runtime defaults now write sessions under `data/streaming_output/streaming_output_<timestamp>/`.
  - ✅ `tests/test_streaming_input.py` now validates the flattened architecture, model-suffixed JSON artifacts, copied `benchmark_summary.json`, and per-frame output writing.
---

### 2 — Streaming + Dashboard + Results

#### 2.1 — Live XAI dashboard
**Goal:** `src/streaming_input/dashboard.py` — a dashboard that runs alongside the streaming loop and updates in real time.

Dashboard must show (in a single screen, readable by a factory operator):

| Panel | Content |
|---|---|
| Current frame | Input image with anomaly heatmap overlaid (colormap: green→red) |
| Anomaly score | Numeric gauge or bar with decision threshold marked |
| Inference throughput | FPS counter (rolling average over last 10 frames) |
| Anomaly rate | % of frames classified as anomalous since session start |
| Score history | Line chart of the last N anomaly scores (N configurable, default 100) |
| Live embedding plot | 2D scatter updated each frame — new points colored by score |

**Implementation notes for the embedding plot:**
- Pre-fit a UMAP or PCA projection on the training set embeddings (offline, once at startup).
- Each new frame: extract embedding → project using the pre-fit transformer → add point to scatter.
- Color scale: green (score near 0) → red (score near threshold and above).
- No ground-truth labels are available at inference time — color by score only.

**Recommended stack:** Streamlit (simple, already available). Use `st.empty()` placeholders and `time.sleep()` loop for live updates. If Streamlit is not suitable, use Plotly Dash with a background thread.

**Current status — 2026-04-28**
- ✅ The existing `dashboard.py` HTTP server now renders the required single-screen panels: current frame + green→red overlay, anomaly score gauge, rolling FPS, anomaly rate, score history, and live embedding scatter.
- ✅ `app.py` now publishes the dashboard state needed by §2.1, including the latest overlaid frame, rolling score history, anomaly rate, and projected embedding points.
- ✅ The embedding path is implemented with a startup-fitted projector (`pca` default, `umap` supported by config) over training embeddings, with a deterministic fallback embedding when a model exposes no native `get_embedding()`.
- 🟡 Targeted automated verification passes (`tests/test_streaming_input.py` + compile check), but an interactive browser run and thesis screenshot capture are still pending.


#### 2.2 — Results analysis and thesis writing
**Goal:** All figures and tables needed for the thesis are generated and exported.

- Import the val_defect TSVs already generated into `notebooks/benchmark_graphs and tables.ipynb`:
  - `data/outputs/jobA_val_defect_V1/_analysis/tableA_per_cat_model.tsv`
  - `tableB_per_model.tsv`, `tableC_threshold_shift.tsv`, `tableD_val_sanity.tsv`
- Update the notebook with the post-rerun run outputs (clean baseline + 30/30 val_f1 cells once §1.1 is closed).
- Generate the following figures:
  1. **Threshold-calibration evidence** — clean vs val_f1 per-model bar chart (F1 / Recall lift, AUROC stability). Direct support for METHOD.md §2.
  2. **Model comparison table:** AUROC and F1 on Real-IAD (val_f1) vs Deceuninck (val_f1).
  3. **F1 lift bar chart and PR scatter** described in [PLAN job A_analize val_defect.md §7](PLAN%20job%20A_analize%20val_defect.md).
  4. **Robustness curves:** AUROC vs corruption severity, per corruption type (line chart, output of §1.3 / Job C).
  5. **Streaming dashboard screenshots:** clean run and corrupted run side by side.
  6. **UMAP embedding plot snapshot** from a streaming session (if available).
- Export all figures at 300 DPI (`.png`) and `.pdf`.
- Write or complete in thesis:
  - **Methodology:** pipeline architecture, threshold-calibration policy (val_f1 + patched splitter, justified in METHOD.md §2-3), corruption module design, streaming setup.
  - **Results:** model selection rationale, robustness analysis, dashboard demonstration.

---

## Integral action map — what's left to ship the thesis

Sequenced by dependency (each block unblocks the next). Status: ✅ done · 🟡 in progress · ⛔ blocked / not started.

### Phase A — Lock the headline numbers (val_f1 default)

1. ✅ Switch `default.yaml` to `val_f1`. Annotate clean configs.
2. 🟡 **JobA val_f1 — finish the 11 missing cats**
   (porcelain_doll, regulator, tape, toy, toy_brick, u_block, usb,
   usb_adaptor, vcpill, wooden_beads, woodstick) on Colab via
   [scripts/run_jobA_val_defect_colab.sh](scripts/run_jobA_val_defect_colab.sh).
   Output dir: `data/outputs/jobA_val_defect_V1/`.
3. 🟡 **Re-run plastic_plug × PaDiM** with a fresh seed. The 19-cat rerun
   showed ΔAUROC = −0.096 there; confirm noise vs. real regression
   before the headline table goes into the thesis.
4. ✅ **JobB Deceuninck val_f1 — feature-based** via
   [scripts/run_jobB_val_defect_colab.sh](scripts/run_jobB_val_defect_colab.sh).
   Seed sweep done for padim + subspacead (s={7,17,42,123}); patchcore at
   s={7,42}, two more seeds pending. Coherence in
   [notebooks/analyze jobB.ipynb §7](notebooks/analyze%20jobB.ipynb).
   🟡 **JobB Deceuninck val_f1 — trained models** still seed=42 only
   (draem, stfpm, rd4ad); needs the multi-seed sweep on Colab/HPC.
5. 🟡 **HPC `git pull`** on `/data/leuven/.../Real-time-visual-defect-detection`,
   then re-run `anomalib_csflow audiojack` and extend to the
   30 cats × 4 trained models matrix. Sanity check: the resulting
   `benchmark_summary.json` must show `threshold_mode: val_f1` and
   `image_size: 512`.
6. 🟡 **Regenerate Tables A–D** by re-running
   [scripts/compare_clean_vd.py](scripts/compare_clean_vd.py) once the
   30 cells are complete. Update
   [data/outputs/jobA_val_defect_V1/_analysis/REPORT.md](data/outputs/jobA_val_defect_V1/_analysis/REPORT.md).

### Phase B — Robustness (Job C, corruptions)

7. 🟡 Finalize [src/corruptions/corruption_registry.py](src/corruptions/corruption_registry.py)
   with the 6 functions from §1.2 and `get_corruption(name, severity)` factory.
8. 🟡 Wire the corruption block into the pipeline (test-set only, training/val stay clean).
9. 🟡 **Launch Job C** on GPU: 3 corruption types × {1, 3, 5} severities
   × 3+ models × Real-IAD (subset acceptable per fallback) + Deceuninck.
10. ⛔ Aggregate Job C → robustness curves (AUROC vs severity per corruption).

### Phase C — Streaming + dashboard

11. ⛔ **Pick the production model** by reading the consolidated headline
    table (do not auto-select). PaDiM is the current frontrunner on both
    quality and cost; confirm after the 30/30 val_f1 grid is complete.
12. 🟡 Build [src/streaming_input/](src/streaming_input/) — `app.py`,
    `settings.py`, `inference.py` per §1.3. `runtime_main.py
    --model <name>` / `--run-dir <path>` wiring is in place; targeted
    verification passes, while interactive end-to-end runtime validation
    remains.
13. 🟡 Build [src/streaming_input/dashboard.py](src/streaming_input/dashboard.py)
    per §2.1 (current frame + heatmap, score gauge, FPS, anomaly rate,
    score history, live UMAP embedding). Implemented on the existing HTTP
    dashboard stack; browser verification and screenshot capture remain.
14. 🟡 Add `--corruption` / `--severity` to `runtime_main.py` for the
    corrupted-stream comparison (§2.2). CLI flags are already present;
    clean/corrupted dashboard screenshot capture is still pending.

### Phase D — Thesis figures and writing

15. ⛔ Open `notebooks/benchmark_graphs and tables.ipynb`, import the
    Phase A TSVs, and render:
    a. Calibration evidence (clean vs val_f1).
    b. Model comparison (Real-IAD val_f1 vs Deceuninck val_f1).
    c. F1 lift bar chart + PR scatter per
       [PLAN job A_analize val_defect.md §7](PLAN%20job%20A_analize%20val_defect.md).
    d. Robustness curves from Job C.
    e. Streaming dashboard screenshots.
    f. UMAP embedding snapshot.
16. ⛔ Export every figure at 300 DPI `.png` + `.pdf` per
    [PLAN job A_analize val_defect.md §2.3](PLAN%20job%20A_analize%20val_defect.md).
17. ⛔ Methodology chapter — pipeline architecture, val_f1 + splitter
    policy (cite METHOD.md §2-3), corruption module design, streaming
    setup. Limitations table from
    [METHOD.md §8](METHOD.md#8-open-methodological-issues-limitations-to-disclose).
18. ⛔ Results chapter — model selection rationale, headline benchmark
    table, robustness analysis, dashboard demonstration.

### Critical-path gates (block downstream phases)

- **Phase A → B**: Job C corruption sweep should reuse the chosen
  calibration (val_f1) so its numbers stack on top of Phase A's
  headline table. Do not start Job C until §A.6 is done.
- **Phase B → C**: streaming model selection must be informed by the
  *robust* AUROC (degradation under corruption), not just the clean
  AUROC. Wait until §B.10 is aggregated.
- **Phase C → D**: dashboard screenshots are required figures (§2.1);
  no thesis figure pass before the dashboard runs end-to-end.

---

## Fallback Priorities (if time runs short)

| Component | Required for thesis? | Fallback |
|---|---|---|
| Batch benchmark (clean) | ✅ Yes | — |
| Corruption batch benchmark | ✅ Yes | Reduce to 3 types, severities 1/3/5 |
| Streaming inference loop | ✅ Yes | Folder-based, no live camera |
| XAI dashboard | ✅ Yes | Static heatmap figures only |
| Live embedding plot | ⚠️ Desirable | Static UMAP from a recorded session |
| Streaming + corruptions | ✅ Yes | Single corruption type only |
| Interactive demo | ❌ Optional | Skip entirely |

---

## Key Output Files (what the thesis needs)

```
data/runs/<run_name>/
├── benchmark_summary.json              # Model comparison table + run context.
│                                       # Per-model entry now includes:
│                                       #   recall_at_fpr_1pct, recall_at_fpr_5pct,
│                                       #   macro_recall, weighted_recall,
│                                       #   per_defect_recall, per_defect_support
├── predictions_<model>.json            # Per-frame records, streaming-shape JSON (with corruption_type, severity, dataset)
├── live_status_<model>.json            # Per-(model × corruption × severity × dataset) session summary, streaming-shape JSON (with AUROC, F1)
├── plots/
│   ├── robustness_curves.png           # AUROC vs severity per corruption type
│   ├── model_comparison.png            # Clean benchmark comparison
│   ├── heatmap_sample_<model>.png      # XAI heatmap examples
│   └── embedding_umap_<model>.html     # Embedding plot
└── streaming_session/
    ├── dashboard_clean.png             # Dashboard screenshot, no corruption
    └── dashboard_corrupted.png         # Dashboard screenshot, with corruption
```

Robustness results are emitted as JSON only (mirroring `data/streaming_output/<session>/`), not CSV.

---

## Out of Scope (do not implement)

- Training new models from scratch (use pretrained weights only).
- Support for video streams or live camera input (folder simulation is sufficient).
- Any web API or REST endpoint.
- Docker deployment or CI/CD pipeline changes.
- Unit test coverage beyond what already exists in `tests/`.
