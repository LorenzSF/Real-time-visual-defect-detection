# PLAN.md — Thesis Project: Real-Time Visual Defect Detection

> **For Claude Code:** Read this file before making any edit.
> Every code change must serve one of the goals listed here.
> Do not refactor outside task scope. Do not add new dependencies without asking first.
> Prefer simple, readable code over clever abstractions.

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
│   ├── benchmark_graphs and tables.ipynb  # Results analysis
│   └── example_notebook.ipynb
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

## Work Plan

### 1 — Benchmark + Corruptions + GPU Runs

#### 1.1 — Stabilize and run batch benchmark
**Goal:** `benchmark_summary.json` with AUROC, F1, and inference time per model, on both datasets.

- Verify `main.py` runs end-to-end with all supported models.
- Fix any model-loading or evaluation bugs found during the run.
- Launch on GPU cluster:
  - **Job A:** Clean benchmark, standard dataset (Real-IAD), all models.
  - **Job B:** Clean benchmark, real industrial dataset (Deceuninck), all models.
- After results: identify the **best-performing model** by inspecting `benchmark_summary.json` manually → this name is then passed to `runtime_main.py --model <name>` for streaming. Never auto-select.

#### 1.2 — Build corruption module
**Goal:** `src/corruptions/corruption_registry.py` integrated into the batch pipeline.

- Implement the following corruption types, each as a standalone function accepting `(image: np.ndarray, severity: int) -> np.ndarray` where severity ∈ {1, 2, 3, 4, 5}:
  - `gaussian_noise` — additive pixel noise
  - `gaussian_blur` — spatial blurring
  - `motion_blur` — directional blur simulating camera motion
  - `brightness_shift` — uniform exposure change
  - `contrast_reduction` — reduce dynamic range
  - `jpeg_compression` — lossy compression artifact

- Expose a `get_corruption(name, severity)` factory function as the public API.
- Wire into `default.yaml` via existing `corruption:` section (key: `type`, `severity`).
- Quick test: one model, one corruption, one dataset category.

#### 1.3 — Launch corruption benchmark + start streaming module
**Goal:** Job C running on GPU, JSON corruption outputs saved, `streaming_input/` ready.

- Configure `src/benchmark_AD/default.yaml` and per-dataset configs (`configs/realiad.yaml`, `configs/industrial.yaml`) for:
  - Full model list, same as Jobs A/B. No automatic model selection.
  - `corruption:` enabled with all 6 types from §1.2 and severities `{1, 3, 5}`.
- Launch on GPU cluster:
  - **Job C:** Batch benchmark with corruptions, all models × 6 corruptions × severities `{1, 3, 5}`, on Real-IAD and Deceuninck.
- Save Job C outputs under `data/runs/<run_name>/`, one session folder per `(model × corruption × severity × dataset)`, using JSON only:
  - `predictions.json`: per-frame records `{model, path, label, defect_type, score, pred_is_anomaly, heatmap_path, corruption_type, severity, dataset}`.
  - `live_status.json`: session summary with streaming status keys (`active_model`, `frames_seen`, `decisions_emitted`, `mean_latency_ms`, `p95_latency_ms`, `threshold`, ...) plus `AUROC`, `F1`, `corruption_type`, `severity`, `dataset`.
  - Shape mirrors `data/streaming_input/streaming_input_20260411_160237/`.
- Start building `src/streaming_input/`:
  - `settings.py`: keep public surface `DEFAULT_SETTINGS_FILE`, `load_settings(path) -> dict`.
  - `inference.py`: wrap the selected model to process a single image and return `{anomaly_score, anomaly_map, embedding}`.
  - `app.py`: `StreamingInputApp` class — init loads model, `run()` iterates over image source (folder or simulated camera), calls inference per frame, saves `predictions.json` and `live_status.json`.
  - `dashboard.py`: empty placeholder until §2.1.
- Clean `src/streaming_input/` to contain only `app.py`, `settings.py`, `inference.py`, `dashboard.py`, and `__init__.py`:
  - Fold folder input handling, model loading, inference contracts, and session-output writing into the flat module structure.
  - Defer dashboard, live metrics, reports, and web app concerns to §2.1.
  - Update `__init__.py` to export only `StreamingInputApp`, `DEFAULT_SETTINGS_FILE`, `load_settings`.
  - Update `runtime_main.py` for manual model selection via CLI flags such as `--model <name>` and `--run-dir <path>`.
- Verify `StreamingInputApp().run()` iterates a folder source, calls inference per frame, and writes the JSON outputs above.
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

#### 2.2 — Streaming with corruptions
**Goal:** The streaming loop can optionally apply corruptions frame-by-frame, reusing the same registry from Block 2.

- Add `--corruption` and `--severity` flags to `runtime_main.py`.
- In `app.py`, if corruption is configured: apply `get_corruption(name, severity)` to each frame before inference.
- This enables direct comparison: dashboard metrics with vs. without corruption active.
- Capture dashboard screenshots for thesis figures.

#### 2.3 — Results analysis and thesis writing
**Goal:** All figures and tables needed for the thesis are generated and exported.

- Update `notebooks/benchmark_graphs and tables.ipynb` with new run outputs.
- Generate the following figures:
  1. Model comparison table: AUROC and F1 on standard vs. industrial dataset (clean).
  2. Robustness curves: AUROC vs. corruption severity, per corruption type (line chart).
  3. Streaming dashboard screenshots: clean run and corrupted run side by side.
  4. UMAP embedding plot snapshot from a streaming session (if available).
- Export all figures at 300 DPI (`.png`) and `.pdf`.
- Write or complete in thesis:
  - **Methodology:** pipeline architecture, corruption module design, streaming setup.
  - **Results:** model selection rationale, robustness analysis, dashboard demonstration.

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
├── benchmark_summary.json              # Model comparison table + run context
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

Robustness results are emitted as JSON only (mirroring `data/streaming_input/<session>/`), not CSV.

---

## Out of Scope (do not implement)

- Training new models from scratch (use pretrained weights only).
- Support for video streams or live camera input (folder simulation is sufficient).
- Any web API or REST endpoint.
- Docker deployment or CI/CD pipeline changes.
- Unit test coverage beyond what already exists in `tests/`.
