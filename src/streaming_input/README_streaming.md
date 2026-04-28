# streaming_input — Streaming Inference + XAI Dashboard

Runs the model selected by the benchmark in a simulated production loop and serves a single-screen operator dashboard with explainable outputs.

## How It Works

- Entry point [runtime_main.py](../../runtime_main.py) loads `settings.yaml`, resolves the benchmark run to consume, and starts the streaming app.
- [app.py](app.py) drives the loop: **iterate frames → preprocess → infer → overlay heatmap → publish dashboard state → write JSON artifacts**.
- [inference.py](inference.py) reloads a previously trained model from a benchmark run directory and exposes `predict(image) → {anomaly_score, anomaly_map, embedding}`.
- [dashboard.py](dashboard.py) serves a dependency-free HTML page on a local HTTP server with two JSON feeds: `/api/bootstrap` (static reference embedding) and `/api/status` (per-frame metrics).
- [settings.py](settings.py) loads and validates `settings.yaml` into the runtime config object.
- Per session, writes `benchmark_summary.json`, `predictions_<model>.json`, `live_status_<model>.json`, and `heatmaps/` overlays under `data/streaming_output/streaming_output_<timestamp>/`.

## Configuration

- File: [settings.yaml](settings.yaml).
- Sections:
  - `run` — output dir, target FPS, latency SLA (ms), max frames per session.
  - `artifact` — which benchmark run to reuse (`runs_root` + optional `run_dir`), `model_name`, and `fit_policy` (`auto | historical_fit | skip_fit`).
  - `input` — folder source, `loop` flag, `sequence_mode` (e.g. `interleaved_labels`).
  - `preprocessing` — must mirror the benchmark run (resize 512×512, `0_1` normalize) so the loaded model sees the input distribution it was calibrated on.
  - `web` — host, port, refresh interval.
  - `dashboard` — score history length, embedding projector (`pca` default, `umap` optional), reference-set cap, live-point cap.
- CLI overrides on `runtime_main.py`: `--model`, `--run-dir`, `--corruption`, `--severity`.

## Considerations

- **Manual model selection.** The streaming app does not auto-pick a winner — `--model <name>` must be passed explicitly, decided from the consolidated headline table and robustness curves.
- **Reuses benchmark artifacts.** A streaming session never trains; it loads the model state from a benchmark run dir, guaranteeing the same threshold and preprocessing as in the reported numbers.
- **Single-screen dashboard panels** (per PLAN.md §2.1): overlaid frame, anomaly-score gauge with threshold marker, rolling FPS, anomaly rate, score-history line chart, live 2D embedding scatter.
- **Embedding projector is fitted once at startup** on training-set embeddings (PCA by default, UMAP optional) and frozen; live frames are projected through the fixed transformer so the scatter is comparable across the session.
- **Score-only colouring** — green→red gradient driven by anomaly score, no ground-truth labels at inference time.
- **Deterministic embedding fallback** — when a model exposes no native `get_embedding()`, a fixed-shape fallback embedding is used so the scatter still works.
- **Corruption symmetry** — `--corruption` / `--severity` use the exact same registry as the batch benchmark, enabling clean-vs-corrupted dashboard screenshots from one code path.
- **Folder input only** — no live camera or REST API by design (out of scope per PLAN.md).

## Why It Is Built This Way

- The four-file flat layout (`app`, `inference`, `dashboard`, `settings`) is the layout prescribed by PLAN.md §1.3 and folds the previous `input_handler / decision_engine / live_metrics / report_generator` split into one module — fewer places where state can desync.
- A dependency-free HTML frontend (one HTML page, two JSON endpoints) avoids pulling Streamlit/Dash into the runtime: the dashboard runs anywhere Python runs, including a factory machine without internet.
- Pre-fitting the projector at startup keeps per-frame cost bounded and the scatter visually stable — refitting per frame would make trajectories meaningless.
- Publishing per-frame state through a thread-safe shared object (with a `Lock`) lets the HTTP server and the inference loop run independently — slow browsers cannot stall inference.
- Writing artifacts in the same JSON shape as the batch benchmark (`benchmark_summary.json`, `predictions_<model>.json`, `live_status_<model>.json`) means the same notebooks aggregate batch and streaming sessions without special-casing.
