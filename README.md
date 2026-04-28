# Real-Time Visual Defect Detection

Thesis pipeline for evaluating and deploying anomaly-detection models on industrial imagery, from offline benchmarking to a live operator dashboard.

## Integral View

- One end-to-end loop: **train/calibrate → benchmark → stress-test under corruption → stream the chosen model with an XAI dashboard**.
- A single config schema (`src/benchmark_AD/default.yaml` + per-dataset overlays) drives every stage so clean, corrupted, and streaming runs share the same preprocessing, splits, and thresholding policy.
- Artifacts produced by the batch benchmark (`benchmark_summary.json`, model weights/state) are the same artifacts consumed by the streaming app — no re-training, no schema drift.
- Two entry points:
  - [main.py](main.py) — batch benchmark (clean or corrupted).
  - [runtime_main.py](runtime_main.py) — streaming inference + live dashboard.

## Sections of Work

The pipeline is split into three independent modules under [src/](src/). Each one has its own README explaining how it works, what to configure, and why.

- [src/benchmark_AD/](src/benchmark_AD/) — **Benchmarking pipeline.** Reproducible offline evaluation of SOTA AD models on Real-IAD and Deceuninck. See [src/benchmark_AD/README.md](src/benchmark_AD/README.md).
- [src/corruptions/](src/corruptions/) — **Corruption robustness.** Synthetic image corruptions injected into the test set to measure degradation curves. See [src/corruptions/README.md](src/corruptions/README.md).
- [src/streaming_input/](src/streaming_input/) — **Streaming inference + XAI dashboard.** Per-frame inference loop and live operator panel reusing a benchmark-trained model. See [src/streaming_input/README.md](src/streaming_input/README.md).

## License

Apache-2.0. See `LICENSE`.
