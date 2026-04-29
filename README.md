# Evaluation of Unsupervised Defect Detection Models on Industrial Data Streams Under Corruption

Thesis pipeline backing the work above: from offline benchmarking of unsupervised AD models, through synthetic corruption stress-testing, to a live streaming-inference dashboard suitable for a factory operator.

## Integral View

- One end-to-end loop: **train/calibrate → benchmark → stress-test under corruption → stream the chosen model with an XAI dashboard**.
- A single config schema (`src/benchmark_AD/default.yaml` + per-dataset overlays) drives every stage so clean, corrupted, and streaming runs share the same preprocessing, splits, and thresholding policy.
- Artifacts produced by the batch benchmark (`benchmark_summary.json`, model weights/state) are the same artifacts consumed by the streaming app — no re-training, no schema drift.
- Two entry points:
  - [main.py](main.py) — batch benchmark (clean or corrupted).
  - [runtime_main.py](runtime_main.py) — streaming inference + live dashboard.

## Sections of Work

The pipeline is split into three independent modules under [src/](src/). Each module ships its own README explaining how it works, what to configure, and why.

- **Benchmarking pipeline** — reproducible offline evaluation of SOTA unsupervised AD models on Real-IAD and Deceuninck → [src/benchmark_AD/README.md](src/benchmark_AD/README.md).
- **Corruption robustness** — synthetic image corruptions injected into the test set to measure degradation curves → [src/corruptions/README.md](src/corruptions/README.md).
- **Streaming inference + XAI dashboard** — per-frame inference loop and live operator panel reusing a benchmark-trained model → [src/streaming_input/README.md](src/streaming_input/README.md).

## License

Apache-2.0. See `LICENSE`.
