# corruptions — Synthetic Robustness Module

Injects deterministic image corruptions into the test set so the same trained models can be re-scored under controlled visual degradation. Backs the robustness-curve figure of the thesis.

## How It Works

- [corruption_registry.py](corruption_registry.py) holds three pure functions, each with the contract `(image: np.ndarray, severity: int) -> np.ndarray`:
  - `gaussian_blur` — focus loss (Gaussian sigma per severity).
  - `motion_blur` — camera shake (rotated line kernel, deterministic angle per image).
  - `jpeg_compression` — transmission/storage artifact (JPEG quality per severity).
- Public API is `get_corruption(name, severity)` — it validates both arguments at factory time and returns a per-image callable.
- The benchmark pipeline calls the factory once per run and applies the callable **after resize, before normalize**, on test images only.
- Each row of `benchmark_summary.json` (and the streaming-shape JSONs) is stamped with `corruption_type` and `corruption_severity` so robustness curves can be plotted directly from the file.

## Configuration

- YAML (in `default.yaml` or any overlay):
  ```yaml
  corruption:
    enabled: false
    type: "gaussian_blur"   # gaussian_blur | motion_blur | jpeg_compression
    severity: 3             # 1..5
  ```
- CLI overrides on both `main.py` and `runtime_main.py`:
  - `--corruption <name>` — enables the block and selects the type.
  - `--severity <1..5>` — sets intensity.
- Severity grid actually used in Job C: `{1, 3, 5}` × the three types × the headline model set.

## Considerations

- **Test-set only.** Training and validation stay clean so threshold calibration is unaffected and clean vs. corrupted runs are directly comparable.
- **Severity is discrete (1..5)** and validated up front — a misconfigured run fails at startup, not on the first test image.
- **Stochastic corruptions are seeded per image** (BLAKE2b hash of the raw bytes), so the same picture always receives the same perturbation across reruns without leaking a global seed.
- **Shape and dtype are preserved** so the corruption is a drop-in swap inside the pipeline — no branching downstream.
- **JPEG round-trip uses Pillow** and respects the BGR convention used by the rest of the pipeline (BGR → RGB → encode → decode → BGR).
- **Three types only** — the minimal set covering focus, motion, and compression. Adding a fourth requires registering a function in `_CORRUPTIONS`; the YAML and CLI surfaces pick it up automatically.

## Why It Is Built This Way

- A plain function-registry is the smallest abstraction that gives a stable public API (`get_corruption`) without locking the project into a class hierarchy it does not need.
- Per-image deterministic seeding gives reproducible numbers across reruns while still varying the perturbation across samples — important for honest robustness reporting.
- Applying corruption *after resize, before normalize* matches the order a real sensor pipeline would degrade a frame and keeps the corruption-free path bit-identical to the clean benchmark.
- Stamping run artifacts with `corruption_type` / `corruption_severity` means notebooks aggregate robustness curves with a single `groupby`, no filename parsing.
- The registry is also imported by [streaming_input/](../streaming_input/) so live corrupted-stream demos use the exact same code path as the batch benchmark.
