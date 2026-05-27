# Evaluation of Unsupervised Defect Detection Models on Industrial Data Streams

Pipeline for streaming inference, visualization, standardized benchmarking, and
corruption robustness analysis of unsupervised industrial anomaly detectors.

## Goals

1. Run industrial image streams and produce anomaly scores, heatmaps, and traces.
2. Benchmark detectors under the same dataset, split, threshold, seed, and output
   schema.
3. Measure robustness under synthetic corruptions.

## Project Info

Author: Lorenzo Fresca

Contact: `lorenzostefano.fresca@student.kuleuven.be` or
`frescalorenzo@gmail.com`

This software was developed as part of a master's thesis in the Advanced Master
of Artificial Intelligence in Business and Industry at KU Leuven.

Supervisor: Prof. Mathias Verbeke
Daily supervisor: Matthias De Ryck

## Install

```bash
pip install -r requirements.txt
```

`anomalib` provides the heavy detector stack. For CUDA runs, install the matching
`torch` and `torchvision` build before installing `requirements.txt`.

## Input Data

`stream.input_path` must point to one flat image folder. Subfolders are rejected.
Only files with suffixes listed in `stream.extensions` are streamed.

```text
data/
  input_images/
    image_0001.jpg
    image_0002.jpg
    image_0003.jpg
    labels.json        # optional for streaming metrics, required for offline
```

`image_id` is the filename without extension and must be unique. If present,
`labels.json` must map every listed id to exactly `"OK"` or `"NG"`:

```json
{
  "image_0001": "OK",
  "image_0002": "NG"
}
```

Streaming mode allows missing labels and excludes unknown labels from AUROC,
AUPR, precision, recall, F1, and accuracy. Offline mode requires OK/NG labels for
all images because it builds labelled train/validation/test splits.

## Config

All runtime behavior is controlled by `config.yaml`; `main.py` is the only entry
point.

Main sections:

- `run.mode`: `streaming` or `offline`
- `stream`: dataset name, flat input folder, extensions, shuffle, max_frames
- `warmup`: warmup_steps, fit_epochs
- `model`: detector name, backbone, device, checkpoint, image_size, batch_size
- `corruption`: enabled flag and ordered corruption specs
- `metrics`: online window and streaming threshold calibration
- `visualization`: file/window/none output and optional dashboard
- `offline`: split ratios and validation-threshold settings

Supported detectors:

- `pca`
- `patchcore` / `anomalib_patchcore`
- `padim` / `anomalib_padim`
- `subspacead`
- `stfpm` / `anomalib_stfpm`
- `csflow` / `anomalib_csflow`
- `draem` / `anomalib_draem`
- `rd4ad` / `reverse_distillation`

`warmup.fit_epochs` is used by `draem`, `stfpm`, `csflow`, and `rd4ad`.
Memory-based detectors (`pca`, `patchcore`, `padim`, `subspacead`) fit in one
pass and ignore it.

Supported streaming threshold modes:

- `max_score_ok`: maximum finite score from the calibration window.
- `pot`: Peaks-over-threshold calibration using a Generalized Pareto tail fit.

Supported offline threshold modes:

- `val_f1`: threshold with best validation F1.
- `val_quantile`: OK-score validation quantile from `offline.threshold.target_fpr`.

Supported corruptions are `gaussian_noise`, `shot_noise`, and `motion_blur`.
Severity is `1`, `2`, or `3`. Multiple specs are sampled independently per frame
and compose in config order. Offline mode requires `corruption.enabled: false`.

## Streaming Flow

Run:

```bash
python main.py
```

When `run.mode: streaming`, the pipeline:

1. Loads and validates `config.yaml`.
2. Seeds Python, NumPy, and Torch when available.
3. Builds the configured detector.
4. Fits it on the first `warmup.warmup_steps` sorted images.
5. Scores the next `metrics.calibration_steps` post-warmup frames to calibrate
   the threshold.
6. Applies configured corruptions lazily to the remaining stream.
7. Runs prediction, metrics, visualization, and per-frame logging.
8. Writes a self-contained `report.json`.

Warm-up and threshold-calibration frames are logged but excluded from benchmark
metrics. If `stream.shuffle: true`, only the post-calibration tail is shuffled;
the warm-up and calibration prefixes stay sorted.

## Offline Flow

When `run.mode: offline`, `main.py`:

1. Requires labels for all images.
2. Builds a deterministic train/validation/test split from `offline.split`.
3. Fits the detector on train frames.
4. Calibrates the threshold on validation scores.
5. Evaluates once on test frames.

Offline mode supports the experiment detectors only: PatchCore, PaDiM,
SubspaceAD, STFPM, CSFlow, DRAEM, and RD4AD.

## Outputs

Each run writes to:

```text
output_dir/<experiment_name>/
```

Streaming experiment names are derived from model, dataset, input folder,
optional corruption specs, and timestamp. Offline names also include
`offline.experiment_name`.

Common outputs:

- `report.json`: metrics, threshold metadata, runtime, hardware, model, stream,
  warm-up, corruption, and evaluation metadata.
- `frames.jsonl`: line-buffered per-frame trace with `idx`, `image_id`, `phase`,
  `score`, `pred_label`, `threshold_used`, `true_label`, and `latency_ms`.
- `frame_*.png`: rendered anomaly frames when `visualization.mode: file`.

Offline mode also writes:

- `validation_predictions.jsonl`
- `predictions.jsonl`

Label convention is `0` for OK, `1` for NG, and `-1` for unknown or unavailable.
Load JSONL files with `pandas.read_json(path, lines=True)`.

## Visualization

`visualization.mode` controls frame rendering:

- `file`: save rendered anomalous frames in the run directory.
- `window`: show an OpenCV window.
- `none`: disable frame rendering.

If `visualization.dashboard_enabled: true`, a FastAPI/WebSocket dashboard runs
beside the pipeline at `http://<dashboard_host>:<dashboard_port>/`. The
dashboard shows live metrics, current frame overlay, and a StandardScaler +
PCA(2) projection when detector vectors are available.

## Active Modules

- [main.py](main.py): orchestration, threshold logic, reporting
- [src/schemas.py](src/schemas.py): dataclasses and strict config loading
- [src/stream.py](src/stream.py): image discovery, loading, warm-up, splits
- [src/models.py](src/models.py): detector construction and prediction contract
- [src/corruption.py](src/corruption.py): lazy per-frame corruptions
- [src/metrics.py](src/metrics.py): online/offline metrics and JSONL logging
- [src/visualization.py](src/visualization.py): rendered frames and dashboard

## Extending

Add a model in `src/models.py` and register it in `build_model`.

Add a corruption kernel in `src/corruption.py` and register it in
`_CORRUPTIONS`.

Any user-facing config change must update `config.yaml`, `src/schemas.py`, and
this README.

## License

Apache-2.0. See `LICENSE`.

Copyright 2026 KU Leuven and Lorenzo Stefano Fresca.
