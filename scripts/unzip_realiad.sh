#!/bin/bash
# Bulk-extract every Real-IAD category zip under $DATASET_DIR.
#
# Each zip `<category>.zip` expands to `<DATASET_DIR>/<category>/<category>/...`
# because the archives embed a top-level category folder. This matches the
# layout expected by the `real_iad` data loader (default.yaml: dataset.path =
# <DATASET_DIR>/<category>/<category>).
#
# Override DATASET_DIR via env var or edit the default below.

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: dataset directory not found: $DATASET_DIR" >&2
    exit 1
fi

shopt -s nullglob
zips=("$DATASET_DIR"/*.zip)
if [[ ${#zips[@]} -eq 0 ]]; then
    echo "No .zip files found under $DATASET_DIR" >&2
    exit 1
fi

for zip in "${zips[@]}"; do
    category="$(basename "$zip" .zip)"
    target="$DATASET_DIR/$category"

    # Skip when the inner <category>/<category> directory already has content,
    # so re-running the script is cheap and idempotent.
    if [[ -d "$target/$category" ]] && [[ -n "$(ls -A "$target/$category" 2>/dev/null)" ]]; then
        echo "[skip] $category already extracted at $target/$category"
        continue
    fi

    echo "[unzip] $category -> $target"
    mkdir -p "$target"
    unzip -q -o "$zip" -d "$target"
done

echo "All Real-IAD categories extracted under $DATASET_DIR"
