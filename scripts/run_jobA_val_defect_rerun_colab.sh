#!/usr/bin/env bash
# Job A val_defect rerun — only the 7 categories whose AUROC dropped on
# the 2026-04-27 v2 batch because train collapsed to ~52 goods. With
# min_train_goods raised to 200 in colab_featurebased_val_defect.yaml,
# AUROC/AUPR should return to within ±0.01 of the clean baseline.
#
# v1 cells (eraser, pcb, rolled_strip_base) are NOT in this list — they
# were run before the val_balance rules landed and AUROC was already
# clean. Re-run them too only if you want a uniformly v2 set of 10.
#
# Resumable via `<category>.done` markers. Delete the existing
# jobA_val_defect_<cat>_*/ directories first if you want to compare
# the two batches side by side; otherwise this script will create new
# timestamped runs and the analyzer will pick the latest by mtime.

set -euo pipefail

ZIPS_DIR="${ZIPS_DIR:-/content/drive/MyDrive/Real-IAD_dataset/realiad_1024}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobA_val_defect_rerun}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_featurebased_val_defect.yaml}"

CATEGORIES=(
  sim_card_set
  switch
  terminalblock
  toothbrush
  transistor1
  usb_adaptor
  zipper
)

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"

if [[ ! -d "${ZIPS_DIR}" ]]; then
  echo "ZIPS_DIR not found: ${ZIPS_DIR}" >&2
  exit 1
fi

echo "[jobA_val_defect_rerun] ${#CATEGORIES[@]} categories (min_train_goods=200)"
echo "[jobA_val_defect_rerun] config:     ${CONFIG}"
echo "[jobA_val_defect_rerun] results to: ${RESULTS_DIR}"
echo

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for category in "${CATEGORIES[@]}"; do
  zip="${ZIPS_DIR}/${category}.zip"
  marker="${RESULTS_DIR}/${category}.done"
  cat_work="${WORK_DIR}/${category}"

  if [[ -f "${marker}" ]]; then
    echo "[skip] ${category} already done (${marker})"
    continue
  fi

  if [[ ! -f "${zip}" ]]; then
    echo "[${category}] ERROR: zip not found at ${zip}" >&2
    continue
  fi

  echo "==============================================================="
  echo "[${category}] starting at $(date -u +%H:%M:%S)"
  echo "==============================================================="

  rm -rf "${cat_work}"
  mkdir -p "${cat_work}"

  local_zip="${cat_work}/${category}.zip"
  echo "[${category}] copying zip..."
  cp "${zip}" "${local_zip}"

  echo "[${category}] extracting..."
  unzip -q -o "${local_zip}" -d "${cat_work}"
  rm -f "${local_zip}"

  dataset_root="${cat_work}/${category}"
  if [[ ! -d "${dataset_root}/OK" ]]; then
    echo "[${category}] ERROR: expected OK/ under ${dataset_root}" >&2
    ls -la "${cat_work}" >&2
    rm -rf "${cat_work}"
    continue
  fi

  run_name="jobA_val_defect_${category}"
  echo "[${category}] running pipeline (3 models, val_f1, min_train_goods=200)..."
  python main.py \
    --config "${CONFIG}" \
    --dataset-path "${dataset_root}" \
    --extract-dir "${dataset_root}" \
    --run-name "${run_name}"

  latest_run="$(ls -1dt "${WORK_DIR}/runs/${run_name}"_* 2>/dev/null | head -n1)"
  if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
    echo "[${category}] ERROR: pipeline produced no output dir" >&2
    rm -rf "${cat_work}"
    continue
  fi

  echo "[${category}] syncing ${latest_run} -> results dir..."
  dest="${RESULTS_DIR}/$(basename "${latest_run}")"
  rsync -a --remove-source-files "${latest_run}/" "${dest}/"
  find "${latest_run}" -type d -empty -delete

  touch "${marker}"
  echo "[${category}] done -> ${dest}"

  rm -rf "${cat_work}"
done

echo
echo "[jobA_val_defect_rerun] all categories processed."
echo "Next: re-run scripts/compare_val_defect.py to see if AUROC/AUPR recovered."
