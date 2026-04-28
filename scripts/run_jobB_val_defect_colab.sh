#!/usr/bin/env bash
# Job B "val_defect" driver: Deceuninck dataset with the patched splitter
# (10% of anomalies routed into val) and val_f1 thresholding.
#
# Single dataset (no per-category loop). Resumable via per-(model,seed)
# markers under RESULTS_DIR — delete the marker to rerun.
#
# Environment (override with `KEY=value bash run_jobB_val_defect_colab.sh`):
#   DATASET_DIR    Directory on Drive containing good/ and defects/
#   RESULTS_DIR    Directory on Drive where the run is copied
#   WORK_DIR       Colab-local scratch dir (fast SSD)
#   REPO_DIR       Local clone of the thesis repo
#   CONFIG         Path to the val_defect Job B config YAML
#   SEED           Seed override for cfg['run']['seed'] (default 42)
#   MODELS         Comma list of model names (e.g. "anomalib_padim,subspacead").
#                  Empty (default) = run all models bundled in one python call.
#                  When set, dataset is copied ONCE then python is invoked per
#                  model so heavy models (patchcore) can run alone.
#   SKIP_CLEANUP   "1" to keep the Colab runtime alive at the end (useful when
#                  you want to chain a second invocation in the same session).

set -euo pipefail

SEED="${SEED:-42}"
MODELS="${MODELS:-}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
DATASET_DIR="${DATASET_DIR:-/content/drive/MyDrive/Deceuninck_dataset}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobB_val_defect}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_featurebased_deceuninck_val_defect.yaml}"

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}/good" || ! -d "${DATASET_DIR}/defects" ]]; then
  echo "Expected good/ and defects/ under ${DATASET_DIR}" >&2
  ls -la "${DATASET_DIR}" >&2
  exit 1
fi

local_dataset="${WORK_DIR}/jobB_val_defect_deceuninck_s${SEED}"

echo "==============================================================="
echo "[jobB_val_defect] starting at $(date -u +%H:%M:%S)"
echo "[jobB_val_defect] seed:       ${SEED}"
echo "[jobB_val_defect] models:     ${MODELS:-<all bundled>}"
echo "[jobB_val_defect] config:     ${CONFIG}"
echo "[jobB_val_defect] dataset:    ${DATASET_DIR}"
echo "[jobB_val_defect] results to: ${RESULTS_DIR}"
echo "==============================================================="

rm -rf "${local_dataset}"
mkdir -p "${local_dataset}"
echo "[jobB_val_defect] copying dataset from Drive to ${local_dataset}..."
cp -r "${DATASET_DIR}/good"    "${local_dataset}/good"
cp -r "${DATASET_DIR}/defects" "${local_dataset}/defects"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

run_one() {
  # Args: $1 = model name ("" for all-bundled), $2 = run_name, $3 = marker path
  local model="$1"
  local run_name="$2"
  local marker="$3"

  if [[ -f "${marker}" ]]; then
    echo "[jobB_val_defect] already done (${marker}). Delete it to rerun." >&2
    return 0
  fi

  local model_args=()
  if [[ -n "${model}" ]]; then
    model_args=(--model "${model}")
    echo "[jobB_val_defect] running ${model} alone..."
  else
    echo "[jobB_val_defect] running pipeline (all models bundled, val_f1)..."
  fi

  python main.py \
    --config "${CONFIG}" \
    "${model_args[@]}" \
    --dataset-path "${local_dataset}" \
    --extract-dir "${local_dataset}" \
    --run-name "${run_name}" \
    --seed "${SEED}"

  local latest_run
  latest_run="$(ls -1dt "${WORK_DIR}/runs/${run_name}"_* 2>/dev/null | head -n1)"
  if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
    echo "[jobB_val_defect] ERROR: pipeline produced no output dir for ${run_name}" >&2
    return 1
  fi

  echo "[jobB_val_defect] syncing ${latest_run} -> Drive..."
  local dest="${RESULTS_DIR}/$(basename "${latest_run}")"
  rsync -a --remove-source-files "${latest_run}/" "${dest}/"
  find "${latest_run}" -type d -empty -delete
  touch "${marker}"
  echo "[jobB_val_defect] done -> ${dest}"
}

if [[ -z "${MODELS}" ]]; then
  run_one "" "jobB_val_defect_deceuninck_s${SEED}" "${RESULTS_DIR}/jobB_val_defect__s${SEED}.done"
else
  IFS=',' read -ra _models <<< "${MODELS}"
  for raw in "${_models[@]}"; do
    m="$(echo "${raw}" | tr -d '[:space:]')"
    [[ -z "${m}" ]] && continue
    run_one "${m}" \
      "jobB_val_defect_deceuninck_${m}_s${SEED}" \
      "${RESULTS_DIR}/jobB_val_defect__${m}__s${SEED}.done"
  done
fi

rm -rf "${local_dataset}"

if [[ "${SKIP_CLEANUP}" == "1" ]]; then
  echo "[jobB_val_defect] SKIP_CLEANUP=1 — leaving Colab runtime alive."
  exit 0
fi

# Release Colab resources once the run is finished and synced.
python - <<'PY'
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
gc.collect()
try:
    from google.colab import runtime
    try:
        runtime.unassign()
    except Exception as exc:
        print(f"[cleanup] runtime.unassign() skipped: {exc}")
except Exception:
    pass
PY
