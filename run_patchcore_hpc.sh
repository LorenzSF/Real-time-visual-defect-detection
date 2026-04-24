#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Run this script with 'bash run_patchcore_hpc.sh ...', do not source it."
  return 1
fi

_run_patchcore_hpc() {
  set -euo pipefail
  trap 'echo "[run_patchcore_hpc.sh] failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

  _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${_script_dir}"

  if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_patchcore_hpc.sh /absolute/path/to/dataset [run_name]"
    return 1
  fi

  dataset_path="$1"
  run_name="${2:-patchcore_smoke}"
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  log_dir="${RVDD_LOG_DIR:-${_script_dir}/data/logs}"
  log_file="${log_dir}/run_patchcore_${run_name}_${timestamp}.log"

  mkdir -p "${log_dir}"
  exec > >(tee -a "${log_file}") 2>&1

  echo "[run_patchcore_hpc.sh] log file: ${log_file}"
  echo "[run_patchcore_hpc.sh] dataset_path=${dataset_path}"
  echo "[run_patchcore_hpc.sh] run_name=${run_name}"

  if [[ ! -e "${dataset_path}" ]]; then
    echo "Dataset path not found: ${dataset_path}"
    return 1
  fi

  source "${_script_dir}/setup_env.sh"

  python main.py \
    --config src/benchmark_AD/default.yaml \
    --model anomalib_patchcore \
    --dataset-path "${dataset_path}" \
    --run-name "${run_name}"
}

_run_patchcore_hpc "$@"
