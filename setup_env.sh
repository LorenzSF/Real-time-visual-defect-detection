#!/usr/bin/env bash

# Source this file from a KU Leuven Genius shell:
#   source setup_env.sh
#
# It loads the module stack, keeps caches in scratch, creates/activates a
# reusable venv in scratch, installs the repo in editable mode, and installs
# only the missing Python packages that are not provided by the cluster.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Use 'source setup_env.sh' so the modules and environment stay active."
  exit 1
fi

_rvdd_setup() {
  set -euo pipefail
  trap 'echo "[setup_env.sh] failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

  if ! command -v module >/dev/null 2>&1; then
    echo "'module' is not available in this shell. Open a Genius shell first."
    return 1
  fi

  _rvdd_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${_rvdd_repo_root}"

  export RVDD_CLUSTER_MODULE="${RVDD_CLUSTER_MODULE:-cluster/genius/batch}"
  export RVDD_PYTHON_MODULE="${RVDD_PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
  export RVDD_SCIPY_MODULE="${RVDD_SCIPY_MODULE:-SciPy-bundle/2023.07-gfbf-2023a}"
  export RVDD_OPENCV_MODULE="${RVDD_OPENCV_MODULE:-OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib}"
  export RVDD_TORCH_MODULE="${RVDD_TORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
  export RVDD_TORCHVISION_MODULE="${RVDD_TORCHVISION_MODULE:-torchvision/0.16.0-foss-2023a-CUDA-12.1.1}"

  if [[ -n "${VSC_SCRATCH:-}" ]]; then
    export RVDD_SCRATCH_ROOT="${RVDD_SCRATCH_ROOT:-${VSC_SCRATCH}}"
  else
    export RVDD_SCRATCH_ROOT="${RVDD_SCRATCH_ROOT:-/scratch/leuven/381/vsc38124}"
  fi

  export RVDD_VENV="${RVDD_VENV:-${RVDD_SCRATCH_ROOT}/.venv-rvdd}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RVDD_SCRATCH_ROOT}/.cache}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
  export TMPDIR="${TMPDIR:-${RVDD_SCRATCH_ROOT}/tmp}"
  export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
  export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"

  mkdir -p "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${TORCH_HOME}" "${HF_HOME}"

  module load "${RVDD_CLUSTER_MODULE}"
  module load "${RVDD_PYTHON_MODULE}"
  module load "${RVDD_SCIPY_MODULE}"
  module load "${RVDD_OPENCV_MODULE}"
  module load "${RVDD_TORCH_MODULE}"
  module load "${RVDD_TORCHVISION_MODULE}"

  if [[ ! -x "${RVDD_VENV}/bin/python" ]]; then
    python -m venv "${RVDD_VENV}" --system-site-packages
  fi

  source "${RVDD_VENV}/bin/activate"

  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir --no-build-isolation -e . --no-deps

  if ! python -c "import anomalib, lightning" >/dev/null 2>&1; then
    python -m pip install --no-cache-dir anomalib lightning
  fi

  python -c "import torch, torchvision, anomalib, lightning, cv2; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('anomalib', anomalib.__version__); print('lightning', lightning.__version__); print('cv2', cv2.__version__)"

  echo
  echo "Environment ready."
  echo "Repo: ${_rvdd_repo_root}"
  echo "Scratch root: ${RVDD_SCRATCH_ROOT}"
  echo "Venv: ${RVDD_VENV}"
  echo
  echo "Run the pipeline with:"
  echo "  python main.py --config src/benchmark_AD/default.yaml --model anomalib_patchcore --dataset-path /absolute/path/to/dataset --run-name patchcore_smoke"
}

_rvdd_setup
_rvdd_status=$?

unset -f _rvdd_setup
return "${_rvdd_status}"
