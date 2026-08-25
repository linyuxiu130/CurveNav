#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
ENV_ROOT="${WORKSPACE_ROOT}/.venvs"

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to at least two free GPUs}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export CPATH="${ENV_ROOT}/python-headers/root/usr/include/python3.10:${ENV_ROOT}/python-headers/root/usr/include${CPATH:+:${CPATH}}"
export TORCHINDUCTOR_CACHE_DIR="${ENV_ROOT}/curvenav/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HIGH_PRIORITY=1

CONFIG_PATH="${1:-configs/base.yaml}"
if (( $# )); then
    shift
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 2 )); then
    echo "CurveNav production training requires at least two CUDA devices" >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
exec "${ENV_ROOT}/curvenav/bin/torchrun" \
    --standalone \
    --nproc-per-node="${NUM_PROCESSES}" \
    -m curvenav.training.train \
    "${CONFIG_PATH}" \
    "$@"
