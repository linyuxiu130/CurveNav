#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
ENV_ROOT="${WORKSPACE_ROOT}/.venvs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export CPATH="${ENV_ROOT}/python-headers/root/usr/include/python3.10:${ENV_ROOT}/python-headers/root/usr/include${CPATH:+:${CPATH}}"
export TORCHINDUCTOR_CACHE_DIR="${ENV_ROOT}/curvenav/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=8

CONFIG_PATH="${1:-configs/base.yaml}"

cd "${PROJECT_ROOT}"
exec "${ENV_ROOT}/curvenav/bin/python" \
    -m curvenav.training.overfit \
    "${CONFIG_PATH}"
