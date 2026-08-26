#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one free GPU}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
ENV_ROOT="${WORKSPACE_ROOT}/.venvs"
export PYTHONPATH="${PROJECT_ROOT}/src"
PYTHON_INCLUDE="$("${ENV_ROOT}/curvenav/bin/python" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
export CPATH="${PYTHON_INCLUDE}${CPATH:+:${CPATH}}"
export TORCHINDUCTOR_CACHE_DIR="${ENV_ROOT}/curvenav/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=2
cd "${PROJECT_ROOT}"
exec "${ENV_ROOT}/curvenav/bin/python" \
  -m curvenav.evaluation.offline \
  "${1:-configs/base.yaml}" \
  "${2:-outputs/train_policy/checkpoint.pt}"
