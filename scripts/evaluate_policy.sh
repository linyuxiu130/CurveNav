#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one free GPU}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src"
cd "${PROJECT_ROOT}"
exec "${WORKSPACE_ROOT}/.venvs/curvenav/bin/python" \
  -m curvenav.evaluation.offline \
  "${1:-configs/base.yaml}" \
  "${2:-outputs/train_policy/checkpoint.pt}"
