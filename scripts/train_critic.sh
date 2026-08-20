#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/src"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_P2P_DISABLE=1

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != 2 )); then
    echo "CurveNav critic training requires exactly two CUDA devices" >&2
    exit 2
fi

CONFIG_PATH="${1:-configs/train_critic_hssd_v2.yaml}"
cd "${PROJECT_ROOT}"
exec "${WORKSPACE_ROOT}/.venvs/curvenav/bin/torchrun" \
    --standalone \
    --nproc-per-node=2 \
    -m curvenav.critic.train \
    "${CONFIG_PATH}"
