#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one or more free GPUs}"
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HIGH_PRIORITY=1

CONFIG_PATH="${1:-data/policy_dataset-depth-forward/config.yaml}"
if (( $# )); then
    shift
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#GPU_IDS[@]}"
exec "${CURVENAV_PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${NUM_PROCESSES}" \
    -m curvenav.training.train \
    "${CONFIG_PATH}" \
    "$@"
