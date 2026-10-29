#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set the eight allocated GPUs}"
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
export OMP_NUM_THREADS=1 NCCL_PROTO=LL
exec "${CURVENAV_PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=8 -m curvenav.training.bspline_replay "$@"
