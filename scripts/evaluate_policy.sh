#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one free GPU}"
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
config_path="${1:-configs/base.yaml}"
checkpoint_path="${2:-outputs/train_policy-e010/checkpoint.pt}"
if (( $# >= 2 )); then
  shift 2
else
  shift "$#"
fi
exec "${CURVENAV_PYTHON}" \
  -m curvenav.evaluation.offline \
  "${config_path}" \
  "${checkpoint_path}" \
  "$@"
