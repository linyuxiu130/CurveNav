#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 CONFIG CHECKPOINT [EVALUATION ARGUMENTS ...]" >&2
  exit 2
fi
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one free GPU}"
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
config_path="$1"
checkpoint_path="$2"
shift 2
exec "${CURVENAV_PYTHON}" \
  -m curvenav.evaluation.offline \
  "${config_path}" \
  "${checkpoint_path}" \
  "$@"
