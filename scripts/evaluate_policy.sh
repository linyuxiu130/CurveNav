#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set the eight allocated GPUs}"
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
exec "${CURVENAV_PYTHON}" -m curvenav.evaluation.run_bspline40 "$@"
