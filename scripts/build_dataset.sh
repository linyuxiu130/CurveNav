#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [GENERATION_CONFIG]" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

exec "${CURVENAV_PYTHON}" -m curvenav.data_generation.generate "${1:-configs/hssd_dataset.json}"
