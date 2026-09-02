#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

exec "${CURVENAV_PYTHON}" -m curvenav.data.prepare \
  --config configs/base.yaml \
  --hssd-root outputs/hssd_policy_dataset \
  --output data/policy_dataset-source-cspace
