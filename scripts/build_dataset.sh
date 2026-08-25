#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$project_root/.." && pwd)
export PYTHONPATH="$project_root/src"
cd "$project_root"

scripts/download_hssd_assets.sh
scripts/generate_hssd_dataset.sh
"$workspace_root/.venvs/curvenav/bin/python" scripts/prepare_hssd_depth_cache.py \
  configs/base.yaml outputs/hssd_policy_dataset
scripts/prepare_policy_dataset.sh \
  configs/base.yaml outputs/hssd_policy_dataset data/policy_dataset
