#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATA_ROOT" >&2
  exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$project_root/.." && pwd)
data_root=$(realpath -m "$1")
sand_root="$data_root/sandplanner"
export PYTHONPATH="$project_root/src"
cd "$project_root"

scripts/download_sand_dataset.sh "$sand_root"
scripts/download_hssd_assets.sh
scripts/generate_hssd_dataset.sh
"$workspace_root/.venvs/curvenav/bin/python" scripts/prepare_sand_depth_cache.py \
  configs/base.yaml "$sand_root"
"$workspace_root/.venvs/curvenav/bin/python" scripts/prepare_hssd_depth_cache.py \
  configs/base.yaml outputs/hssd_policy_dataset
scripts/prepare_policy_dataset.sh \
  configs/base.yaml "$sand_root" outputs/hssd_policy_dataset data/policy_dataset
