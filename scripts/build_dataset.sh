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

exec "$workspace_root/.venvs/curvenav/bin/python" -m curvenav.data.prepare \
  --config configs/base.yaml \
  --hssd-root outputs/hssd_policy_dataset \
  --output data/policy_dataset-source-cspace
