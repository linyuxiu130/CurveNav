#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$project_root/.." && pwd)
cd "$project_root"
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

habitat_python="$workspace_root/.venvs/habitat-gs/bin/python"
log_path=outputs/hssd_policy_dataset.log

/usr/bin/time -p "${habitat_python}" -m curvenav.data_generation.generate \
  configs/hssd_dataset.json \
  2>&1 | tee "${log_path}"
