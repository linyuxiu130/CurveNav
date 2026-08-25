#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 CONFIG SAND_ROOT HSSD_DATASET_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src"
exec "${WORKSPACE_ROOT}/.venvs/curvenav/bin/python" -m curvenav.data.prepare \
  --sand-root "$2" \
  --hssd-root "$3" \
  --output "$4" \
  --config "$1"
