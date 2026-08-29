#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
ENV_ROOT="${WORKSPACE_ROOT}/.venvs"
export PYTHONPATH="${PROJECT_ROOT}/src"
cd "${PROJECT_ROOT}"
exec "${ENV_ROOT}/curvenav/bin/python" -m curvenav.evaluation.compare "$@"
