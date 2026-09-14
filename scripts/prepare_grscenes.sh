#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
# Use the USD runtime already shipped with Isaac; keep it out of training Python.
CURVE_USD_LIB="${CONDA_PREFIX}/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311"
export PYTHONPATH="${CURVENAV_PROJECT_ROOT}/src:${CURVE_USD_LIB}"
export LD_LIBRARY_PATH="${CURVE_USD_LIB}/bin:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
exec "${CURVENAV_PYTHON}" -m curvenav.data_generation.grscenes "$@"
