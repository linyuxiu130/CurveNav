#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NAVBENCH_EVAL_ENV="${NAVBENCH_EVAL_ENV:-/opt/conda/envs/curvenav-unified}"
export PIP_CONSTRAINT="$ROOT_DIR/config/unified-constraints.txt"
bash "$ROOT_DIR/scripts/setup_evaluator_env.sh"
"$NAVBENCH_EVAL_ENV/bin/python" -m pip install -e "$ROOT_DIR/..[test]" -e "$ROOT_DIR" 'timm==1.0.29'
(cd "$ROOT_DIR/.runtime/wheels" && sha256sum -c SHA256SUMS)
"$NAVBENCH_EVAL_ENV/bin/python" -m pip install \
    "$ROOT_DIR/.runtime/wheels/magnum-0.0.0-cp311-cp311-linux_x86_64.whl" \
    "$ROOT_DIR/.runtime/wheels/habitat_sim-0.3.3-cp311-cp311-linux_x86_64.whl"
"$NAVBENCH_EVAL_ENV/bin/python" -m pip check
