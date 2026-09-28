#!/usr/bin/env bash
# Reuse each free main 8x3 slot after all 100 tasks have been dispatched.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
exec /opt/conda/envs/behavior/bin/python -u behavior_adapter/retry_idle_slots.py \
    --main-pid "${1:?Pass the initial scheduler PID}"
