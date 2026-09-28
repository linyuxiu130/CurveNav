#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session=curvenav-behavior-live
mode="${1:-start}"
if (($#)); then shift; fi

case "$mode" in
  start)
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "BEHAVIOR viewer is already running (tmux: $session)"
      exit 0
    fi
    args=()
    printf -v args '%q ' "$root/behavior_adapter/live_viewer.sh" run "$@"
    tmux new-session -d -s "$session" "cd $(printf '%q' "$root") && $args"
    echo "Starting persistent BEHAVIOR simulator and web viewer..."
    echo "Follow startup: bash behavior_adapter/live_viewer.sh logs"
    ;;
  status)
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux display-message -p -t "$session" 'BEHAVIOR viewer session: #{session_name} (attached clients: #{session_attached})'
      curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:8765/api/state || true
      echo
    else
      echo "BEHAVIOR viewer is not running"
      exit 1
    fi
    ;;
  logs)
    tmux capture-pane -p -S -100 -t "$session" 2>/dev/null || { echo "BEHAVIOR viewer is not running"; exit 1; }
    ;;
  stop)
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      echo "Stopped persistent BEHAVIOR simulator."
    else
      echo "BEHAVIOR viewer is not running"
    fi
    ;;
  run)
    behavior_root="${CURVENAV_BEHAVIOR_ROOT:-/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K-v3.9.2-baidu}"
    behavior_data="${CURVENAV_OMNIGIBSON_DATA_PATH:-/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K/datasets}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
    export OMNIGIBSON_DATA_PATH="$behavior_data"
    export PYTHONPATH="/shibo_huang/data/curvenav/runtime/cuda_python_12_6_2:$root/behavior_adapter/live_viewer:$root/behavior_adapter:$root/behavior_adapter/vendor/OmniGibson:$root/src:$behavior_root/bddl3:$behavior_root/joylo"
    export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 OMNIGIBSON_NO_OMNI_LOGS=1
    export OMNIGIBSON_RENDER_VIEWER_CAMERA=False
    export NUMBA_CUDA_USE_NVIDIA_BINDING=1 NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    unset DISPLAY
    cache="/tmp/curvenav-behavior-live-${UID}-gpu${GPU_ID:-0}"
    mkdir -p "$cache"
    exec 8>"$cache/simulator.lock"
    flock -n 8 || { echo "A BEHAVIOR scene already owns simulator cache $cache" >&2; exit 1; }
    export OMNIGIBSON_APPDATA_PATH="$cache/appdata"
    export OMNIGIBSON_GLOBAL_CACHE_PATH="$cache/global"
    task="${CURVENAV_LIVE_TASK:-turning_on_radio}"
    instance="${CURVENAV_LIVE_INSTANCE:-0}"
    cd "$root"
    exec /opt/conda/envs/behavior/bin/python -u behavior_adapter/live_viewer/server.py \
      --task "$task" --instance "$instance" --host 127.0.0.1 --port 8765 "$@" 2>&1 | tee "$cache/viewer.log"
    ;;
  *)
    echo "Usage: $0 [start|status|logs|stop] [server arguments]" >&2
    exit 2
    ;;
esac
