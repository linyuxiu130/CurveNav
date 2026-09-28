#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_behavior_root=/Benchmark/Behavior2026/BEHAVIOR-1K-v3.9.2-baidu
default_behavior_data=/Benchmark/Behavior2026/BEHAVIOR-1K/datasets
if [[ ! -d "$default_behavior_root" ]]; then
  default_behavior_root=/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K-v3.9.2-baidu
  default_behavior_data=/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K/datasets
fi
behavior_root="${CURVENAV_BEHAVIOR_ROOT:-$default_behavior_root}"
export OMNIGIBSON_DATA_PATH="${CURVENAV_OMNIGIBSON_DATA_PATH:-$default_behavior_data}"
export PYTHONPATH="$root/behavior_adapter/vendor/OmniGibson:$root/src:$behavior_root/bddl3:$behavior_root/joylo"

# Both hosts use the same GPFS dataset. Lock one task across hosts and skip
# tasks whose complete.json was published by the other collector.
args=("$@")
for ((arg_index=0; arg_index<${#args[@]}; arg_index++)); do
  if [[ "${args[$arg_index]}" == "--collect-plan" ]]; then
    (( arg_index + 1 < ${#args[@]} )) || { echo '--collect-plan needs a path' >&2; exit 2; }
    plan_path="${args[$((arg_index + 1))]}"
    route_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["route_root"])' "$plan_path")"
    lock_root="$(dirname "$plan_path")/../.task_locks"
    mkdir -p "$lock_root"
    exec 7>"$lock_root/$(basename "$plan_path" .json).lock"
    flock 7
    if [[ -f "$route_root/complete.json" ]]; then
      printf '{"event": "dataset_task_already_complete", "plan": "%s"}\n' "$plan_path"
      exit 0
    fi
    break
  fi
done
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 OMNIGIBSON_NO_OMNI_LOGS=1
export OMNIGIBSON_RENDER_VIEWER_CAMERA=False
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset DISPLAY
cache="/tmp/curvenav-behavior-scene-${UID}-gpu${GPU_ID:-0}"
if [[ -n "${CURVENAV_GPU_SLOT:-}" ]]; then
  cache="${cache}-slot${CURVENAV_GPU_SLOT}"
fi
mkdir -p "$cache"
exec 8>"$cache/simulator.lock"
flock -n 8 || { echo 'Scene adapter cache is already in use' >&2; exit 1; }
export OMNIGIBSON_APPDATA_PATH="$cache/appdata"
export OMNIGIBSON_GLOBAL_CACHE_PATH="$cache/global"
# Isaac shutdown can return zero after an exception; require the completed export event.
/opt/conda/envs/behavior/bin/python -u "$root/behavior_adapter/scene.py" "$@" 2>&1 | tee "$cache/run.log"
grep -q '"event": "scene_adapter_complete"' "$cache/run.log"
