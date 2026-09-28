#!/usr/bin/env bash
# Recurring GPT-6 Luna health checks for the shared BEHAVIOR generation.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="/shibo_huang/data/curvenav/datasets/behavior_task_goal_224_v1"
run_root="$data_root/h800_runs"
mkdir -p "$run_root"
exec 9>"$run_root/monitor_generation.lock"
flock -n 9 || { echo 'monitor already running' >&2; exit 1; }
exec > >(tee -a "$run_root/monitor_generation.log") 2>&1

export CODEX_HOME="/shibo_huang/.codex-h800"

check_complete() {
  /opt/conda/bin/python - "$data_root" "$run_root/monitor_status.json" <<'PY'
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1]) / 'routes'
status_path = Path(sys.argv[2])
complete = []
total_routes = 0
invalid = {}
for task_id in range(100):
    task_root = root / f'task_{task_id:03d}'
    journal = task_root / 'routes.jsonl'
    marker = task_root / 'complete.json'
    try:
        records = [json.loads(line) for line in journal.open()] if journal.exists() else []
        total_routes += len(records)
        valid = (marker.exists() and len(records) == 200
                 and len({record['route_directory'] for record in records}) == 200
                 and len({(record['split'], record['plan_route_id']) for record in records}) == 200
                 and sum(record['goal_type'] == 'task_target' for record in records) == 160
                 and sum(record['goal_type'] == 'scene_random' for record in records) == 40
                 and all(record['result']['success']
                         and record['result']['minimum_grid_clearance_m'] >= .10 - 1e-6
                         and record['result']['maximum_base_z_m'] - record['result']['minimum_base_z_m'] <= .16
                         and record['result']['goal_error_m'] < .005
                         and (record['goal_type'] != 'task_target' or record.get('target'))
                         and (task_root/record['route_directory']).is_dir()
                         for record in records)
                 and json.loads(marker.read_text())['routes'] == 200)
    except (OSError, ValueError, KeyError) as error:
        valid = False
        invalid[str(task_id)] = str(error)
    if valid:
        complete.append(task_id)
status = {'checked_at': time.time(), 'completed_tasks': len(complete),
          'total_routes': total_routes,
          'remaining': [i for i in range(100) if i not in complete],
          'invalid': invalid}
status_path.write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
raise SystemExit(0 if len(complete) == 100 else 1)
PY
}

verified_complete() {
  # A marker/count check alone does not prove the 3D expert-route contract.
  # The metadata finalizer touches only closed 200-route journals.
  /opt/conda/envs/behavior/bin/python "$repo/behavior_adapter/finalize_route_provenance.py" \
    --root "$data_root" --apply || return 1
  /opt/conda/envs/behavior/bin/python "$repo/behavior_adapter/audit_routes.py" \
    --root "$data_root" --output "$run_root/final_audit.json" || return 1
}

cd "$repo"
printf '%s monitor_started model=gpt-6-luna effort=high interval=1800s\n' "$(date -u +%FT%TZ)"
while true; do
  if check_complete && verified_complete; then
    printf '%s all_100_tasks_complete_monitor_stopping\n' "$(date -u +%FT%TZ)"
    printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)" > "$run_root/monitor_generation_complete"
    exit 0
  fi
  printf '%s check_start\n' "$(date -u +%FT%TZ)"
  set +e
  /usr/local/bin/codex exec \
    --model gpt-6-luna \
    --config model_reasoning_effort=high \
    --dangerously-bypass-approvals-and-sandbox \
    --ephemeral \
    --cd "$repo" \
    --output-last-message "$run_root/monitor_last_message.md" \
    - < "$repo/behavior_adapter/monitor_generation_prompt.md"
  result=$?
  set -e
  printf '%s check_end exit_status=%s\n' "$(date -u +%FT%TZ)" "$result"
  if check_complete && verified_complete; then
    printf '%s all_100_tasks_complete_monitor_stopping\n' "$(date -u +%FT%TZ)"
    printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)" > "$run_root/monitor_generation_complete"
    exit 0
  fi
  sleep 1800
done
