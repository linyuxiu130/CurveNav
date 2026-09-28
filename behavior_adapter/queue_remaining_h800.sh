#!/usr/bin/env bash
# Drain the current collectors, then fill every unfinished shared task on H800.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="/shibo_huang/data/curvenav/datasets/behavior_generic_224_v1"
queue_root="$data_root/h800_runs/tmux_remaining_queue"
mkdir -p "$queue_root"
exec > >(tee -a "$queue_root/queue.log") 2>&1

started="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'queue_started_utc=%s\n' "$started"
printf 'state=waiting_for_existing_collectors\n' > "$queue_root/state"

# Existing launchers predate this tmux queue and already own their GPU cache
# slots. Wait for them to exit before reusing any slot.
while pgrep -f 'behavior_adapter/(dataset_h800\.py|scene\.py|run_scene\.sh)' >/dev/null; do
  printf '%s waiting_for_existing_collectors\n' "$(date -u +%FT%TZ)"
  sleep 30
done

cd "$repo"
run_id="tmux_remaining_${started}"
printf 'state=running\nrun_id=%s\n' "$run_id" > "$queue_root/state"
printf '%s starting_all_unfinished_tasks run_id=%s\n' "$(date -u +%FT%TZ)" "$run_id"

set +e
/opt/conda/bin/python -u behavior_adapter/dataset_h800.py \
  --task-min 0 --task-max 99 \
  --gpus 0,1,2,3,4,5,6,7 --slots 0,1,2 --slots-per-gpu 3 \
  --run-id "$run_id"
status=$?
set -e

printf 'state=finished\nrun_id=%s\nexit_status=%s\n' "$run_id" "$status" > "$queue_root/state"
printf '%s queue_finished exit_status=%s\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"
