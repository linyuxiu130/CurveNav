"""Resume failed BEHAVIOR tasks as soon as a main 8x3 slot is permanently idle.

The initial scheduler is one-pass. Once it has assigned every task, each
terminal (GPU, slot) can be reused without ever running a fourth simulator on
that GPU. A task is attempted once here; failures require diagnosis, not a
blind retry loop.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/shibo_huang/data/curvenav/datasets/behavior_task_goal_224_v1')
MAIN_RUN = 'taskgoal_20k_work3_20260924'


def events(path):
    if not path.exists():
        return []
    # The writer appends one JSON object per line. Ignore a partial final line
    # while it is being written; it will be present on the next poll.
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def reusable_slots(source, slots, main_finished=False):
    """Return only slots that cannot receive another main-queue assignment."""
    assigned = {row['task_id'] for row in source
                if row['event'] in ('task_started', 'already_complete')}
    if len(assigned) != 100 and not main_finished:
        return []
    if main_finished:
        return slots
    latest = {}
    for row in source:
        if row['event'] in ('task_started', 'task_completed', 'task_failed', 'already_complete'):
            latest[(row['gpu'], row['slot'])] = row['event']
    return [slot for slot in slots
            if latest.get(slot) in ('task_completed', 'task_failed', 'already_complete')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-pid', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    run_root = ROOT/'h800_runs'
    lock = (run_root/'retry_scheduler.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    main_root = run_root/MAIN_RUN
    main_status = main_root/'status.jsonl'
    main_summary = main_root/'summary.json'
    slots = [(entry['gpu'], entry['slot'])
             for entry in json.loads((main_root/'run_info.json').read_text())['gpu_slots']]
    if len(slots) != 24 or len(set(slots)) != 24:
        raise RuntimeError('Expected exactly 8 GPUs × 3 main slots')

    journal = run_root/'retry_idle_slots_status.jsonl'

    def record(event):
        row = {'time': time.time(), **event}
        with journal.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    active = {}
    attempted = set()
    main_abnormal = False
    record({'event': 'retry_coordinator_started', 'main_pid': args.main_pid})
    while True:
        for slot, (task_id, process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            active.pop(slot)
            record({'event': 'retry_task_finished', 'task_id': task_id,
                    'gpu': slot[0], 'slot': slot[1], 'returncode': code,
                    'complete': (ROOT/'routes'/f'task_{task_id:03d}'/'complete.json').exists()})

        source = events(main_status)
        failed = []
        for row in source:
            if row['event'] == 'task_failed':
                failed.append(row['task_id'])

        if main_summary.exists():
            for task_id in json.loads(main_summary.read_text())['unfinished']:
                if task_id not in failed:
                    failed.append(task_id)
        elif not Path(f'/proc/{args.main_pid}').exists():
            if not main_abnormal:
                record({'event': 'main_disappeared_without_summary'})
            main_abnormal = True
        if main_abnormal or any(row['event'] in ('gpu_memory_guard', 'stopping') for row in source):
            if not active:
                return 1
            time.sleep(args.poll_seconds)
            continue

        # Before the main queue empties, a terminal slot may immediately take
        # another main task. All 100 assignments make free slots permanent.
        free = reusable_slots(source, slots, main_summary.exists())
        if free:
            pending = [task_id for task_id in failed if task_id not in attempted
                       and not (ROOT/'routes'/f'task_{task_id:03d}'/'complete.json').exists()]
            for gpu, slot in free:
                key = (gpu, slot)
                if not pending or key in active:
                    continue
                task_id = pending.pop(0)
                run_id = f'taskgoal_20k_work3_retry1_task{task_id:03d}_gpu{gpu}_slot{slot}'
                if (run_root/run_id).exists():
                    attempted.add(task_id)
                    record({'event': 'retry_run_already_exists', 'task_id': task_id,
                            'run_id': run_id})
                    continue
                command = [sys.executable, '-u', 'behavior_adapter/dataset_h800.py',
                           '--root', str(ROOT), '--task-min', str(task_id),
                           '--task-max', str(task_id), '--gpus', str(gpu),
                           '--slots', str(slot), '--slots-per-gpu', '3', '--run-id', run_id]
                log = (run_root/f'{run_id}.log').open('w')
                env = dict(os.environ, PYTHONPATH='src:behavior_adapter')
                process = subprocess.Popen(command, cwd=REPO, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                attempted.add(task_id)
                active[key] = (task_id, process, log)
                record({'event': 'retry_task_started', 'task_id': task_id,
                        'gpu': gpu, 'slot': slot, 'pid': process.pid, 'run_id': run_id})

        if main_summary.exists() and not active:
            remaining = [i for i in range(100)
                         if not (ROOT/'routes'/f'task_{i:03d}'/'complete.json').exists()]
            unattempted = [i for i in remaining if i not in attempted]
            if not unattempted:
                record({'event': 'retry_coordinator_finished', 'remaining': remaining})
                return int(bool(remaining))
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
