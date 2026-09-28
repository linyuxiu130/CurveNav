"""Collect a disjoint range of the shared CurveNav BEHAVIOR plans on H800 GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import threading
import time


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/shibo_huang/data/curvenav/datasets/behavior_task_goal_224_v1')
BEHAVIOR_ROOT = '/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K-v3.9.2-baidu'
BEHAVIOR_DATA = '/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K/datasets'


def parse_ids(value, minimum, maximum):
    result = [int(token) for token in value.split(',')]
    if not result or len(result) != len(set(result)) or any(not minimum <= item <= maximum for item in result):
        raise argparse.ArgumentTypeError(f'Expected unique comma-separated IDs in {minimum}-{maximum}')
    return result


def cpu_range(gpu, slot, slots_per_gpu):
    base = gpu * 20 if gpu < 4 else 90 + (gpu - 4) * 20
    start = base + slot * 20 // slots_per_gpu
    end = base + (slot + 1) * 20 // slots_per_gpu - 1
    return f'{start}-{end}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--task-min', type=int, required=True)
    parser.add_argument('--task-max', type=int, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots', default='0,1,2')
    parser.add_argument('--slots-per-gpu', type=int, default=3)
    parser.add_argument('--skip-tasks', default='', help='Comma-separated task IDs to defer')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run_id):
        parser.error('run-id may contain only letters, numbers, dot, underscore, and dash')
    root = args.root.resolve()
    if (root/'INVALID_FOR_TASK_GOAL_TRAINING.json').exists():
        parser.error('This rollout dataset is quarantined; generate into the task-goal dataset instead')
    gpus = parse_ids(args.gpus, 0, 7)
    slots = parse_ids(args.slots, 0, args.slots_per_gpu - 1)
    skipped = set(parse_ids(args.skip_tasks, 0, 99)) if args.skip_tasks else set()
    if not 1 <= args.slots_per_gpu <= 5 or not 0 <= args.task_min <= args.task_max <= 99:
        parser.error('Invalid task range or slots-per-gpu')
    plans = [root / 'plans' / f'task_{index:03d}.json'
             for index in range(args.task_max, args.task_min - 1, -1) if index not in skipped]
    if any(not plan.is_file() for plan in plans):
        parser.error('One or more shared task plans are missing')
    if any(json.loads(plan.read_text()).get('goal_contract') != 'behavior_task_object_v1'
           for plan in plans):
        parser.error('Every plan must declare the BEHAVIOR task-object goal contract')
    slots_to_run = [(gpu, slot, cpu_range(gpu, slot, args.slots_per_gpu))
                    for gpu in gpus for slot in slots]
    settings = {'task_min': args.task_min, 'task_max': args.task_max,
                'task_order': [int(plan.stem[-3:]) for plan in plans],
                'gpu_slots': [{'gpu': gpu, 'slot': slot, 'cpus': cpus}
                              for gpu, slot, cpus in slots_to_run],
                'root': str(root), 'run_id': args.run_id,
                'skipped_tasks': sorted(skipped)}
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return 0

    run_root = root / 'h800_runs' / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    settings.update(pid=os.getpid(), host=socket.gethostname(), started_at=time.time())
    (run_root / 'run_info.json').write_text(json.dumps(settings, indent=2) + '\n')
    pending = queue.Queue()
    for plan in plans:
        pending.put(plan)
    stop = threading.Event()
    active = {}
    active_lock = threading.Lock()
    event_lock = threading.Lock()
    failures = []
    status_path = run_root / 'status.jsonl'

    def record(event):
        event = {'time': time.time(), **event}
        with event_lock:
            with status_path.open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            print(json.dumps(event), flush=True)

    def terminate_active():
        with active_lock:
            processes = list(active.values())
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def handle_signal(signum, _frame):
        stop.set()
        record({'event': 'stopping', 'signal': signum})
        terminate_active()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def monitor_gpu():
        with (run_root / 'gpu.csv').open('w', newline='', buffering=1) as stream:
            writer = csv.writer(stream)
            writer.writerow(['time', 'gpu', 'memory_mib', 'util_pct'])
            while not stop.is_set():
                output = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if output.returncode == 0:
                    for line in output.stdout.splitlines():
                        fields = [item.strip() for item in line.split(',')]
                        if len(fields) != 3 or not fields[0].isdigit():
                            continue
                        gpu, memory = int(fields[0]), int(fields[1])
                        if gpu in gpus:
                            writer.writerow([time.time(), *fields])
                            if memory >= 75000:
                                record({'event': 'gpu_memory_guard', 'gpu': gpu,
                                        'memory_mib': memory})
                                stop.set()
                                terminate_active()
                                return
                stop.wait(5)

    def worker(gpu, slot, cpus):
        while not stop.is_set():
            try:
                plan_path = pending.get_nowait()
            except queue.Empty:
                return
            plan = json.loads(plan_path.read_text())
            task_id = plan['task_index']
            route_root = Path(plan['route_root'])
            if (route_root / 'complete.json').is_file():
                record({'event': 'already_complete', 'task_id': task_id,
                        'gpu': gpu, 'slot': slot})
                continue
            attempt = root / 'jobs' / f'task_{task_id:03d}' / f'h800_{args.run_id}_gpu{gpu}_slot{slot}'
            attempt.parent.mkdir(parents=True, exist_ok=True)
            attempt.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, GPU_ID=str(gpu), CURVENAV_GPU_SLOT=str(slot),
                       CURVENAV_BEHAVIOR_ROOT=BEHAVIOR_ROOT,
                       CURVENAV_OMNIGIBSON_DATA_PATH=BEHAVIOR_DATA)
            command = ['taskset', '--cpu-list', cpus, 'bash', 'behavior_adapter/run_scene.sh',
                       '--task', plan['task'], '--instance', str(plan['instances'][0]['instance']),
                       '--output', str(attempt / 'audit'), '--collect-plan', str(plan_path)]
            record({'event': 'task_started', 'task_id': task_id, 'task': plan['task'],
                    'gpu': gpu, 'slot': slot, 'cpus': cpus})
            with (attempt / 'process.log').open('w') as log:
                process = subprocess.Popen(command, cwd=REPO, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                with active_lock:
                    active[(gpu, slot)] = process
                returncode = process.wait()
                with active_lock:
                    active.pop((gpu, slot), None)
            if stop.is_set():
                return
            if returncode != 0 or not (route_root / 'complete.json').is_file():
                failures.append(task_id)
                record({'event': 'task_failed', 'task_id': task_id, 'gpu': gpu,
                        'slot': slot, 'returncode': returncode, 'log': str(attempt / 'process.log')})
                continue
            record({'event': 'task_completed', 'task_id': task_id,
                    'gpu': gpu, 'slot': slot})

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    with ThreadPoolExecutor(max_workers=len(slots_to_run)) as pool:
        futures = [pool.submit(worker, gpu, slot, cpus) for gpu, slot, cpus in slots_to_run]
        for future in futures:
            future.result()
    stop.set()
    monitor.join(timeout=10)
    unfinished = [int(plan.stem[-3:]) for plan in plans
                  if not (Path(json.loads(plan.read_text())['route_root']) / 'complete.json').is_file()]
    summary = {'completed': len(plans) - len(unfinished), 'total': len(plans),
               'unfinished': unfinished, 'failures': failures, 'stopped': bool(unfinished)}
    (run_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    record({'event': 'run_finished', **summary})
    return int(bool(unfinished))


if __name__ == '__main__':
    raise SystemExit(main())
