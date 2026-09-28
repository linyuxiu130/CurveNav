"""Freeze current BEHAVIOR routes and train five epochs at the largest H800 batch.

Run after stopping collection and completing audit_routes.py. This keeps the
source journals intact and writes all derived data to a unique run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = Path('/opt/conda/envs/curvenav-unified/bin/python')
EXPECTED_GPUS = '0,1,2,3,4,5,6,7'
BENCHMARK_STEPS = 40
BENCHMARK_CANDIDATES = (512, 448, 384, 320, 256)
TRAIN_EPOCHS = 5


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open('w') as log:
        subprocess.run(command, cwd=PROJECT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def freeze_routes(root: Path, run: Path) -> tuple[list[Path], int]:
    roots = []
    inventory = []
    for task_root in sorted((root / 'routes').glob('task_*')):
        journal = task_root / 'routes.jsonl'
        if not journal.exists():
            continue
        content = journal.read_bytes()
        records = [json.loads(line) for line in content.splitlines()]
        if not records:
            continue
        if not (task_root / 'dataset_manifest.json').exists():
            raise ValueError(f'Missing route manifest: {task_root}')
        roots.append(task_root)
        inventory.append({'task_root': str(task_root), 'routes': len(records),
                          'train_routes': sum(r['split'] == 'train' for r in records),
                          'validation_routes': sum(r['split'] == 'validation' for r in records),
                          'sha256': hashlib.sha256(content).hexdigest()})
    if not roots:
        raise ValueError('No collected routes')
    count = sum(item['routes'] for item in inventory)
    (run / 'route_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return roots, count


def verify_frozen_routes(run: Path) -> None:
    inventory = json.loads((run / 'route_inventory.json').read_text())
    for item in inventory:
        journal = Path(item['task_root']) / 'routes.jsonl'
        if hashlib.sha256(journal.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Route journal changed after snapshot: {journal}')


def training_config(cache: Path, output: Path, batch: int, epochs: int,
                    steps: int | None = None) -> tuple[Path, int]:
    raw = yaml.safe_load((cache / 'config.yaml').read_text())
    samples = json.loads((cache / 'train/manifest.json').read_text())['samples']
    global_batch = batch * 8
    full_steps = (samples + global_batch - 1) // global_batch
    raw['training'].update(
        per_device_batch_size=batch,
        gradient_accumulation_steps=1,
        samples_per_epoch=(steps or full_steps) * global_batch,
        epochs=epochs,
        num_workers=4,
        prefetch_factor=2,
        checkpoint_every_epochs=1,
        output_dir=str(output),
    )
    raw['training']['warmup_epochs'] = min(raw['training']['warmup_epochs'], epochs // 2)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.yaml'
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config_path, full_steps


def distributed_command(config_path: Path) -> list[str]:
    return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=8', '-m', 'curvenav.training.train', str(config_path)]


def benchmark(cache: Path, run: Path, env: dict[str, str]) -> int:
    results = []
    for batch in BENCHMARK_CANDIDATES:
        output = run / f'benchmark_batch_{batch}'
        config_path, _ = training_config(cache, output, batch, 1,
                                         steps=BENCHMARK_STEPS + 1)
        log_path = output / 'train.log'
        rates = []
        with log_path.open('w') as log:
            process = subprocess.Popen(distributed_command(config_path), cwd=PROJECT,
                                       env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            offset = 0
            # Cold max-autotune on eight GPUs can spend more than 15 minutes
            # compiling before the first timed update is available.
            deadline = time.monotonic() + int(
                os.environ.get('CURVENAV_BENCHMARK_TIMEOUT_SECONDS', '1800')
            )
            try:
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(2)
                    with log_path.open() as reader:
                        reader.seek(offset)
                        chunk = reader.read()
                        offset = reader.tell()
                    for line in chunk.splitlines():
                        if not line.startswith('{'):
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get('step', 0) >= 20 and 'samples_per_second' in event:
                            rates.append(float(event['samples_per_second']))
                    if len(rates) >= 2:
                        break
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        result = {'batch_per_gpu': batch, 'samples_per_second': rates[-1] if len(rates) >= 2 else None,
                  'exit_code': process.returncode, 'log': str(log_path)}
        results.append(result)
        (run / 'benchmark_results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'benchmark': result}), flush=True)
        # The requested objective is the largest stable per-device batch.
        # Candidates are descending; stop once a complete warm run succeeds.
        if len(rates) >= 2:
            return batch
    viable = [item for item in results if item['samples_per_second'] is not None]
    if not viable:
        raise RuntimeError('All H800 training batch benchmarks failed')
    return max(viable, key=lambda item: item['samples_per_second'])['batch_per_gpu']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if not PYTHON.exists():
        raise FileNotFoundError(f'Training environment missing: {PYTHON}')
    audit = json.loads(args.audit.read_text())
    if audit['violation_count']:
        raise ValueError(f"Route audit rejected {audit['violation_count']} routes")
    roots, count = freeze_routes(root, run)
    if count != audit['counts']['routes']:
        raise ValueError(f'Audit covered {audit["counts"]["routes"]} != frozen {count}')
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=EXPECTED_GPUS, PYTHONPATH='/shibo_huang/data/curvenav/runtime/cuda_python_12_6_2:' + str(PROJECT / 'src'),
               XDG_CACHE_HOME='/shibo_huang/data/curvenav/cache',
               CURVENAV_DEPTH_CACHE_DIR='/tmp/curvenav-depth-cache',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CURVENAV_PREPARE_WORKERS=os.environ.get('CURVENAV_PREPARE_WORKERS', '8'),
               CURVENAV_PREPARE_MEMORY_DEVICE='cuda',
               NUMBA_CUDA_USE_NVIDIA_BINDING='1', NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS='0',
               CURVENAV_PREPARE_QUERY_DEVICE='cuda:0',
               CURVENAV_GOAL_DISTANCE_BACKEND='numba',
               CURVENAV_COMPILE_MODE=os.environ.get('CURVENAV_COMPILE_MODE', 'default'),
               TORCHINDUCTOR_COMPILE_THREADS='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1', TORCH_NCCL_HIGH_PRIORITY='1')
    cache = run / 'policy_cache'
    if not (cache / 'manifest.json').exists():
        command = [str(PYTHON), '-m', 'curvenav.data.prepare',
                   '--config', str(root / 'config.yaml'), '--output', str(cache)]
        for task_root in roots:
            command.extend(('--route-root', str(task_root)))
        print(f'Compiling {count} routes from {len(roots)} tasks', flush=True)
        run_logged(command, run / 'compile.log', env)
    verify_frozen_routes(run)
    samples = json.loads((cache / 'train/manifest.json').read_text())['samples']
    validation = json.loads((cache / 'validation/manifest.json').read_text())['samples']
    print(f'Compiled train={samples} validation={validation}', flush=True)
    batch = benchmark(cache, run, env)
    output = run / 'training'
    config_path, steps_per_epoch = training_config(cache, output, batch, TRAIN_EPOCHS)
    plan = {'routes': count, 'tasks': len(roots), 'train_samples': samples,
            'validation_samples': validation, 'batch_per_gpu': batch,
            'global_batch': batch * 8, 'epochs': TRAIN_EPOCHS,
            'steps_per_epoch': steps_per_epoch, 'total_steps': steps_per_epoch * TRAIN_EPOCHS}
    (run / 'training_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'training_start': plan}), flush=True)
    run_logged(distributed_command(config_path), output / 'train.log', env)
    (run / 'complete.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'training_complete': plan}), flush=True)


if __name__ == '__main__':
    main()
