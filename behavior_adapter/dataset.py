"""Plan, collect on two GPUs, and compile through CurveNav's existing cache builder."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

from collect import DATA

REPO = Path(__file__).resolve().parents[1]
META = Path('/mnt/dataset/Benchmark/Behavior2026/2026-challenge-demos/meta')
SCENES = Path('/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K/datasets/2026-challenge-task-instances/scenes')
PYTHON = '/opt/conda/envs/curvenav-unified/bin/python'


def candidate_instance_pool(selected, group_index, group_count):
    """Reserve distinct initializations across route groups in one split."""
    return [int(selected[group_index + stratum * group_count])
            for stratum in range(5)
            if group_index + stratum * group_count < len(selected)]


def create_plans(root, preflight):
    tasks = [json.loads(s) for s in (META/'tasks.jsonl').read_text().splitlines()]
    plans = []
    for task in tasks:
        index = task['task_index']
        if preflight and index not in (0, 35):
            continue
        rng = np.random.default_rng(20260923+index)
        files = SCENES.glob(f"*/json/*_task_{task['task_name']}_instances/*-tro_state.json")
        instances = sorted({int(p.name.removesuffix('_template-tro_state.json').rsplit('_', 1)[1]) for p in files})
        if len(instances) < 200:
            raise ValueError(f"{task['task_name']}: only {len(instances)} train initializations")
        rng.shuffle(instances)
        train, validation = instances[:-20], instances[-20:]
        groups = []
        route_types = ['task_target']*144 + ['scene_random']*36
        rng.shuffle(route_types)
        if preflight:
            route_types = ['task_target', 'scene_random']
        validation_types = ['task_target'] if preflight else ['task_target']*16+['scene_random']*4
        for split, selected, requested in [('train', train, route_types), ('validation', validation, validation_types)]:
            group_count = (len(requested)+4)//5
            for offset in range(0, len(requested), 5):
                group_index = offset//5
                candidates = candidate_instance_pool(selected, group_index, group_count)
                groups.append({'instance': int(selected[group_index]),
                               'candidate_instances': candidates,
                               'split': split,
                               'routes': [{'id': f'route_{j:04d}', 'band': 'any', 'goal_type': requested[j]}
                                          for j in range(offset, min(offset+5, len(requested)))]})
        plan = {'task': task['task_name'], 'task_index': index, 'seed': 20260923+index,
                'goal_contract': 'behavior_task_object_v1',
                'route_root': str(root/'routes'/f'task_{index:03d}'), 'instances': groups}
        file = root/'plans'/f'task_{index:03d}.json'
        file.write_text(json.dumps(plan, indent=2)+'\n')
        plans.append(file)
    config = yaml.safe_load((REPO/'configs/base.yaml').read_text())
    config['data'] = asdict(DATA)
    config['training']['output_dir'] = str(root/'training')
    (root/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (root/'plan.json').write_text(json.dumps({'target_train_routes': 4 if preflight else 18000,
                                            'target_validation_routes': 2 if preflight else 2000,
                                            'plans': [str(p) for p in plans],
                                            'split': '20 held-out local training initializations per task; remaining local training initializations supply training routes'}, indent=2)+'\n')
    return plans


def compile_cache(root, completed):
    if (root/'INVALID_FOR_TASK_GOAL_TRAINING.json').exists():
        raise ValueError('Unsafe generic rollout dataset is quarantined and cannot be compiled')
    for plan_path in completed:
        plan = json.loads(plan_path.read_text())
        if plan.get('goal_contract') != 'behavior_task_object_v1':
            raise ValueError(f'Plan has no task-goal contract: {plan_path}')
        route_root = Path(plan['route_root'])
        records = [json.loads(line) for line in (route_root/'routes.jsonl').read_text().splitlines()]
        expected = [route for group in plan['instances'] for route in group['routes']]
        if len(records) != len(expected) or len({record['route_directory'] for record in records}) != len(expected):
            raise ValueError(f'Route count or IDs do not match plan: {plan_path}')
        for record in records:
            result = record['result']
            if (record.get('goal_type') not in ('task_target', 'scene_random')
                    or (record['goal_type'] == 'task_target' and not record.get('target'))
                    or not result['success']
                    or result['minimum_grid_clearance_m'] < .10 - 1e-6
                    or result['maximum_base_z_m'] - result['minimum_base_z_m'] > .16
                    or result['goal_error_m'] >= .005):
                raise ValueError(f'Unsafe or ungrounded route in {route_root}: {record.get("route_id")}')
    destination = root/f'policy_cache_{len(completed):03d}_tasks'
    command = [PYTHON, '-m', 'curvenav.data.prepare', '--config', str(root/'config.yaml'), '--output', str(destination)]
    for plan_path in sorted(completed):
        command += ['--route-root', json.loads(plan_path.read_text())['route_root']]
    env = dict(os.environ, PYTHONPATH=str(REPO/'src'), XDG_CACHE_HOME='/shibo_huang/data/curvenav/cache',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (root/f'compile_{len(completed):03d}.log').open('w') as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (root/'latest_cache.txt').write_text(str(destination)+'\n')
    return str(destination)


def collect_worker(root, plans, gpu):
    for plan_path in plans:
        plan = json.loads(plan_path.read_text())
        job = root/'jobs'/f"task_{plan['task_index']:03d}"
        job.mkdir(exist_ok=True)
        attempt = job/f"attempt_{time.time_ns()}"
        attempt.mkdir()
        with (attempt/'process.log').open('w') as log:
            subprocess.run(['bash', 'behavior_adapter/run_scene.sh', '--task', plan['task'],
                            '--instance', str(plan['instances'][0]['instance']),
                            '--output', str(attempt/'audit'), '--collect-plan', str(plan_path)],
                           cwd=REPO, env=dict(os.environ, GPU_ID=str(gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)
        with (root/f'worker_{gpu}_completed.jsonl').open('a') as stream:
            stream.write(json.dumps({'plan': str(plan_path)})+'\n')
        print(json.dumps({'event': 'task_collected', 'gpu': gpu, 'task': plan['task']}), flush=True)
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.resume:
        plans = [Path(p) for p in json.loads((root/'plan.json').read_text())['plans']]
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root/'plans').mkdir()
        (root/'jobs').mkdir()
        plans = create_plans(root, args.preflight)
    completed = [p for p in plans if (Path(json.loads(p.read_text())['route_root'])/'complete.json').exists()]
    remaining = [p for p in plans if p not in completed]
    queue = iter(remaining)
    cache = (root/'latest_cache.txt').read_text().strip() if (root/'latest_cache.txt').exists() else None
    published = int(Path(cache).name.split('_')[2]) if cache else 0
    cache_jobs = []
    with ThreadPoolExecutor(max_workers=1) as cache_pool, ThreadPoolExecutor(max_workers=2) as pool:
        pending = {}
        for gpu in (0, 1):
            first = next(queue, None)
            if first is not None:
                pending[pool.submit(collect_worker, root, [first], gpu)] = gpu
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for job in finished:
                gpu = pending.pop(job)
                completed.extend(job.result())
                following = next(queue, None)
                if following is not None:
                    pending[pool.submit(collect_worker, root, [following], gpu)] = gpu
            if any(published < milestone <= len(completed) for milestone in (2, 10, 25, 50, len(plans))):
                cache_jobs.append(cache_pool.submit(compile_cache, root, tuple(completed)))
                published = len(completed)
        if published < len(plans):
            cache_jobs.append(cache_pool.submit(compile_cache, root, tuple(completed)))
        for job in cache_jobs:
            cache = job.result()
    (root/'complete.json').write_text(json.dumps({'cache': cache, 'tasks': len(plans)})+'\n')
    print(json.dumps({'event': 'dataset_complete', 'cache': cache}), flush=True)


if __name__ == '__main__':
    main()
