"""Fixed original40 evaluation using only the native B-spline actor and pure scenario helper."""
from __future__ import annotations

from curvenav.experiment_io import sha
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from curvenav.deployment.bspline_bundle import load_bundle,CANDIDATES,INTEGRATION_STEPS,FORMAT
from curvenav.data.six_frame_memory import MEMORY_CONTRACT
from curvenav.evaluation import bspline_scenarios as scenario
from curvenav.evaluation import bspline_execution as execution
from curvenav.evaluation.bspline_execution import DT_POLICY, advance_path


def goal_at_view(actor, env, case):
    present = bool(case.get('goal_present', False))
    if not present:
        return np.zeros(2, dtype=np.float32), False
    pose = scenario.body_pose(env.view)
    target = np.asarray(case['goal'], dtype=np.float64)
    return ((target - pose[:2, 3]) @ pose[:2, :2]).astype(np.float32), True


def choose(actor, env, case):
    local_goal, goal_present = goal_at_view(actor, env, case)
    condition = actor.encode(env.memory, local_goal, goal_present)
    return local_goal, actor.candidates(condition, local_goal)


def candidate_arrays(bank):
    return (bank['paths'].detach().cpu().numpy().astype(np.float32, copy=False),
        bank['values'].detach().cpu().numpy().astype(np.float32, copy=False),
        bank['scores'].detach().cpu().numpy().astype(np.float32, copy=False))


def save_torch(path, payload):
    pending = path.with_suffix(path.suffix + '.pending')
    torch.save(payload, pending)
    pending.replace(path)


def in_closed_branch(view, case):
    routes = case['branch_routes'][case['known_complete_branch']]
    fork = np.asarray(case['start'])[:2]
    if abs(view[:2] - fork).max() <= .8:
        return False
    for route in routes:
        route = np.asarray(route)
        low = np.minimum(route[:-1], route[1:]) - case['hall'] / 2
        high = np.maximum(route[:-1], route[1:]) + case['hall'] / 2
        if ((view[:2] >= low) & (view[:2] <= high)).all(-1).any():
            return True
    return False


def new_route_progress(view, case):
    route = np.asarray(case['other_route'] if case['expected_open_branch'] == 'other'
                       else case['side_route'])
    direction = route[1] - route[0]
    direction /= np.linalg.norm(direction)
    delta = view[:2] - route[0]
    lateral = abs(delta[0] * direction[1] - delta[1] * direction[0])
    return float(delta @ direction), bool(lateral < case['hall'] / 2)


def local_knowledge_audit(env, item):
    result = {}
    for name in ('known', 'other'):
        direction = np.asarray(item[f'{name}_direction'])
        plane = item[f'{name}_length'] + item['hall'] / 2
        points = env.knowledge.wall_xy
        cross = points[:, 0] * direction[1] - points[:, 1] * direction[0]
        selected = (abs(points @ direction - plane) < 1e-5) & (abs(cross) < item['hall'] / 2 - .05)
        assert selected.any()
        result[f'{name}_wall_fraction'] = float(env.knowledge.wall_seen[selected].mean())
        along = env.knowledge.xy @ direction
        cross_free = env.knowledge.xy[:, 0] * direction[1] - env.knowledge.xy[:, 1] * direction[0]
        free = (along > .2) & (along < plane) & (abs(cross_free) < item['hall'] / 2)
        assert free.any()
        result[f'{name}_free_fraction'] = float(env.knowledge.known[free].mean())
        side_fractions = []
        for sign in (-1, 1):
            side = (abs(cross - sign * item['hall'] / 2) < 1e-5) & \
                (points @ direction > .2) & (points @ direction < plane - .05)
            assert side.any()
            side_fractions.append(float(env.knowledge.wall_seen[side].mean()))
        result[f'{name}_side_wall_fractions'] = side_fractions
    result['remaining_unknown_free_points'] = int((~env.knowledge.known).sum())
    return result


def branch_at_d2(view, case, confirmed):
    for index in confirmed:
        route = np.asarray(case['deadend_arms'][index]['route'])
        if abs(view[:2] - route[0]).max() <= .8:
            continue
        low = np.minimum(route[:-1], route[1:]) - case['hall'] / 2
        high = np.maximum(route[:-1], route[1:]) + case['hall'] / 2
        if ((view[:2] >= low) & (view[:2] <= high)).all(-1).any():
            return index
    return None


def branch_at_junction(view, case, source_row, branch_pairs):
    delta = view[:2] - np.asarray(case['deadend_arms'][source_row['arm']]['fork'])
    if np.linalg.norm(delta) <= .8:
        return None
    if abs(delta[0]) <= abs(delta[1]):
        return 'other_corridor'
    pair = branch_pairs[source_row['fork_id']]
    return pair[int(delta[0] > 0)]


@torch.no_grad()
def episode(actor, item, folder, gpu):
    folder.mkdir(parents=True, exist_ok=True)
    case = item['case']
    panel = item['panel']
    env = scenario.Environment(actor, case, gpu=gpu, timestamp_origin=0. if panel=='J' else DT_POLICY)
    acquired_views = [np.asarray(view, dtype=np.float64).copy()
                      for view in case['prefix_views']]
    acquired_depths = [np.asarray(env.cache[tuple(np.round(view, 7))]).copy()
                       for view in acquired_views]
    timestamp_origin = 0.0 if panel == 'J' else DT_POLICY
    acquired_timestamps = timestamp_origin + np.arange(len(acquired_views), dtype=np.float64) * DT_POLICY
    case_id = item['trial_id']
    budget = item['budget']
    candidate_paths, candidate_values, candidate_scores, selected_ids = [], [], [], []
    trace = []
    status = 'budget_exhausted'
    started = time.monotonic()

    if panel == 'K':
        prefix_known, prefix_walls = env.knowledge.known.copy(), env.knowledge.wall_seen.copy()
        entries, was_closed = 0, False
    elif panel == 'L':
        known_item = {key:value for key,value in case.items() if key != 'phases'}
        initial_audit = local_knowledge_audit(env, known_item)
        assert initial_audit['known_wall_fraction'] >= .9 and initial_audit['other_wall_fraction'] < .1
        fork_step = 0 if item['phase'] == 'returned' else None
        exit_step, exit_branch, exit_known = None, None, None
        entries, was_known = 0, False
    elif panel == 'D2':
        prefix_views = np.asarray(case['prefix_views'])
        audit = {index: scenario.audit_branch(case, index, prefix_views, env.cache, env.k, env.camera)
                 for index in range(len(case['deadend_arms']))}
        confirmed = [index for index, value in audit.items() if value['confirmed']]
        initial_known, initial_walls = env.knowledge.known.copy(), env.knowledge.wall_seen.copy()
        fork = np.asarray(case['deadend_arms'][item['arm']]['fork'])
        fork_step = 0 if item['stage'] == 'returned' else None
        fork_known = initial_known.copy() if fork_step == 0 else None
        exit_step, exit_view, previous_branch = None, None, None
        entries, repeated_m = 0, 0.0
    else:
        target = np.asarray(item['target_world'])
        completed_arms = set(item['source_row']['completed_arms'])
        source_row = item['source_row']
        branch_pairs = item['branch_pairs']

    try:
        for step in range(budget):
            before, old_path, policy_clock = env.view.copy(), env.path, env.clock
            local_goal, bank = choose(actor, env, case)
            paths, values, scores = candidate_arrays(bank)
            selected = int(bank['selected_index'])
            candidate_paths.append(paths)
            candidate_values.append(values)
            candidate_scores.append(scores)
            selected_ids.append(selected)
            executed = advance_path(env, paths[selected])
            accepted = executed['status'] != 'collision'
            if accepted:
                acquired_views.append(env.view.copy())
                acquired_depths.append(np.asarray(env.cache[tuple(np.round(env.view, 7))]).copy())
                acquired_timestamps = np.append(acquired_timestamps, env.clock)

            if panel == 'K':
                closed = in_closed_branch(env.view, case)
                if closed and not was_closed:
                    entries += 1
                was_closed = closed
                progress, inside = new_route_progress(env.view, case)
                new_free = int((env.knowledge.known & ~prefix_known).sum())
                new_walls = int((env.knowledge.wall_seen & ~prefix_walls).sum())
            elif panel == 'L':
                if fork_step is None and np.linalg.norm(env.view[:2]) <= .75:
                    fork_step = step + 1
                known = float(env.view[:2] @ np.asarray(known_item['known_direction']))
                other = float(env.view[:2] @ np.asarray(known_item['other_direction']))
                in_known = fork_step is not None and known > .8 and abs(other) < known_item['hall'] / 2
                in_other = fork_step is not None and other > .8 and abs(known) < known_item['hall'] / 2
                if in_known and not was_known:
                    entries += 1
                was_known = in_known
                if exit_step is None and (in_known or in_other):
                    exit_step, exit_branch, exit_known = step + 1, 'known' if in_known else 'other', env.knowledge.known.copy()
                other_progress = other
                known_progress = known
            elif panel == 'D2':
                if fork_step is None and np.linalg.norm(env.view[:2] - fork) <= .75:
                    fork_step, fork_known = step + 1, env.knowledge.known.copy()
                branch = None if fork_step is None else branch_at_d2(env.view, case, confirmed)
                if branch is not None:
                    repeated_m += env.path - old_path
                    if previous_branch != branch:
                        entries += 1
                previous_branch = branch
                if exit_step is None and fork_step is not None and branch is None and np.linalg.norm(env.view[:2] - fork) >= 1.5:
                    exit_step, exit_view = step + 1, env.view.copy().tolist()
            else:
                branch = branch_at_junction(env.view, case, source_row, branch_pairs) if accepted else None
                goal_distance = float(np.linalg.norm(env.view[:2] - target))

            point = dict(step=step, policy_clock=float(policy_clock), view=before.tolist(),
                next_view=env.view.tolist(), goal_body=local_goal.tolist(),
                input_frame_ids=list(env.memory.selection['indices']),
                candidate_count=CANDIDATES, candidate_source_ids=bank['source_ids'],
                candidate_source_groups=bank['source_groups'], selected_index=selected,
                selected_source_id=bank['source_ids'][selected],
                selected_curve_body_m=paths[selected].tolist(),
                selected_values=values[selected].tolist(), selected_score=float(scores[selected]),
                execution_margin=executed['margin_m'], accepted=accepted,
                executed_substep_views=([part['view'] for part in executed['substeps']] if accepted else []),
                attempted_substep_views=[part.get('view', part.get('attempted_view')) for part in executed['substeps']],
                                substep_margins=[part['sweep_margin_m'] for part in executed['substeps']],
                execution_substeps=executed['substeps'])
            if panel == 'K':
                point.update(inside_initially_closed_branch=closed,
                    expected_route_progress=progress, expected_route_lateral_valid=inside,
                    new_free_points=new_free, new_wall_points=new_walls)
            elif panel == 'L':
                point.update(known_progress=known_progress, other_progress=other_progress,
                    in_known_branch=in_known, in_other_branch=in_other)
            elif panel == 'D2':
                point.update(confirmed_branch=branch,
                    observed_free_points=int(env.knowledge.known.sum()))
            else:
                point.update(branch=branch, goal_dist=goal_distance)
            trace.append(point)

            if not accepted:
                status = 'collision'
                break
            if panel == 'K' and progress >= 1.5 and inside and new_free + new_walls > 0:
                status = 'productive_transition'
                break
            if panel == 'L' and other_progress >= 1.5 and abs(known_progress) < known_item['hall'] / 2:
                status = 'other_1p5_reached'
                break
            if panel == 'J' and goal_distance <= .35:
                status = 'goal_reached'
                break
    finally:
        env.sim.close()

    raw_path = folder / 'raw.pt'
    raw_provenance = dict(source_kind='original40_bspline_rollout', split='fixed_evaluation',
        training_admission=False, trial_id=case_id, panel=panel,
        original_job=item['original_job'], checkpoint=actor.provenance['bundle_path'],
        checkpoint_sha256=actor.provenance['bundle_sha256'],
        bundle_sha256=actor.provenance['bundle_sha256'],
        map_sha256=case.get('map_sha256', case.get('sha256')),
        timestamp_contract='K/L/D2 prefix frames at .5..N*.5; original J Replay prefix at 0..(N-1)*.5; each accepted omni policy cycle adds one frame at env.clock.',
        sensor_source='Only frames already present in env.cache after prefix setup or accepted policy execution; no CNN feature tensors.')
    assert len(acquired_views) == len(acquired_depths) == len(acquired_timestamps)
    save_torch(raw_path, dict(acquired_views=np.asarray(acquired_views, dtype=np.float64),
        depths=np.asarray(acquired_depths), camera_intrinsics=np.repeat(env.k[None],len(acquired_views),axis=0),
        camera_to_body=np.repeat(env.camera[None],len(acquired_views),axis=0),
        body_to_world=np.stack([scenario.body_pose(view) for view in acquired_views]), timestamps=np.asarray(acquired_timestamps, dtype=np.float64),
        case=case, memory_writes=env.memory.writes, input_contract=MEMORY_CONTRACT, task=dict(trial_id=case_id, panel=panel, original_job=item['original_job'],
            budget=budget, scorer=item['scorer']), provenance=raw_provenance))
    raw_sha256 = sha(raw_path)
    arrays_path = folder / 'candidates.npz'
    np.savez_compressed(arrays_path, paths=np.asarray(candidate_paths, dtype=np.float32),
        values=np.asarray(candidate_values, dtype=np.float32), scores=np.asarray(candidate_scores, dtype=np.float32),
        selected=np.asarray(selected_ids, dtype=np.int64))
    common = dict(complete=True, trial_id=case_id, panel=panel, original_job=item['original_job'],
        bundle_sha256=actor.provenance['bundle_sha256'],
        status=status, collision=status == 'collision', steps=len(trace), budget=budget,
        path_m=env.path, minimum_execution_margin_m=min(row['execution_margin'] for row in trace),
        raw_path=str(raw_path.resolve()), raw_sha256=raw_sha256, raw_frames=len(acquired_views),
        candidates_file=str(arrays_path.resolve()), candidates_sha256=sha(arrays_path),
        elapsed_seconds=time.monotonic() - started, trace=trace)
    if panel == 'K':
        common.update(worker=item['original_job']['worker'], pair=case['pair'], kind=case['kind'],
            variant=case['variant'], expected_open_branch=case['expected_open_branch'],
            strict_success=status == 'productive_transition' and entries == 0,
            initially_closed_branch_entries=entries, new_free_m2=trace[-1]['new_free_points'] * .04,
            new_wall_points=trace[-1]['new_wall_points'])
    elif panel == 'L':
        gained = 0 if exit_known is None else int((env.knowledge.known & ~exit_known).sum())
        common.update(worker=item['worker'], phase=item['phase'],
            arrival=status == 'other_1p5_reached',
            strict_success=status == 'other_1p5_reached' and entries == 0,
            collision=status == 'collision', confirmed_known_leg_reentries=entries,
            fork_reached_step=fork_step, exit_fork_step=exit_step, first_exit_branch=exit_branch,
            initial_audit=initial_audit, new_free_after_exit_m2=gained * .04,
            minimum_execution_margin_m=min(row['execution_margin'] for row in trace))
    elif panel == 'D2':
        new_free = 0 if fork_known is None else int((env.knowledge.known & ~fork_known).sum())
        safe_exit = exit_step is not None and status != 'collision'
        memory_success = safe_exit and entries == 0 if item['arm'] in confirmed else None
        common.update(worker=item['worker'], stage=item['stage'], source_arm=item['arm'],
            prefix_audit=audit, initially_confirmed_arms=confirmed, fork_reached_step=fork_step,
            legal_exit_step=exit_step, first_legal_exit_view=exit_view,
            confirmed_branch_reentries=entries, confirmed_branch_reentry_m=repeated_m,
            physical_safe_exit=safe_exit, memory_success=memory_success,
            productive_memory_exit=None if memory_success is None else memory_success and new_free > 0,
            return_steps_reference=item['scorer']['return_steps_reference'],
            new_free_after_fork_m2=new_free * .04,
            new_free_total_m2=int((env.knowledge.known & ~initial_known).sum()) * .04,
            new_wall_points=int((env.knowledge.wall_seen & ~initial_walls).sum()))
    else:
        ever_closed = any(row['branch'] in completed_arms for row in trace if row['branch'] is not None)
        common.update(worker=item['worker'], source_worker=item['source_worker'],
            stage=item['stage'], mode=item['mode'], target=target.tolist(),
            goal_reached=status == 'goal_reached', ever_closed_branch_reentry=ever_closed,
            closed_branch_entry_is_not_navigation_failure=True,
            initial_goal_distance_m=float(np.linalg.norm(np.asarray(trace[0]['view'])[:2] - target)),
            final_goal_distance_m=trace[-1]['goal_dist'])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'result.json').write_text(json.dumps(common, ensure_ascii=False, indent=2) + '\n')
    return common


def snapshot_sources(folder, runtime_assets):
    """Bind evaluator, model and controller code, not only the checkpoint weights."""
    snapshot = folder / 'source_snapshot'
    snapshot.mkdir()
    package = Path(__file__).resolve().parents[1]
    sources = [(source, Path('curvenav') / source.relative_to(package))
               for source in sorted(package.rglob('*.py'))]
    sources += [(runtime_assets, Path('runtime_assets') / runtime_assets.name)]
    recorded = []
    for source, relative in sources:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        recorded.append(dict(path=str(source.resolve()), relative_path=str(relative), sha256=sha(target),
            snapshot_path=str(target.resolve())))
    return recorded


def run_shard(args):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    payload = json.loads(args.input.read_text())
    assert payload['complete'] and payload['contract'] == 'fixed_original40_cases_v1'
    scenario.configure(args.runtime_assets)
    actor = load_bundle(args.checkpoint, device=f'cuda:{args.device}')
    actor.provenance = dict(actor.provenance, bundle_path=str(args.checkpoint.resolve()),
        bundle_sha256=sha(args.checkpoint), bundle_format=FORMAT)
    out = args.output / f'shard_{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    source_snapshot = snapshot_sources(out, args.runtime_assets)
    records, job_reports = [], []
    grouped = {}
    for item in payload['cases']:
        key = (item['original_job']['panel'], item['original_job']['worker'])
        grouped.setdefault(key, []).append(item)
    for job_index, job in enumerate(payload['jobs']['original40_jobs']):
        if job_index % args.shards != args.shard:
            continue
        key = (job['panel'], job['worker'])
        trial_records = []
        for item in grouped[key]:
            folder = out / item['trial_id']
            result = episode(actor, item, folder, args.device)
            records.append(dict(trial_id=item['trial_id'], result_path=str((folder / 'result.json').resolve()),
                result_sha256=sha(folder / 'result.json'), raw_path=result['raw_path'],
                raw_sha256=result['raw_sha256'], reused=False))
            trial_records.append(item['trial_id'])
        job_reports.append(dict(job_index=job_index, **job, trials=trial_records))
        print(json.dumps(dict(shard=args.shard, panel=job['panel'], worker=job['worker'],
            cases=trial_records), ensure_ascii=False), flush=True)
    manifest = dict(complete=True, shard=args.shard, shards=args.shards,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        numerical_contract='deterministic_algorithms_true_warn_only_bf16_fp32_v1',
        integration_steps=actor.policy.integration_steps,
        input_path=str(args.input.resolve()), input_sha256=sha(args.input),
        checkpoint=actor.provenance['bundle_path'],
        checkpoint_sha256=actor.provenance['bundle_sha256'],
        bundle_sha256=actor.provenance['bundle_sha256'],
        action_codec='native B-spline 16 candidate paths; no VW conversion',
        candidate_selection='raw native Q argmax over 16 candidates; no safety filter',
        executor=execution.EXECUTOR,
        actor_provenance=actor.provenance,
        runtime_assets_path=str(scenario.RUNTIME_ASSETS_PATH.resolve()),
        runtime_assets_sha256=scenario.RUNTIME_ASSETS_SHA256,
        collision_sampling=dict(method='all traversed reference-polyline segments',guard_m=0.),
        original_task_identity_and_thresholds_unchanged=True,
        original_12s_vw_horizon_metrics_comparable=False,
        source_snapshot=source_snapshot, jobs=job_reports, results=records,
        cases=len(records), elapsed_seconds=time.monotonic() - args.started)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(complete=True, shard=args.shard, cases=len(records),
        output=str(out), seconds=manifest['elapsed_seconds']), ensure_ascii=False), flush=True)


def summarize(args):
    input_payload = json.loads(args.input.read_text())
    case_by_id = {item['trial_id']:item for item in input_payload['cases']}
    results, identities, numerical_modes = [], [], set()
    source_identity = None
    checkpoint_sha = sha(args.checkpoint)
    for shard in range(args.shards):
        manifest_path = args.output / f'shard_{shard}/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['complete'] and manifest['input_sha256'] == sha(args.input)
        assert manifest['checkpoint_sha256'] == checkpoint_sha, 'Evaluation checkpoint differs'
        current_sources = {(record['relative_path'], record['sha256']) for record in manifest['source_snapshot']}
        for record in manifest['source_snapshot']:
            assert sha(Path(record['snapshot_path'])) == record['sha256']
        if source_identity is None:
            source_identity = current_sources
        assert current_sources == source_identity, 'Evaluation shards use different source code'
        numerical_modes.add(manifest['deterministic_algorithms'])
        for record in manifest['results']:
            result_path = Path(record['result_path'])
            assert sha(result_path) == record['result_sha256']
            assert sha(record['raw_path']) == record['raw_sha256']
            result = json.loads(result_path.read_text())
            assert result['trial_id'] == record['trial_id']
            expected_panel = case_by_id[result['trial_id']]['panel']
            assert result['panel'] == expected_panel
            identities.append(result['trial_id'])
            results.append(result)
    assert len(numerical_modes)==1, 'Do not mix historical False and deterministic True evaluation shards'
    assert len(identities) == len(set(identities)) == 40
    expected = {item['trial_id'] for item in input_payload['cases']}
    assert set(identities) == expected
    by_panel = {panel:[result for result in results if result['panel'] == panel]
                for panel in ('K','L','D2','J')}
    strict = dict(K=sum(bool(row['strict_success']) for row in by_panel['K']),
        L=sum(bool(row['strict_success']) for row in by_panel['L']),
        D2=sum(bool(row['physical_safe_exit'] and not row['collision']
            and row['confirmed_branch_reentries'] == 0
            and (row['stage'] != 'returned' or row['memory_success'] is True))
            for row in by_panel['D2']),
        J=sum(bool(row['goal_reached'] and not row['collision']) for row in by_panel['J']))
    deterministic = numerical_modes.pop()
    summary = dict(complete=True, trials=40, jobs=24, deterministic_algorithms=deterministic,
        numerical_contract=('deterministic_algorithms_true_warn_only_bf16_fp32_v1' if deterministic
            else 'historical_deterministic_algorithms_false'), panels={key:len(value) for key,value in by_panel.items()},
        strict_passed=sum(strict.values()), strict_by_panel=strict,
        qualified_d2_memory_success=dict(confirmed=sum(row['memory_success'] is not None for row in by_panel['D2']),
            passed=sum(row['memory_success'] is True for row in by_panel['D2']),
            unconfirmed_null=sum(row['memory_success'] is None for row in by_panel['D2'])),
        candidates=CANDIDATES, integration_steps=INTEGRATION_STEPS, codec='native B-spline', executor=execution.EXECUTOR, shard_manifests=args.shards,
        cases=[dict(trial_id=row['trial_id'], panel=row['panel'], status=row['status'],
            strict_success=row.get('strict_success'), physical_safe_exit=row.get('physical_safe_exit'),
            memory_success=row.get('memory_success'), goal_reached=row.get('goal_reached'),
            reused_baseline='reused_baseline' in row) for row in results])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(complete=True, trials=40, strict_passed=summary['strict_passed'],
        strict_by_panel=strict, summary=str(args.output / 'summary.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'summarize'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--runtime-assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    args.started = time.monotonic()
    if args.mode == 'run':
        run_shard(args)
    else:
        summarize(args)
