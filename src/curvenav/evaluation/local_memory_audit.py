"""Paired held-out audit of memory, raw neural proposals and route selection.

Truth maps only LABEL completed model proposals. They never generate a path.
This evaluates the existing static source-map contract, not simulator execution.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from curvenav.data.batch import unpack_policy_batch
from curvenav.data.loader import source_balanced_indices
from curvenav.data.prepared import PreparedPolicyDataset
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.deployment.runtime import load_policy
from curvenav.evaluation.offline import _current_frame_batch
from curvenav.models.policy import repeat_condition
from curvenav.training.critic import RouteUtilityTeacher


def progress_utility(progress, length, clearance, *, horizon, radius, margin):
    """Experimental static-map target; preserve the existing hard margin.

    Unknown-space and tracking risks must be supplied by a future observation /
    execution contract. This function alone is not a deployment safety filter.
    """
    p, arc = progress / horizon, length / horizon
    comfort = torch.asinh((margin - clearance).clamp_min(0) / radius)
    positive = p.tanh() / (1 + .5 * arc + comfort)
    nonpositive = -(-p + .5 * arc).tanh()
    violation = (margin - clearance).clamp_min(0) / radius
    return torch.where(clearance >= margin,
                       torch.where(p > 0, positive, nonpositive),
                       -1 - violation.tanh())


@torch.no_grad()
def raw_paths(policy, encoded, steps, source=None):
    """16 goal + 16 no-goal flows, shared noise across all interventions."""
    count, batch = 32, len(encoded.tokens)
    repeated = replace(
        repeat_condition(encoded, count),
        goal_present=torch.tensor([1., 0.], device=encoded.tokens.device)
        .repeat(batch * count // 2)[:, None],
    )
    memory = policy.trajectory_decoder.project_condition_memory(encoded)
    if source is None:
        source = policy.inference_source[:16].flatten(0, 1)
    state = source.repeat(batch, 1)
    for step in range(steps):
        t = torch.full((len(state),), 1 - step / steps, device=state.device)
        state = state - policy._predict_velocity(state, t, repeated, memory) / steps
    paths, _ = policy.curve_codec.decode(state)
    return paths.unflatten(0, (batch, count))


def summarize(arrays, names, margin, progress_threshold):
    result = {}
    for name in names:
        clearance, progress, length, scores = (
            arrays[f'{name}_{key}'] for key in ('clearance', 'progress', 'length', 'scores')
        )
        # Last two entries are diagnostic stop/expert, never selected by the model.
        safe = clearance[:, :32] >= margin
        useful = safe & (progress[:, :32] >= progress_threshold)
        index = scores[:, :32].argmax(1)
        rows = np.arange(len(index))
        covered = useful.any(1)
        chosen = useful[rows, index]
        target = arrays[f'{name}_old_target']
        feasible_positive = safe & (progress[:, :32] > 0)
        eligible = (clearance[:, 33] >= margin) & (progress[:, 33] >= progress_threshold)
        old_index = target[:, :32].argmax(1)
        old_margin_index = np.where(safe, target[:, :32], -np.inf).argmax(1)
        new_index = arrays[f'{name}_progress_target'][:, :32].argmax(1)
        result[name] = {
            'cases': len(index),
            'bank_has_safe_progress': int(covered.sum()),
            'selected_safe_progress': int(chosen.sum()),
            'selection_miss_given_coverage': int((covered & ~chosen).sum()),
            'selected_source_collision': int((clearance[rows, index] < 0).sum()),
            'selected_below_margin': int((~safe[rows, index]).sum()),
            'selected_length_below_0p15m': int((length[rows, index] < .15).sum()),
            'selected_nonpositive_geodesic_progress': int((progress[rows, index] <= 0).sum()),
            'safe_positive_candidates_old_target_below_stop': int(
                (feasible_positive & (target[:, :32] < target[:, 32:33])).sum()),
            'safe_positive_candidates': int(feasible_positive.sum()),
            'expert_safe_progress': int(((clearance[:, 33] >= margin) &
                                        (progress[:, 33] >= progress_threshold)).sum()),
            'expert_beats_stop_learned': int((scores[:, 33] > scores[:, 32]).sum()),
            'eligible_cases': int(eligible.sum()),
            'eligible_selected_short': int((eligible & (length[rows, index] < .15)).sum()),
            'eligible_short_with_useful_bank': int((eligible & covered & (length[rows, index] < .15)).sum()),
            'all_useful_old_target_not_above_stop_cases': int((covered &
                (np.where(useful, target[:, :32], -np.inf).max(1) <= target[:, 32])).sum()),
            'oracle_old_selects_safe_progress': int((eligible & useful[rows, old_index]).sum()),
            'oracle_old_with_margin_selects_safe_progress': int((eligible & useful[rows, old_margin_index]).sum()),
            'oracle_progress_selects_safe_progress': int((eligible & useful[rows, new_index]).sum()),
        }
    return result


def gather_depth(dataset, indices):
    """Read only requested frames from existing runs; no full-bank cache copy."""
    flat = indices.numpy().reshape(-1)
    spec = dataset.depth_bank
    ends = np.array([run.offset + run.frames for run in spec.runs])
    runs = np.searchsorted(ends, flat, side='right')
    result = np.empty((len(flat), spec.height, spec.width), dtype=np.float16)
    for index in np.unique(runs):
        run = spec.runs[index]
        mask = runs == index
        frames = np.load(run.path, mmap_mode='r', allow_pickle=False)
        result[mask] = frames[flat[mask] - run.offset]
    return torch.from_numpy(result.reshape(*indices.shape, 1, spec.height, spec.width))


@torch.no_grad()
def detour_goals(dataset, indices, teacher, margin_teacher):
    """Fixed 1.5 m goal sweep: certify goal connectivity, reject clear shortcuts.

    Only goal coordinates are constructed. The line tests the need to detour;
    neither it nor a searched route is ever offered to the neural policy.
    """
    selected, goals = [], []
    rejected = dict(goal_not_margin_safe=0, disconnected=0, direct_path_clear=0)
    for index in indices:
        batch = dataset.__getitems__([index])
        for degrees in (0, 90, -90, 180):
            angle = np.deg2rad(degrees)
            goal = torch.tensor([[1.5 * np.cos(angle), 1.5 * np.sin(angle)]], dtype=torch.float32)
            fixed = torch.stack((torch.zeros_like(goal), goal), 1)
            cells = teacher.source_query.point_cells(fixed, batch['source_grid_index'],
                batch['source_origin_xy'], batch['source_yaw_rad']).numpy()[0]
            source = int(batch['source_grid_index'][0])
            grid = teacher.source_query.grids[source]
            target = cells[1]
            if ((target < 0).any() or (target >= grid.signed_clearance_m.shape).any()
                    or grid.signed_clearance_m[tuple(target)] < .1):
                rejected['goal_not_margin_safe'] += 1
                continue
            distance = margin_teacher.goal_distance.query(source, target, cells[:1])[0]
            if not np.isfinite(distance):
                rejected['disconnected'] += 1
                continue
            line = teacher.source_query.query(fixed, batch['source_grid_index'],
                batch['source_origin_xy'], batch['source_yaw_rad'], teacher.horizon)
            if line.minimum_clearance_m.item() >= .1:
                rejected['direct_path_clear'] += 1
                continue
            selected.append(index)
            goals.append(goal[0])
    if not goals:
        raise ValueError('the selected states contain no connected detour goals')
    return selected, torch.stack(goals), rejected


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples-per-source', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--detour-goals', action='store_true', help='fixed goal sweep with blocked direct segment; source truth only selects test cases')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    started = time.monotonic()
    config, policy = load_policy(args.checkpoint, args.config, args.device)
    dataset = PreparedPolicyDataset(config.data.root, 'validation', config.data, config.trajectory)
    indices = source_balanced_indices(dataset.arrays['source_grid_index'], args.samples_per_source)
    query = SourceConfigurationSpaceQuery.from_prepared_split(config.data.root, 'validation')
    teacher = RouteUtilityTeacher(query, policy.planning_horizon_m, config.data.robot_radius_m)
    goals, rejected, margin_teacher = None, None, None
    if args.detour_goals:
        margin_query = SourceConfigurationSpaceQuery(tuple(
            replace(grid, signed_clearance_m=grid.signed_clearance_m - .1)
            for grid in query.grids
        ))
        margin_teacher = RouteUtilityTeacher(margin_query, teacher.horizon, config.data.robot_radius_m)
        indices, goals, rejected = detour_goals(dataset, indices, teacher, margin_teacher)
        print(json.dumps({'detour_cases': len(indices), 'rejected': rejected}), flush=True)
    names = ['full_raw2', 'current_raw2', 'current_tokens_raw2',
             'current_geometry_raw2', 'full_raw8', 'full_production2']
    collected = {'indices': [], 'source': [], 'goal': [], 'memory_cells': []}
    for offset in range(0, len(indices), args.batch_size):
        selected_indices = indices[offset:offset + args.batch_size]
        batch = dataset.__getitems__(selected_indices)
        if goals is not None:
            batch['point_goal'] = goals[offset:offset + args.batch_size]
        batch['depth'] = gather_depth(dataset, batch['depth_indices'])
        batch = {k: v.to(args.device) for k, v in batch.items()}
        condition = unpack_policy_batch(batch).condition
        full = policy.encode_condition(condition)
        current_raster = _current_frame_batch(
            batch, policy.planning_horizon_m, config.data.max_depth_m,
            config.data.robot_geometry)['obstacle_memory']
        current = policy.encode_condition(replace(condition, obstacle_memory=current_raster))
        encodings = [full, current, replace(full, tokens=current.tokens),
                     replace(full, configuration_field=current.configuration_field), full, full]
        expert, _ = policy.curve_codec.decode_values(batch['curve_values'])
        diagnostic = torch.stack((torch.zeros_like(expert), expert), 1)
        fixed_paths = None
        for name, encoded in zip(names, encodings, strict=True):
            paths = (policy.sample_candidate_paths(encoded) if name == 'full_production2'
                     else raw_paths(policy, encoded, 8 if name == 'full_raw8' else 2))
            if fixed_paths is None:
                fixed_paths = paths
            # Match deployment's 32-candidate score batch. Appending diagnostics
            # before scoring changes BF16 GEMM shapes and can break close ties.
            scores = torch.cat((
                policy.score_candidate_paths(encoded, condition.point_goal, paths),
                policy.score_candidate_paths(encoded, condition.point_goal, diagnostic),
            ), 1)
            paths = torch.cat((paths, diagnostic), 1)
            labels = teacher(paths, batch)
            progress = (labels.progress_m if margin_teacher is None
                        else margin_teacher(paths, batch).progress_m)
            length = paths.diff(dim=2).norm(dim=-1).sum(2)
            fixed_scores = policy.score_candidate_paths(encoded, condition.point_goal, fixed_paths)
            values = dict(paths=paths, scores=scores, clearance=labels.clearance_m,
                          progress=progress, original_progress=labels.progress_m, length=length,
                          old_target=labels.score, fixed_scores=fixed_scores,
                          progress_target=progress_utility(
                              progress, length, labels.clearance_m,
                              horizon=policy.planning_horizon_m,
                              radius=config.data.robot_radius_m, margin=.1))
            for key, value in values.items():
                collected.setdefault(f'{name}_{key}', []).append(value.float().cpu().numpy())
        collected['indices'].append(np.array(selected_indices))
        collected['source'].append(batch['source_grid_index'].cpu().numpy())
        collected['goal'].append(condition.point_goal.cpu().numpy())
        collected['memory_cells'].append(torch.stack((condition.obstacle_memory.sum((1, 2)),
                                                     current_raster.sum((1, 2))), 1).cpu().numpy())
        print(json.dumps({'completed': offset + len(selected_indices), 'total': len(indices),
                          'seconds': round(time.monotonic() - started, 1)}), flush=True)
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    np.savez_compressed(args.output / 'cases.npz', **arrays)
    with args.checkpoint.open('rb') as file:
        checkpoint_sha = hashlib.file_digest(file, 'sha256').hexdigest()
    result = {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'config': str(args.config.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': checkpoint_sha, 'dataset_root': str(dataset.root),
        'dataset_samples': len(dataset), 'sample_seed': 0, 'torch_seed': 20260928,
        'samples_per_source': args.samples_per_source, 'sources': int(len(np.unique(arrays['source']))),
        'detour_goals': args.detour_goals, 'goal_sweep_rejected': rejected,
        'progress_map_margin_m': .1 if args.detour_goals else 0.,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'margin_m': .1, 'meaningful_progress_m': .2, 'seconds': time.monotonic() - started,
        'interpretation': 'Source-map offline audit, not physical execution. Current-only interventions keep all four images. Tokens-only and geometry-only keep all metadata fixed. Diagnostic stop/expert never join the selectable bank. Truth geodesic queries label paths only.',
        'reference_scope': ('With goal sweep, entry 33 is the original-goal expert, NOT a new-goal expert; expert eligibility metrics do not certify availability for the new goal.' if args.detour_goals else 'Entry 33 is the same-goal expert.'),
        'variants': summarize(arrays, names, .1, .2),
    }
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
