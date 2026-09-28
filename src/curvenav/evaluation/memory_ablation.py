"""Replay one observation: isolate obstacle-memory effects on actor and critic.

This is a counterfactual diagnostic, not a controller or safety certificate.
All variants retain the same four depth observations, goal and flow noise bank.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from curvenav.deployment.interface import _read_request
from curvenav.deployment.runtime import load_policy
from curvenav.types import PolicyCondition


@torch.no_grad()
def compare_memory(policy, condition, current=None):
    if policy.training:
        raise ValueError("memory comparison requires policy.eval()")
    condition.validate()
    variants = {'full': condition.obstacle_memory}
    if current is not None:
        if current.shape != condition.obstacle_memory.shape or current.dtype != torch.bool:
            raise ValueError("current obstacles must match the boolean memory raster")
        variants['current_obstacles'] = current
    variants['empty_obstacle_memory'] = torch.zeros_like(condition.obstacle_memory)
    report, arrays = {}, {}
    base_paths = None
    for name, obstacles in variants.items():
        encoded = policy.encode_condition(replace(condition, obstacle_memory=obstacles))
        paths = policy.sample_candidate_paths(encoded)
        if base_paths is None:
            base_paths = paths
        scores = policy.score_candidate_paths(encoded, condition.point_goal, paths)
        # Hold candidates fixed: score differences here can only come from memory.
        fixed_scores = policy.score_candidate_paths(encoded, condition.point_goal, base_paths)
        selected = scores.argmax(dim=1)
        selected_paths = paths[torch.arange(len(paths), device=paths.device), selected]
        lengths = selected_paths.diff(dim=1).norm(dim=-1).sum(dim=1)
        progress = condition.point_goal.norm(dim=-1) - (condition.point_goal - selected_paths[:, -1]).norm(dim=-1)
        report[name] = {
            'occupied_cells': obstacles.sum(dim=(1, 2)).tolist(),
            'selected_index': selected.tolist(),
            'fixed_bank_selected_index': fixed_scores.argmax(dim=1).tolist(),
            'selected_length_m': lengths.tolist(),
            'selected_direct_goal_progress_m': progress.tolist(),
            'candidate_change_mean_m': (paths - base_paths).norm(dim=-1).mean(dim=(1, 2)).tolist(),
        }
        for key, value in {'paths': paths, 'scores': scores, 'fixed_bank_scores': fixed_scores}.items():
            arrays[f'{name}_{key}'] = value.float().cpu().numpy()
    baseline_scores = arrays['full_fixed_bank_scores']
    for name in variants:
        report[name]['fixed_bank_score_change_mae'] = np.abs(
            arrays[f'{name}_fixed_bank_scores'] - baseline_scores,
        ).mean(axis=1).tolist()
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('request', type=Path, help='exact production request NPZ')
    parser.add_argument('--output', type=Path, required=True, help='new result directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--current-obstacles', type=Path, help='optional bool [B,G,G] NPY from the same sensor tick and world-voxel alignment')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    _, policy = load_policy(args.checkpoint, args.config, args.device)
    request = _read_request(args.request.read_bytes())
    condition = PolicyCondition(**{
        name: torch.as_tensor(value, device=args.device)
        for name, value in request.items()
    })
    current = None
    if args.current_obstacles is not None:
        current = torch.as_tensor(np.load(args.current_obstacles, allow_pickle=False), device=args.device)
    report, arrays = compare_memory(policy, condition, current)
    result = {
        'checkpoint': str(args.checkpoint.resolve()),
        'request': str(args.request.resolve()),
        'interpretation': 'Fixed-bank scores isolate the critic; regenerated curves also change the actor. No collision or reachability guarantee.',
        'variants': report,
    }
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(args.output / 'candidates.npz', **arrays)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
