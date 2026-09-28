"""Nested 32/64/128/256 raw-neural budgets on a completed detour audit.

Offline diagnostic: no optimizer, curve template, path repair, or deployment.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from curvenav.data.batch import unpack_policy_batch
from curvenav.data.prepared import PreparedPolicyDataset
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.deployment.runtime import load_policy
from curvenav.evaluation.local_memory_audit import gather_depth, raw_paths
from curvenav.training.critic import RouteUtilityTeacher


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    previous = json.loads((args.audit / 'report.json').read_text())
    if not previous['detour_goals'] or previous['progress_map_margin_m'] != .1:
        raise ValueError('sampling comparison requires the margin-connected detour audit')
    config, policy = load_policy(previous['checkpoint'], previous['config'], 'cuda:0')
    cases = np.load(args.audit / 'cases.npz', allow_pickle=False)
    dataset = PreparedPolicyDataset(config.data.root, 'validation', config.data, config.trajectory)
    query = SourceConfigurationSpaceQuery.from_prepared_split(config.data.root, 'validation')
    margin_query = SourceConfigurationSpaceQuery(tuple(
        replace(grid, signed_clearance_m=grid.signed_clearance_m-.1) for grid in query.grids))
    teacher = RouteUtilityTeacher(margin_query, policy.planning_horizon_m, config.data.robot_radius_m)
    rng = torch.Generator().manual_seed(20260928)
    sources = torch.cat((policy.inference_source[:16].flatten(0, 1).cpu(),
                         torch.randn(224, 14, generator=rng)), 0).cuda()
    collected = {key: [] for key in ('paths', 'scores', 'clearance', 'progress', 'length')}
    for offset in range(0, len(cases['indices']), 4):
        batch = dataset.__getitems__(cases['indices'][offset:offset+4].tolist())
        batch['depth'] = gather_depth(dataset, batch['depth_indices'])
        batch['point_goal'] = torch.from_numpy(cases['goal'][offset:offset+4])
        batch = {k: v.cuda() for k, v in batch.items()}
        condition = unpack_policy_batch(batch).condition
        encoded = policy.encode_condition(condition)
        parts = {key: [] for key in collected}
        for source in sources.split(32):
            paths = raw_paths(policy, encoded, 2, source)
            scores = policy.score_candidate_paths(encoded, condition.point_goal, paths)
            labels = teacher(paths, batch)
            values = dict(paths=paths, scores=scores, clearance=labels.clearance_m+.1,
                          progress=labels.progress_m, length=paths.diff(dim=2).norm(dim=-1).sum(2))
            for key, value in values.items():
                parts[key].append(value.float().cpu().numpy())
        for key in parts:
            collected[key].append(np.concatenate(parts[key], 1))
        print(json.dumps({'completed': min(offset+4, len(cases['indices']))}), flush=True)
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    np.savez_compressed(args.output / 'candidates.npz', **arrays)
    report = dict(audit=str(args.audit.resolve()), checkpoint_sha256=previous['checkpoint_sha256'],
                  extra_noise_seed=20260928, neural_steps=2, goal_and_nogoal='equal counts',
                  interpretation='Offline oracle geometry labels only; no training or physical execution.', budgets={})
    for count in (32, 64, 128, 256):
        clear, progress, scores = (arrays[k][:, :count] for k in ('clearance', 'progress', 'scores'))
        good = (clear >= .1) & (progress >= .2)
        selected = scores.argmax(1)
        rows = np.arange(len(selected))
        report['budgets'][str(count)] = dict(cases=len(selected), bank_has_safe_progress=int(good.any(1).sum()),
            selected_safe_progress=int(good[rows, selected].sum()), selected_source_collision=int((clear[rows, selected] < 0).sum()),
            selected_short=int((arrays['length'][rows, selected] < .15).sum()))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
