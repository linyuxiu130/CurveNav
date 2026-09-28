"""One small, anchored flow-improvement experiment, without deploying weights."""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from curvenav.data.batch import unpack_policy_batch
from curvenav.data.prepared import PreparedPolicyDataset
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.deployment.runtime import load_policy
from curvenav.evaluation.local_memory_audit import gather_depth, progress_utility, raw_paths
from curvenav.training.critic import RouteUtilityTeacher
from curvenav.training.group_flow import group_flow_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ordinary_audit', type=Path)
    parser.add_argument('detour_audit', type=Path)
    parser.add_argument('sampling_audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    detour = np.load(args.detour_audit/'cases.npz')
    ordinary = np.load(args.ordinary_audit/'cases.npz')
    bank = np.load(args.sampling_audit/'candidates.npz')
    meta = json.loads((args.detour_audit/'report.json').read_text())
    sampling_meta = json.loads((args.sampling_audit/'report.json').read_text())
    ordinary_meta = json.loads((args.ordinary_audit/'report.json').read_text())
    if (Path(sampling_meta['audit']) != args.detour_audit.resolve()
            or sampling_meta['checkpoint_sha256'] != meta['checkpoint_sha256']
            or ordinary_meta['checkpoint_sha256'] != meta['checkpoint_sha256']):
        raise ValueError('candidate bank and both audits must refer to the same checkpoint and detour cases')
    with Path(meta['checkpoint']).open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('the reference checkpoint changed after candidate generation')
    config, policy = load_policy(meta['checkpoint'], meta['config'], 'cuda:0')
    policy.requires_grad_(False)
    names = []
    for name, parameter in policy.trajectory_decoder.named_parameters():
        if 'path_geometry_embedding.' in name or '.cross_attention.relative_bias.' in name:
            parameter.requires_grad_(True)
            names.append(name)
    reference = deepcopy(policy).eval()  # detached targets, never in optimizer
    parameters = [p for p in policy.parameters() if p.requires_grad]
    mix = .05
    # Adam normalizes gradient magnitudes; scale the step as well as the
    # distribution change so zero improvement strength really means no update.
    optimizer = torch.optim.AdamW(parameters, lr=mix*2e-4, weight_decay=0.)
    dataset = PreparedPolicyDataset(config.data.root, 'validation', config.data, config.trajectory)
    split_root = Path(config.data.root)/'validation'
    manifest = json.loads((split_root/'manifest.json').read_text())
    hashes = []
    for item in manifest['source_configuration_space']['grids']:
        with np.load(split_root/item['file']) as grid:
            digest = hashlib.sha256()
            for key in sorted(grid.files):
                digest.update(key.encode())
                digest.update(grid[key].tobytes())
            hashes.append(digest.hexdigest())
    source_query = SourceConfigurationSpaceQuery.from_prepared_split(config.data.root, 'validation')
    margin_query = SourceConfigurationSpaceQuery(tuple(
        replace(grid, signed_clearance_m=grid.signed_clearance_m-.1) for grid in source_query.grids))
    teacher = RouteUtilityTeacher(margin_query, policy.planning_horizon_m, config.data.robot_radius_m)
    ordinary_teacher = RouteUtilityTeacher(source_query, policy.planning_horizon_m, config.data.robot_radius_m)
    groups = np.array(hashes)[detour['source']]
    unique = np.unique(groups)
    rng = np.random.default_rng(20260928)
    rng.shuffle(unique)
    train_groups = unique[:len(unique)//2]
    train = np.flatnonzero(np.isin(groups, train_groups))
    holdout = np.flatnonzero(~np.isin(groups, train_groups))
    ordinary_groups = np.array(hashes)[ordinary['source']]
    replay = np.flatnonzero(np.isin(ordinary_groups, train_groups))
    retain = np.flatnonzero(~np.isin(ordinary_groups, train_groups))
    # Fixed 64-state retention probe on maps not used by either training source.
    retain = rng.choice(retain, min(64, len(retain)), replace=False)

    def batch_for(data, ids):
        batch = dataset.__getitems__(data['indices'][ids].tolist())
        batch['depth'] = gather_depth(dataset, batch['depth_indices'])
        batch['point_goal'] = torch.from_numpy(data['goal'][ids])
        return {k: v.cuda() for k, v in batch.items()}

    inverse = torch.linalg.pinv(policy.curve_codec.basis[:, 1:])

    def controls(paths):
        paths = torch.as_tensor(paths, device='cuda')
        values = torch.einsum('cp,bkpd->bkcd', inverse, paths).flatten(2)
        recovered, _ = policy.curve_codec.decode_values(values.flatten(0, 1))
        torch.testing.assert_close(recovered, paths.flatten(0, 1), atol=2e-5, rtol=2e-5)
        return values

    @torch.no_grad()
    def evaluate(data, ids, truth):
        outputs = {k: [] for k in ('paths', 'scores', 'clearance', 'progress')}
        for offset in range(0, len(ids), 4):
            batch = batch_for(data, ids[offset:offset+4])
            condition = unpack_policy_batch(batch).condition
            encoded = policy.encode_condition(condition)
            paths = raw_paths(policy, encoded, 2)
            scores = policy.score_candidate_paths(encoded, condition.point_goal, paths)
            labels = (teacher if truth else ordinary_teacher)(paths, batch)
            values = dict(paths=paths, scores=scores, clearance=labels.clearance_m+(.1 if truth else 0.),
                          progress=labels.progress_m)
            for key, value in values.items():
                outputs[key].append(value.cpu().float().numpy())
        return {k: np.concatenate(v) for k, v in outputs.items() if v}

    before = evaluate(detour, np.arange(len(groups)), True)
    retained_before = evaluate(ordinary, retain, False)
    losses = []
    for step in range(args.steps):
        selected = rng.choice(train, 2, replace=True)
        old = rng.choice(replay, 2, replace=True)
        first, second = batch_for(detour, selected), batch_for(ordinary, old)
        batch = {k: torch.cat((first[k], second[k])) for k in first}
        condition = unpack_policy_batch(batch).condition
        # Banks alternate goal/no-goal. The improvement measure must match the
        # anchored goal-conditioned flow; no-goal is preserved separately.
        columns = 2*rng.integers(128, size=(2, 8))
        old_columns = 2*rng.integers(16, size=(2, 8))
        paths = np.concatenate((bank['paths'][selected[:, None], columns],
            ordinary['full_raw2_paths'][old[:, None], old_columns]))
        values = controls(paths)
        utility = progress_utility(torch.as_tensor(bank['progress'][selected[:, None], columns], device='cuda'),
            torch.as_tensor(bank['length'][selected[:, None], columns], device='cuda'),
            torch.as_tensor(bank['clearance'][selected[:, None], columns], device='cuda'),
            horizon=3.6, radius=.42, margin=.1)
        # Equal utility on old data supplies only the same velocity preservation term.
        utility = torch.cat((utility, torch.zeros_like(utility)))
        optimizer.zero_grad(set_to_none=True)
        loss = group_flow_loss(policy, reference, condition, values, utility, mix=mix)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss))
        if step % 20 == 0:
            print(json.dumps({'step': step, 'loss': float(loss), 'gradient_norm': float(norm)}), flush=True)
    after = evaluate(detour, np.arange(len(groups)), True)
    retained_after = evaluate(ordinary, retain, False)

    def metrics(values, ids):
        good = (values['clearance'][ids] >= .1) & (values['progress'][ids] >= .2)
        index = values['scores'][ids].argmax(1)
        rows = np.arange(len(ids))
        return dict(cases=len(ids), coverage=int(good.any(1).sum()), chosen=int(good[rows, index].sum()),
                    collision=int((values['clearance'][ids][rows, index] < 0).sum()))

    change = np.linalg.norm(retained_after['paths']-retained_before['paths'], axis=-1).mean((1, 2))
    report = dict(checkpoint_sha256=meta['checkpoint_sha256'], seed=20260928, steps=args.steps,
        train_parameters=sum(p.numel() for p in parameters), trained_names=names,
        learning_rate=1e-5, mix=.05, temperature=.25, preservation_weight=10.,
        improvement_measure='goal-conditioned candidates only; no-goal preservation separately',
        train_before=metrics(before, train), train_after=metrics(after, train),
        holdout_before=metrics(before, holdout), holdout_after=metrics(after, holdout),
        ordinary_replay_states=len(replay), retention_probe_states=len(retain),
        retention_before=metrics(retained_before, np.arange(len(retain))),
        retention_after=metrics(retained_after, np.arange(len(retain))),
        retention_candidate_change_mean_m=float(change.mean()), retention_candidate_change_max_case_mean_m=float(change.max()),
        retention_selected_index_changes=int((retained_before['scores'].argmax(1)!=retained_after['scores'].argmax(1)).sum()),
        first_loss=losses[0], final_loss=losses[-1],
        interpretation='Small offline update of existing memory-related projections only. Oracle-labelled self-generated bank, no new memory channels, no deployment. Retention movement is not a closed-loop safety guarantee.')
    torch.save({'model': policy.state_dict(), 'report': report}, args.output/'research_state.pt')
    np.savez_compressed(args.output/'predictions.npz', **{'before_'+k:v for k,v in before.items()},
                        **{'after_'+k:v for k,v in after.items()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
