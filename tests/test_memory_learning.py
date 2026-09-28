from dataclasses import replace

import pytest
import torch

from curvenav.evaluation.memory_ablation import compare_memory
from curvenav.factory import build_policy
from curvenav.training.group_flow import group_flow_loss, group_weights, sample_learned_group
from test_policy import make_condition, tiny_config


def test_group_weights_and_actor_update_preserve_same_state_credit():
    q = torch.tensor([[-8., -7., -6., -5.], [1., 1., 1., 1.]], requires_grad=True)
    weights = group_weights(q, top_k=2)
    torch.testing.assert_close(weights, group_weights(q + torch.tensor([[100.], [-50.]]), top_k=2))
    assert not weights.requires_grad and weights[0, 3] > weights[0, 2] > 0
    assert weights[0, :2].sum() == 0 and weights[1].sum() == 0
    assert torch.isfinite(group_weights(q, top_k=2, temperature=1e-300)).all()
    with pytest.raises(ValueError, match='finite'):
        group_weights(torch.full((2, 4), float('nan')))

    torch.manual_seed(2026)
    policy = build_policy(tiny_config()).eval()
    condition = make_condition(2)
    # Sampling must not call the production hand-perturbed proposal constructor.
    policy._propose = lambda *args: pytest.fail('handcrafted proposals used')
    values = sample_learned_group(policy, condition, count=4)
    assert values.shape == (2, 4, policy.curve_codec.coordinate_dim)
    assert not values.requires_grad
    loss = group_flow_loss(policy, condition, values, q, top_k=2)
    loss.backward()
    gradient = sum(p.grad.abs().sum() for p in policy.trajectory_decoder.parameters() if p.grad is not None)
    assert torch.isfinite(loss) and gradient > 0 and q.grad is None
    assert all(p.grad is None for p in policy.trajectory_evaluator.parameters())
    policy.zero_grad(set_to_none=True)
    zero = group_flow_loss(policy, condition, values, torch.ones_like(q), top_k=2)
    zero.backward()
    assert zero == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in policy.parameters())


@torch.no_grad()
def test_memory_counterfactual_preserves_request_and_fixed_candidate_bank():
    torch.manual_seed(2026)
    policy = build_policy(tiny_config()).eval()
    condition = make_condition(1)
    full = torch.zeros_like(condition.obstacle_memory)
    full[:, 26:38, 35:45] = True
    condition = replace(condition, obstacle_memory=full)
    original = full.clone()
    current = torch.zeros_like(full)
    report, arrays = compare_memory(policy, condition, current)
    torch.testing.assert_close(condition.obstacle_memory, original)
    assert report['full']['candidate_change_mean_m'] == [0.]
    assert report['full']['fixed_bank_score_change_mae'] == [0.]
    assert report['current_obstacles']['candidate_change_mean_m'][0] > 0
    assert report['current_obstacles']['fixed_bank_score_change_mae'][0] > 0
    assert report['current_obstacles'] == report['empty_obstacle_memory']
    encoded = policy.encode_condition(replace(condition, obstacle_memory=current))
    expected = policy.score_candidate_paths(encoded, condition.point_goal, torch.from_numpy(arrays['full_paths']))
    torch.testing.assert_close(expected.detach(), torch.from_numpy(arrays['current_obstacles_fixed_bank_scores']))
    full_encoded = policy.encode_condition(condition)
    for name, isolated in (
        ('empty_memory_tokens_only', replace(full_encoded, tokens=encoded.tokens)),
        ('empty_path_geometry_only', replace(full_encoded, configuration_field=encoded.configuration_field)),
    ):
        expected = policy.score_candidate_paths(isolated, condition.point_goal, torch.from_numpy(arrays['full_paths']))
        torch.testing.assert_close(expected, torch.from_numpy(arrays[f'{name}_fixed_bank_scores']))
        assert report[name]['fixed_bank_score_change_mae'][0] > 0
