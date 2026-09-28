"""Experimental same-state Q-weighted Flow Matching; no path-search teacher.

Adapted from X-NavDP's group weighting (arXiv:2607.28560), not its DDPM loss.
Callers supply detached long-horizon Q estimates learned from actual rollouts.
The production route utility critic is NOT such a Q function.
"""

from dataclasses import replace

import torch

from curvenav.models.policy import (
    FLOW_LOGIT_NORMAL_MEAN, FLOW_LOGIT_NORMAL_STD, repeat_condition,
)


@torch.no_grad()
def sample_learned_group(policy, condition, count=32):
    """Return physical curve values [B,K,C] from raw goal/no-goal neural flows.

    No straight templates, axis flips, graph search, or geometric curve repair.
    Goal-agnostic proposals are subsequently evaluated against the mission goal.
    """
    if count < 2 or count % 2:
        raise ValueError("count must be a positive even number >= 2")
    encoded = policy.encode_condition(condition)
    memory = policy.trajectory_decoder.project_condition_memory(encoded)
    batch = len(condition.point_goal)
    repeated = repeat_condition(encoded, count)
    repeated = replace(
        repeated,
        goal_present=torch.tensor([1., 0.], device=condition.point_goal.device)
        .repeat(batch * count // 2)[:, None],
    )
    source = torch.randn(
        batch * count, policy.curve_codec.coordinate_dim,
        device=condition.point_goal.device,
    )
    coordinates = policy._generate_coordinates(repeated, memory, source)
    return policy.curve_codec.values_from_coordinates(coordinates).unflatten(0, (batch, count))


def group_weights(q_values, *, top_k=4, temperature=1.0):
    """Normalize within each state; tied groups contribute zero actor gradient."""
    if q_values.ndim != 2 or not 1 <= top_k <= q_values.shape[1]:
        raise ValueError("Q must be [B,K], with 1 <= top_k <= K")
    if not torch.isfinite(q_values).all() or not 0 < temperature < float('inf'):
        raise ValueError("Q must be finite and temperature finite and positive")
    q = q_values.detach().float()
    centered = q - q.mean(dim=1, keepdim=True)
    advantage = (centered / q.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)).clamp(-3, 3)
    selected = torch.zeros_like(q, dtype=torch.bool)
    selected.scatter_(1, q.topk(top_k, dim=1).indices, True)
    selected &= advantage > 0
    # Subtract the group maximum for stable exponentiation at low temperatures.
    shifted = (advantage - advantage.amax(dim=1, keepdim=True)).double()
    weights = torch.exp(shifted / temperature).float() * selected
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def group_flow_loss(policy, condition, curve_values, q_values, *, top_k=4, temperature=1.0):
    """One actor update from its own candidate bank and externally learned Q.

    Per-state normalized weights are an explicit experimental choice. This does
    not reproduce X-NavDP's online double-Q learner or implement a replay buffer.
    Do not populate Q with straight-line progress and expect dead-end recovery.
    """
    if (curve_values.ndim != 3 or curve_values.shape[:2] != q_values.shape
            or curve_values.shape[0] != len(condition.point_goal)):
        raise ValueError("curve_values [B,K,C] and Q [B,K] must match the observation batch")
    weights = group_weights(q_values, top_k=top_k, temperature=temperature)
    batch, count = weights.shape
    clean = policy.curve_codec.coordinates_from_values(curve_values.detach().flatten(0, 1).float())
    source = torch.randn_like(clean)
    time = torch.sigmoid(torch.randn(len(clean), device=clean.device) * FLOW_LOGIT_NORMAL_STD + FLOW_LOGIT_NORMAL_MEAN)
    encoded = policy.encode_condition(condition)
    memory = policy.trajectory_decoder.project_condition_memory(encoded)
    state = (1 - time[:, None]) * clean + time[:, None] * source
    # Improve the goal-conditioned actor even when the useful proposal was no-goal.
    velocity = policy._predict_velocity(state, time, repeat_condition(encoded, count), memory)
    error = (velocity - (source - clean)).square().mean(dim=-1).reshape(batch, count)
    active_groups = (weights.sum(dim=1) > 0).sum().clamp_min(1)
    return (error * weights).sum() / active_groups
