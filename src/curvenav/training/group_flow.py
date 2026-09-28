"""Conservative local policy improvement of the existing conditional flow.

Candidates come from the policy. Detached utilities reweight their endpoint
measure; a frozen reference anchors the velocity field. No long-horizon Q,
extra inference network, trajectory repair, top-k gate, or path search.
"""

from dataclasses import replace

import torch

from curvenav.models.policy import (
    FLOW_LOGIT_NORMAL_MEAN, FLOW_LOGIT_NORMAL_STD, repeat_condition,
)


@torch.no_grad()
def sample_learned_group(policy, condition, count=32):
    """Raw goal-conditioned samples from the reference measure [B,K,14]."""
    if count < 2:
        raise ValueError("count must be >= 2")
    encoded = policy.encode_condition(condition)
    memory = policy.trajectory_decoder.project_condition_memory(encoded)
    batch = len(condition.point_goal)
    repeated = repeat_condition(encoded, count)
    source = torch.randn(batch * count, policy.curve_codec.coordinate_dim,
                         device=condition.point_goal.device)
    coordinates = policy._generate_coordinates(repeated, memory, source)
    return policy.curve_codec.values_from_coordinates(coordinates).unflatten(0, (batch, count))


def group_weights(utilities, *, temperature=.25, mix=.05):
    """KL-tilted measure mixed with the original uniform candidate measure.

    w = (1-mix)/K + mix*softmax(U/temperature). Its total variation from
    uniform is at most mix, and its expected supplied utility cannot decrease.
    These are finite-bank statements, not guarantees about a fitted policy.
    """
    if utilities.ndim != 2 or utilities.shape[1] < 2:
        raise ValueError("utilities must be [B,K] with K >= 2")
    if (not torch.isfinite(utilities).all() or not 0 < temperature < float('inf')
            or not 0 <= mix <= 1):
        raise ValueError("utilities and temperature must be finite; temperature > 0 and mix in [0,1]")
    value = utilities.detach().double()
    tilted = ((value - value.amax(1, keepdim=True)) / temperature).softmax(1).float()
    return (1-mix) / utilities.shape[1] + mix * tilted


def group_flow_loss(policy, reference, condition, curve_values, utilities, *,
                    temperature=.25, mix=.05, preservation_weight=10.):
    """Proximal change of the flow objective, anchored to the generating policy.

    The reference must be an eval snapshot, excluded from the optimizer, with
    the same requires_grad mask as the policy. Its outputs are detached. Matching
    grad participation also matches BF16 forward kernels: a no-grad teacher can
    otherwise create a spurious preservation gradient at identical weights.
    Local utility is permitted; do not call it a learned long-horizon Q.
    Improvement candidates must be goal-conditioned samples of that reference;
    no-goal samples have a different base measure and are not silently mixed in.
    Subtracting the original measure's flow loss removes self-reconstruction
    drift: at the reference, the update vanishes continuously as mix -> 0.
    The reference is needed only during training, never during inference.
    """
    if (curve_values.ndim != 3 or curve_values.shape[:2] != utilities.shape
            or curve_values.shape[0] != len(condition.point_goal)):
        raise ValueError("curve_values [B,K,C] and utilities [B,K] must match observations")
    if policy.training or reference.training or any(p.requires_grad != r.requires_grad
                                 for p, r in zip(policy.parameters(), reference.parameters(), strict=True)):
        raise ValueError("both policies must be in eval mode with matching grad masks")
    if any(p.requires_grad for module in (policy.depth_encoder, policy.condition_encoder)
           for p in module.parameters()):
        raise ValueError("this small flow update requires frozen scene encoders")
    if not mix < preservation_weight < float('inf'):
        raise ValueError("preservation_weight must exceed mix for a positive quadratic")
    weights = group_weights(utilities, temperature=temperature, mix=mix)
    batch, count = weights.shape
    clean = policy.curve_codec.coordinates_from_values(curve_values.detach().flatten(0, 1).float())
    source = torch.randn_like(clean)
    time = torch.sigmoid(torch.randn(len(clean), device=clean.device) * FLOW_LOGIT_NORMAL_STD + FLOW_LOGIT_NORMAL_MEAN)
    state = (1-time[:, None])*clean + time[:, None]*source
    encoded = policy.encode_condition(condition)
    repeated = repeat_condition(encoded, count)
    memory = policy.trajectory_decoder.project_condition_memory(encoded)
    velocity = policy._predict_velocity(state, time, repeated, memory)
    no_goal = replace(repeated, goal_present=torch.zeros_like(repeated.goal_present))
    no_goal_velocity = policy._predict_velocity(state, time, no_goal, memory)
    with torch.enable_grad():
        # Both policies use the same frozen scene representation. Reuse it:
        # independent CUDA scatter reductions can otherwise perturb BF16 tokens.
        previous = encoded
        previous_repeated = repeat_condition(previous, count)
        previous_memory = reference.trajectory_decoder.project_condition_memory(previous)
        baseline = reference._predict_velocity(state, time, previous_repeated, previous_memory).detach()
        no_goal_baseline = reference._predict_velocity(state, time,
            replace(previous_repeated, goal_present=torch.zeros_like(previous_repeated.goal_present)),
            previous_memory).detach()
    delta = velocity-baseline
    # ||v-u||² - ||v0-u||², evaluated without subtracting two large losses.
    change = (delta.square()+2*delta*(baseline-(source-clean))).mean(-1).reshape(batch, count)
    preservation = (delta.square()+(no_goal_velocity-no_goal_baseline).square()).mean(-1).reshape(batch, count)
    return (((weights-1/count)*change).sum(1)
            + preservation_weight*preservation.mean(1)).mean()
