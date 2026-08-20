"""State-balanced pairwise and primitive-label losses for the critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from curvenav.critic.model import CriticPrediction


@dataclass
class CriticLoss:
    loss: Tensor
    pairwise: Tensor
    collision: Tensor
    margin: Tensor
    progress: Tensor
    clearance: Tensor


def _state_mean(values: Tensor, valid: Tensor) -> Tensor:
    counts = valid.sum(dim=1)
    states = counts > 0
    if not torch.any(states):
        return values.sum() * 0.0
    reduced = (values * valid).sum(dim=1) / counts.clamp_min(1)
    return reduced[states].mean()


def critic_loss(
    prediction: CriticPrediction,
    *,
    preference: Tensor,
    candidate_valid: Tensor,
    collision: Tensor,
    margin_violation: Tensor,
    progress_m: Tensor,
    progress_valid: Tensor,
    clearance_m: Tensor,
    auxiliary_weight: float,
    progress_scale_m: float = 5.6,
) -> CriticLoss:
    """Compute one equal-weight contribution per state, independent of pair count."""
    pair_valid = preference.bool()
    score_delta = prediction.score.unsqueeze(2) - prediction.score.unsqueeze(1)
    pairwise = _state_mean(F.softplus(-score_delta).flatten(1), pair_valid.flatten(1))
    candidate_valid = candidate_valid.bool()
    collision_loss = _state_mean(
        F.binary_cross_entropy_with_logits(
            prediction.collision_logit, collision.float(), reduction="none"
        ),
        candidate_valid,
    )
    margin_loss = _state_mean(
        F.binary_cross_entropy_with_logits(
            prediction.margin_logit, margin_violation.float(), reduction="none"
        ),
        candidate_valid,
    )
    progress_loss = _state_mean(
        F.smooth_l1_loss(
            prediction.progress,
            progress_m / progress_scale_m,
            reduction="none",
        ),
        candidate_valid & progress_valid.bool(),
    )
    clearance_loss = _state_mean(
        F.smooth_l1_loss(prediction.clearance, clearance_m, reduction="none"),
        candidate_valid,
    )
    auxiliary = torch.stack(
        (collision_loss, margin_loss, progress_loss, clearance_loss)
    ).mean()
    total = pairwise + auxiliary_weight * auxiliary
    return CriticLoss(
        loss=total,
        pairwise=pairwise,
        collision=collision_loss,
        margin=margin_loss,
        progress=progress_loss,
        clearance=clearance_loss,
    )
