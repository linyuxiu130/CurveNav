"""Held-out ranking and safety-opportunity metrics without a scalar utility."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from curvenav.critic.data import POLICY_CANDIDATES
from curvenav.critic.model import CriticPrediction


@dataclass
class CriticMetricCounts:
    all_pair_correct: Tensor
    all_pair_count: Tensor
    policy_pair_correct: Tensor
    policy_pair_count: Tensor
    states: Tensor
    selected_dominated: Tensor
    selected_collision: Tensor
    collision_opportunity_loss: Tensor
    selected_margin_violation: Tensor
    margin_opportunity_loss: Tensor

    def stacked(self) -> Tensor:
        return torch.stack(
            (
                self.all_pair_correct,
                self.all_pair_count,
                self.policy_pair_correct,
                self.policy_pair_count,
                self.states,
                self.selected_dominated,
                self.selected_collision,
                self.collision_opportunity_loss,
                self.selected_margin_violation,
                self.margin_opportunity_loss,
            )
        ).double()

    @classmethod
    def from_stacked(cls, values: Tensor) -> "CriticMetricCounts":
        return cls(*values.unbind())

    def summary(self) -> dict[str, float]:
        def ratio(numerator: Tensor, denominator: Tensor) -> float:
            return float((numerator / denominator.clamp_min(1)).item())

        return {
            "all_pair_accuracy": ratio(self.all_pair_correct, self.all_pair_count),
            "policy_pair_accuracy": ratio(
                self.policy_pair_correct, self.policy_pair_count
            ),
            "selected_dominated_fraction": ratio(
                self.selected_dominated, self.states
            ),
            "selected_collision_fraction": ratio(
                self.selected_collision, self.states
            ),
            "collision_opportunity_loss_fraction": ratio(
                self.collision_opportunity_loss, self.states
            ),
            "selected_margin_violation_fraction": ratio(
                self.selected_margin_violation, self.states
            ),
            "margin_opportunity_loss_fraction": ratio(
                self.margin_opportunity_loss, self.states
            ),
            "states": float(self.states.item()),
        }


def select_critic_indices(prediction: CriticPrediction) -> Tensor:
    """Select by predicted collision safety, margin safety, then Pareto score."""
    policy_score = prediction.score[:, :POLICY_CANDIDATES]
    collision_safe = prediction.collision_logit[:, :POLICY_CANDIDATES] < 0
    margin_safe = prediction.margin_logit[:, :POLICY_CANDIDATES] < 0
    has_collision_safe = collision_safe.any(dim=1, keepdim=True)
    eligible = collision_safe | ~has_collision_safe
    has_margin_safe = (eligible & margin_safe).any(dim=1, keepdim=True)
    eligible &= margin_safe | ~has_margin_safe
    return policy_score.masked_fill(~eligible, -torch.inf).argmax(dim=1)


@torch.no_grad()
def critic_metric_counts(
    prediction: CriticPrediction,
    *,
    preference: Tensor,
    collision: Tensor,
    margin_violation: Tensor,
) -> CriticMetricCounts:
    """Evaluate only the eight deployable policy candidates for selection metrics."""
    score_delta = prediction.score.unsqueeze(2) - prediction.score.unsqueeze(1)
    correct = score_delta > 0
    preference = preference.bool()
    policy_preference = preference[:, :POLICY_CANDIDATES, :POLICY_CANDIDATES]
    policy_correct = correct[:, :POLICY_CANDIDATES, :POLICY_CANDIDATES]
    selected = select_critic_indices(prediction)
    selected_column = policy_preference.gather(
        2,
        selected[:, None, None].expand(-1, POLICY_CANDIDATES, 1),
    ).squeeze(2)
    policy_collision = collision[:, :POLICY_CANDIDATES].bool()
    policy_margin = margin_violation[:, :POLICY_CANDIDATES].bool()
    selected_collision = policy_collision.gather(1, selected[:, None]).squeeze(1)
    selected_margin = policy_margin.gather(1, selected[:, None]).squeeze(1)
    return CriticMetricCounts(
        all_pair_correct=(correct & preference).sum(),
        all_pair_count=preference.sum(),
        policy_pair_correct=(policy_correct & policy_preference).sum(),
        policy_pair_count=policy_preference.sum(),
        states=torch.tensor(
            prediction.score.shape[0], device=prediction.score.device
        ),
        selected_dominated=selected_column.any(dim=1).sum(),
        selected_collision=selected_collision.sum(),
        collision_opportunity_loss=(
            selected_collision & (~policy_collision).any(dim=1)
        ).sum(),
        selected_margin_violation=selected_margin.sum(),
        margin_opportunity_loss=(selected_margin & (~policy_margin).any(dim=1)).sum(),
    )
