"""Geometry-defined strata for held-out local-navigation evaluation."""

from __future__ import annotations

import torch
from torch import Tensor


def evaluation_strata(metrics: dict[str, Tensor]) -> dict[str, Tensor]:
    """Partition samples without inventing synthetic goals or score thresholds.

    Every sample already lies on a held-out, collision-free expert route whose
    mission endpoint is the PointGoal.  These masks only separate the physical
    situations present in that fixed validation set.
    """
    forward = ~metrics["point_goal_is_behind"]
    reference_safe = ~metrics["reference_observed_safety_margin_violation"]
    straight_blocked = metrics["straight_observed_safety_margin_violation"]
    return {
        "all": torch.ones_like(forward),
        "forward": forward,
        "forward_direct": forward & reference_safe & ~straight_blocked,
        "forward_visible_detour": (
            forward & reference_safe & straight_blocked
        ),
        "rear_goal": ~forward,
        "expert_moves_away_from_goal": (
            metrics["reference_goal_progress_m"] < 0
        ),
    }


def summarize_strata(metrics: dict[str, Tensor]) -> dict[str, dict[str, float | int]]:
    """Report fidelity, progress, and observed continuous-path safety per stratum."""
    result: dict[str, dict[str, float | int]] = {}
    for name, selected in evaluation_strata(metrics).items():
        count = int(selected.sum().item())
        values: dict[str, float | int] = {"samples": count}
        if count:
            values.update(
                fixed_horizon_ade_m=metrics["fixed_horizon_ade_m"][selected]
                .mean()
                .item(),
                fixed_horizon_fde_m=metrics["fixed_horizon_fde_m"][selected]
                .mean()
                .item(),
                horizon_coverage_fraction=metrics["horizon_coverage_fraction"][
                    selected
                ].mean().item(),
                goal_progress_regret_m=metrics["goal_progress_regret_m"][selected]
                .mean()
                .item(),
                footprint_collision_fraction=(
                    metrics["observed_footprint_collision"][selected]
                    .float()
                    .mean()
                    .item()
                ),
                safety_margin_violation_fraction=(
                    metrics["observed_safety_margin_violation"][selected]
                    .float()
                    .mean()
                    .item()
                ),
                extra_footprint_collision_fraction=(
                    metrics["observed_footprint_collision"][selected]
                    & ~metrics["reference_observed_footprint_collision"][selected]
                )
                .float()
                .mean()
                .item(),
            )
        result[name] = values
    return result
