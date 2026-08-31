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
    reference_safe = ~metrics["reference_safety_margin_violation"]
    straight_blocked = metrics["straight_safety_margin_violation"]
    return {
        "all": torch.ones_like(forward),
        "forward": forward,
        "forward_direct": forward & reference_safe & ~straight_blocked,
        "forward_detour": (
            forward & reference_safe & straight_blocked
        ),
        "rear_goal": ~forward,
        "expert_moves_away_from_goal": (
            metrics["reference_goal_progress_m"] < 0
        ),
    }


def summarize_strata(metrics: dict[str, Tensor]) -> dict[str, dict[str, float | int]]:
    """Report fidelity, progress, and complete configuration-space safety."""
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
                    metrics["footprint_collision"][selected]
                    .float()
                    .mean()
                    .item()
                ),
                safety_margin_violation_fraction=(
                    metrics["safety_margin_violation"][selected]
                    .float()
                    .mean()
                    .item()
                ),
                extra_footprint_collision_fraction=(
                    metrics["footprint_collision"][selected]
                    & ~metrics["reference_footprint_collision"][selected]
                )
                .float()
                .mean()
                .item(),
            )
        result[name] = values
    return result


def summarize_goal_bearing_strata(
    metrics: dict[str, Tensor],
) -> dict[str, dict[str, float | int]]:
    """Measure goal response on fixed angular regions of the real validation set."""
    absolute_bearing = metrics["point_goal_bearing_rad"].abs()
    boundaries = (0.0, torch.pi / 6, torch.pi / 3, torch.pi / 2, torch.pi)
    names = ("0_30_deg", "30_60_deg", "60_90_deg", "90_180_deg")
    result: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(names):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (absolute_bearing >= lower) & (absolute_bearing < upper)
        if index == len(names) - 1:
            selected = (absolute_bearing >= lower) & (absolute_bearing <= upper)
        count = int(selected.sum().item())
        values: dict[str, float | int] = {"samples": count}
        if count:
            values.update(
                mean_abs_goal_bearing_rad=absolute_bearing[selected].mean().item(),
                endpoint_goal_bearing_error_rad=metrics[
                    "path_endpoint_goal_bearing_error_rad"
                ][selected].mean().item(),
                reference_endpoint_goal_bearing_error_rad=metrics[
                    "reference_path_endpoint_goal_bearing_error_rad"
                ][selected].mean().item(),
                endpoint_goal_bearing_regret_rad=metrics[
                    "endpoint_goal_bearing_regret_rad"
                ][selected].mean().item(),
                footprint_collision_fraction=metrics["footprint_collision"][
                    selected
                ].float().mean().item(),
            )
        result[name] = values
    return result
