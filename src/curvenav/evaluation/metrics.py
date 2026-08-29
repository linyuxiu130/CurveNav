"""Model-independent metric geometry for local trajectory evaluation."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from curvenav.models.safety import SAFETY_CLEARANCE_M, sample_configuration_field
from curvenav.trajectory import path_arc_length


COMPARISON_HORIZON_M = 2.0
EVALUATION_SPACING_M = 0.025
COMPARISON_PATH_SAMPLES = round(COMPARISON_HORIZON_M / EVALUATION_SPACING_M) + 1


def _validate_path(path: Tensor, name: str) -> None:
    if path.ndim != 3 or path.shape[-1] != 2 or path.shape[1] < 3:
        raise ValueError(f"{name} must have shape [B,P,2] with P >= 3")
    if not torch.isfinite(path).all():
        raise ValueError(f"{name} must contain only finite coordinates")


def resample_path_at_distance(path: Tensor, distance_m: Tensor) -> Tensor:
    """Linearly sample a polyline at per-example physical arc distances.

    Requests beyond the path endpoint hold the endpoint.  This is deliberate:
    a short prediction must incur displacement error on the missing horizon
    instead of receiving an artificially favorable overlap-only score.
    """
    _validate_path(path, "path")
    if distance_m.ndim != 2 or distance_m.shape[0] != path.shape[0]:
        raise ValueError("distance_m must have shape [B,Q]")
    if (distance_m < 0).any() or not torch.isfinite(distance_m).all():
        raise ValueError("distance_m must contain finite non-negative values")

    segment = torch.linalg.vector_norm(path[:, 1:] - path[:, :-1], dim=-1)
    cumulative = torch.cat(
        (torch.zeros_like(segment[:, :1]), segment.cumsum(dim=-1)), dim=-1
    )
    query = torch.minimum(distance_m, cumulative[:, -1:]).contiguous()
    lower = torch.searchsorted(cumulative.contiguous(), query, right=True) - 1
    lower = lower.clamp(0, path.shape[1] - 2)
    upper = lower + 1
    lower_distance = cumulative.gather(1, lower)
    upper_distance = cumulative.gather(1, upper)
    weight = (query - lower_distance) / (
        upper_distance - lower_distance
    ).clamp_min(1e-8)
    gather_index = lower[..., None].expand(-1, -1, 2)
    start = path.gather(1, gather_index)
    end = path.gather(1, (upper[..., None].expand_as(gather_index)))
    return start + weight[..., None] * (end - start)


def _uniform_distance(length_m: Tensor, samples: int) -> Tensor:
    progress = torch.linspace(
        0.0,
        1.0,
        samples,
        device=length_m.device,
        dtype=length_m.dtype,
    )
    return length_m[:, None] * progress[None]


def _path_shape_metrics(path: Tensor, horizon_m: float) -> dict[str, Tensor]:
    length = path_arc_length(path)
    evaluated_length = length.clamp_max(horizon_m)
    dense = resample_path_at_distance(
        path,
        _uniform_distance(evaluated_length, COMPARISON_PATH_SAMPLES),
    )
    delta = dense[:, 1:] - dense[:, :-1]
    segment_length = torch.linalg.vector_norm(delta, dim=-1)
    heading = torch.atan2(delta[..., 1], delta[..., 0])
    turn = torch.atan2(
        (heading[:, 1:] - heading[:, :-1]).sin(),
        (heading[:, 1:] - heading[:, :-1]).cos(),
    )
    support = 0.5 * (segment_length[:, 1:] + segment_length[:, :-1])
    curvature = turn / support.clamp_min(1e-6)
    tangent_dot = (delta[:, 1:] * delta[:, :-1]).sum(dim=-1)
    return {
        "max_abs_curvature_inv_m": curvature.abs().amax(dim=-1),
        "abs_curvature_inv_m_p95": torch.quantile(
            curvature.abs(), 0.95, dim=-1
        ),
        "rms_curvature_inv_m": curvature.square().mean(dim=-1).sqrt(),
        "curvature_variation_inv_m2": (
            (curvature[:, 1:] - curvature[:, :-1]).abs().sum(dim=-1)
            / evaluated_length.clamp_min(EVALUATION_SPACING_M)
        ),
        "total_abs_heading_change_rad": turn.abs().sum(dim=-1),
        "has_tangent_reversal": (tangent_dot < 0).any(dim=-1),
    }


def trajectory_metrics(
    path: Tensor,
    reference_path: Tensor,
    point_goal: Tensor,
    comparison_horizon_m: float = COMPARISON_HORIZON_M,
) -> dict[str, Tensor]:
    """Compare trajectories on one physical horizon, independent of point count."""
    _validate_path(path, "path")
    _validate_path(reference_path, "reference_path")
    if path.shape[0] != reference_path.shape[0]:
        raise ValueError("path and reference_path batch dimensions must match")
    if point_goal.shape != (path.shape[0], 2):
        raise ValueError("point_goal must have shape [B,2]")
    if comparison_horizon_m <= 0:
        raise ValueError("comparison_horizon_m must be positive")

    predicted_length = path_arc_length(path)
    reference_length = path_arc_length(reference_path)
    evaluation_length = reference_length.clamp_max(comparison_horizon_m)
    if (evaluation_length <= 1e-6).any():
        raise ValueError("reference trajectories must have positive length")
    query = _uniform_distance(evaluation_length, COMPARISON_PATH_SAMPLES)
    predicted = resample_path_at_distance(path, query)
    reference = resample_path_at_distance(reference_path, query)
    displacement = torch.linalg.vector_norm(predicted - reference, dim=-1)
    goal_distance = torch.linalg.vector_norm(point_goal, dim=-1)
    predicted_progress = goal_distance - torch.linalg.vector_norm(
        point_goal - predicted[:, -1], dim=-1
    )
    reference_progress = goal_distance - torch.linalg.vector_norm(
        point_goal - reference[:, -1], dim=-1
    )
    predicted_shape = _path_shape_metrics(path, comparison_horizon_m)
    reference_shape = _path_shape_metrics(reference_path, comparison_horizon_m)
    result = {
        "fixed_horizon_ade_m": displacement.mean(dim=-1),
        "fixed_horizon_fde_m": displacement[:, -1],
        "comparison_horizon_m": evaluation_length,
        "horizon_coverage_fraction": (
            predicted_length / evaluation_length
        ).clamp(max=1.0),
        "arc_length_m": predicted_length,
        "reference_arc_length_m": reference_length,
        "arc_length_error_m": (predicted_length - reference_length).abs(),
        "goal_progress_m": predicted_progress,
        "reference_goal_progress_m": reference_progress,
        "goal_progress_regret_m": reference_progress - predicted_progress,
        "point_goal_distance_m": goal_distance,
        "point_goal_bearing_rad": torch.atan2(point_goal[:, 1], point_goal[:, 0]),
        "point_goal_is_behind": point_goal[:, 0] < 0,
    }
    result.update(predicted_shape)
    result.update(
        {f"reference_{name}": value for name, value in reference_shape.items()}
    )
    return result


def observed_safety_metrics(
    path: Tensor,
    configuration_field: Tensor,
    planning_horizon_m: float,
) -> dict[str, Tensor]:
    """Evaluate perceived safety at <=2.5 cm spacing over the local horizon."""
    _validate_path(path, "path")
    if planning_horizon_m <= 0:
        raise ValueError("planning_horizon_m must be positive")
    length = path_arc_length(path)
    evaluated_length = length.clamp_max(planning_horizon_m)
    samples = math.ceil(planning_horizon_m / EVALUATION_SPACING_M) + 1
    dense_path = resample_path_at_distance(
        path,
        _uniform_distance(evaluated_length, samples),
    )
    sampled = sample_configuration_field(
        configuration_field,
        dense_path,
        planning_horizon_m,
    )
    observed = sampled[..., 3] > 0.5
    clearance = sampled[..., 0]
    observed_clearance = clearance.masked_fill(~observed, torch.inf)
    minimum = observed_clearance.amin(dim=-1)
    collision = observed & (clearance < 0.0)
    margin_violation = observed & (clearance < SAFETY_CLEARANCE_M)
    maximum_margin_violation = torch.where(
        observed,
        torch.relu(SAFETY_CLEARANCE_M - clearance),
        torch.zeros_like(clearance),
    ).amax(dim=-1)
    return {
        "observed_path_fraction": observed.float().mean(dim=-1),
        "observed_min_clearance_m": minimum,
        "observed_footprint_collision": collision.any(dim=-1),
        "observed_safety_margin_violation": margin_violation.any(dim=-1),
        "observed_max_margin_violation_m": maximum_margin_violation,
        "arc_length_beyond_local_horizon_m": torch.relu(
            length - planning_horizon_m
        ),
    }


def summarize_metrics(metrics: dict[str, Tensor]) -> dict[str, float | int]:
    """Summarize accuracy, progress, coverage, and executable path geometry."""
    reference_turn = metrics["reference_total_abs_heading_change_rad"]
    high_turn_threshold = torch.quantile(reference_turn, 0.9)
    high_turn = reference_turn >= high_turn_threshold
    result: dict[str, float | int] = {
        "fixed_horizon_ade_m_mean": metrics["fixed_horizon_ade_m"].mean().item(),
        "fixed_horizon_ade_m_p90": torch.quantile(
            metrics["fixed_horizon_ade_m"], 0.9
        ).item(),
        "fixed_horizon_fde_m_mean": metrics["fixed_horizon_fde_m"].mean().item(),
        "fixed_horizon_fde_m_p90": torch.quantile(
            metrics["fixed_horizon_fde_m"], 0.9
        ).item(),
        "horizon_coverage_fraction_mean": metrics[
            "horizon_coverage_fraction"
        ].mean().item(),
        "complete_horizon_fraction": (
            metrics["horizon_coverage_fraction"] >= 1.0 - 1e-5
        ).float().mean().item(),
        "goal_progress_m_mean": metrics["goal_progress_m"].mean().item(),
        "reference_goal_progress_m_mean": metrics[
            "reference_goal_progress_m"
        ].mean().item(),
        "goal_progress_regret_m_mean": metrics[
            "goal_progress_regret_m"
        ].mean().item(),
        "negative_progress_fraction": (
            metrics["goal_progress_m"] < 0
        ).float().mean().item(),
        "arc_length_m_mean": metrics["arc_length_m"].mean().item(),
        "reference_arc_length_m_mean": metrics[
            "reference_arc_length_m"
        ].mean().item(),
        "arc_length_error_m_mean": metrics["arc_length_error_m"].mean().item(),
        "max_abs_curvature_inv_m_p95": torch.quantile(
            metrics["max_abs_curvature_inv_m"], 0.95
        ).item(),
        "reference_max_abs_curvature_inv_m_p95": torch.quantile(
            metrics["reference_max_abs_curvature_inv_m"], 0.95
        ).item(),
        "abs_curvature_inv_m_p95_median": torch.quantile(
            metrics["abs_curvature_inv_m_p95"], 0.5
        ).item(),
        "reference_abs_curvature_inv_m_p95_median": torch.quantile(
            metrics["reference_abs_curvature_inv_m_p95"], 0.5
        ).item(),
        "rms_curvature_inv_m_mean": metrics["rms_curvature_inv_m"].mean().item(),
        "curvature_variation_inv_m2_mean": metrics[
            "curvature_variation_inv_m2"
        ].mean().item(),
        "reference_curvature_variation_inv_m2_mean": metrics[
            "reference_curvature_variation_inv_m2"
        ].mean().item(),
        "total_abs_heading_change_rad_mean": metrics[
            "total_abs_heading_change_rad"
        ].mean().item(),
        "tangent_reversal_fraction": metrics["has_tangent_reversal"]
        .float()
        .mean()
        .item(),
        "high_turn_threshold_rad": high_turn_threshold.item(),
        "high_turn_samples": int(high_turn.sum().item()),
        "fixed_horizon_ade_m_high_turn_10pct": metrics[
            "fixed_horizon_ade_m"
        ][high_turn].mean().item(),
        "fixed_horizon_fde_m_high_turn_10pct": metrics[
            "fixed_horizon_fde_m"
        ][high_turn].mean().item(),
    }
    valid_frames = metrics.get("valid_observation_frames")
    if valid_frames is not None:
        for frame_count in valid_frames.unique(sorted=True).tolist():
            selected = valid_frames == frame_count
            result[f"samples_with_{frame_count}_frames"] = int(selected.sum().item())
            result[f"fixed_horizon_ade_m_with_{frame_count}_frames"] = metrics[
                "fixed_horizon_ade_m"
            ][selected].mean().item()
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise FloatingPointError(f"offline trajectory metrics are non-finite: {result}")
    return result


def summarize_paired_safety(metrics: dict[str, Tensor]) -> dict[str, float]:
    """Summarize perceived risk relative to the same-field expert baseline."""
    predicted_finite = torch.isfinite(metrics["observed_min_clearance_m"])
    reference_finite = torch.isfinite(
        metrics["reference_observed_min_clearance_m"]
    )
    if not predicted_finite.any() or not reference_finite.any():
        raise RuntimeError("evaluated trajectories do not intersect observed geometry")
    return {
        "observed_path_fraction_mean": metrics[
            "observed_path_fraction"
        ].mean().item(),
        "observed_min_clearance_m_mean": metrics["observed_min_clearance_m"][
            predicted_finite
        ].mean().item(),
        "observed_footprint_collision_fraction": metrics[
            "observed_footprint_collision"
        ].float().mean().item(),
        "observed_safety_margin_violation_fraction": metrics[
            "observed_safety_margin_violation"
        ].float().mean().item(),
        "observed_max_margin_violation_m_mean": metrics[
            "observed_max_margin_violation_m"
        ].mean().item(),
        "arc_length_beyond_local_horizon_m_mean": metrics[
            "arc_length_beyond_local_horizon_m"
        ].mean().item(),
        "extra_footprint_collision_fraction_over_reference": (
            metrics["observed_footprint_collision"]
            & ~metrics["reference_observed_footprint_collision"]
        ).float().mean().item(),
        "extra_safety_margin_violation_fraction_over_reference": (
            metrics["observed_safety_margin_violation"]
            & ~metrics["reference_observed_safety_margin_violation"]
        ).float().mean().item(),
        "reference_observed_path_fraction_mean": metrics[
            "reference_observed_path_fraction"
        ].mean().item(),
        "reference_observed_min_clearance_m_mean": metrics[
            "reference_observed_min_clearance_m"
        ][reference_finite].mean().item(),
        "reference_observed_footprint_collision_fraction": metrics[
            "reference_observed_footprint_collision"
        ].float().mean().item(),
        "reference_observed_safety_margin_violation_fraction": metrics[
            "reference_observed_safety_margin_violation"
        ].float().mean().item(),
    }
