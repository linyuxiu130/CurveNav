"""Model-independent metric geometry for local trajectory evaluation."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from curvenav.models.safety import SAFETY_CLEARANCE_M, sample_configuration_field
from curvenav.trajectory import (
    path_arc_length,
    resample_path_at_distance,
    resample_path_to_horizon,
)
from curvenav.data.privileged import SourcePathQuery
from curvenav.physical import PATH_CONFIGURATION_QUERY_SPACING_M


COMPARISON_HORIZON_M = 2.0
EVALUATION_SPACING_M = PATH_CONFIGURATION_QUERY_SPACING_M
COMPARISON_PATH_SAMPLES = round(COMPARISON_HORIZON_M / EVALUATION_SPACING_M) + 1


def _validate_path(path: Tensor, name: str) -> None:
    if path.ndim != 3 or path.shape[-1] != 2 or path.shape[1] < 3:
        raise ValueError(f"{name} must have shape [B,P,2] with P >= 3")
    if not torch.isfinite(path).all():
        raise ValueError(f"{name} must contain only finite coordinates")


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
    goal_bearing = torch.atan2(point_goal[:, 1], point_goal[:, 0])
    endpoint_bearing = torch.atan2(path[:, -1, 1], path[:, -1, 0])
    reference_endpoint_bearing = torch.atan2(
        reference_path[:, -1, 1], reference_path[:, -1, 0]
    )
    endpoint_goal_error = torch.atan2(
        (endpoint_bearing - goal_bearing).sin(),
        (endpoint_bearing - goal_bearing).cos(),
    ).abs()
    reference_endpoint_goal_error = torch.atan2(
        (reference_endpoint_bearing - goal_bearing).sin(),
        (reference_endpoint_bearing - goal_bearing).cos(),
    ).abs()
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
        "point_goal_bearing_rad": goal_bearing,
        "point_goal_is_behind": point_goal[:, 0] < 0,
        "path_endpoint_bearing_rad": endpoint_bearing,
        "reference_path_endpoint_bearing_rad": reference_endpoint_bearing,
        "path_endpoint_goal_bearing_error_rad": endpoint_goal_error,
        "reference_path_endpoint_goal_bearing_error_rad": (
            reference_endpoint_goal_error
        ),
        "endpoint_goal_bearing_regret_rad": (
            endpoint_goal_error - reference_endpoint_goal_error
        ),
    }
    result.update(predicted_shape)
    result.update(
        {f"reference_{name}": value for name, value in reference_shape.items()}
    )
    return result


def configuration_space_safety_metrics(
    path: Tensor,
    configuration_field: Tensor,
    planning_horizon_m: float,
) -> dict[str, Tensor]:
    """Evaluate path safety in a signed robot configuration-space field."""
    _validate_path(path, "path")
    if planning_horizon_m <= 0:
        raise ValueError("planning_horizon_m must be positive")
    if configuration_field.ndim != 4 or configuration_field.shape[1] != 5:
        raise ValueError(
            "evaluation configuration field must contain clearance, gradient, "
            "coverage, and occupancy channels"
        )
    length = path_arc_length(path)
    dense_path, active = resample_path_to_horizon(
        path,
        planning_horizon_m,
        EVALUATION_SPACING_M,
    )
    sampled = sample_configuration_field(
        configuration_field,
        dense_path,
        planning_horizon_m,
    )
    covered = active & (sampled[..., 3] > 0.5)
    clearance = sampled[..., 0]
    covered_clearance = clearance.masked_fill(~covered, torch.inf)
    minimum = covered_clearance.amin(dim=-1)
    collision = covered & (clearance < 0.0)
    margin_violation = covered & (clearance < SAFETY_CLEARANCE_M)
    maximum_margin_violation = torch.where(
        covered,
        torch.relu(SAFETY_CLEARANCE_M - clearance),
        torch.zeros_like(clearance),
    ).amax(dim=-1)
    return {
        "path_field_coverage_fraction": covered.sum(dim=-1) / active.sum(
            dim=-1
        ).clamp_min(1),
        "min_clearance_m": minimum,
        "footprint_collision": collision.any(dim=-1),
        "safety_margin_violation": margin_violation.any(dim=-1),
        "max_margin_violation_m": maximum_margin_violation,
        "arc_length_beyond_local_horizon_m": torch.relu(
            length - planning_horizon_m
        ),
    }


def configuration_space_collision_attribution(
    source_query: SourcePathQuery,
    raw_depth_field: Tensor,
    planning_horizon_m: float,
) -> dict[str, Tensor]:
    """Attribute source-truth collisions at identical dense local points."""
    dense_path = source_query.local_path
    raw = sample_configuration_field(
        raw_depth_field, dense_path, planning_horizon_m
    )
    truth_collision = source_query.active & (source_query.clearance_m < 0.0)
    truth_free = source_query.active & ~truth_collision
    raw_ray_coverage = source_query.active & (raw[..., 3] > 0.5)
    raw_collision = raw_ray_coverage & (raw[..., 0] < 0.0)

    def point_count(mask: Tensor) -> Tensor:
        return mask.sum(dim=-1)

    return {
        "truth_collision_point_count": point_count(truth_collision),
        "truth_collision_point_raw_depth_count": point_count(
            truth_collision & raw_collision
        ),
        "truth_collision_point_raw_ray_coverage_count": point_count(
            truth_collision & raw_ray_coverage
        ),
        "truth_collision_point_raw_ray_coverage_missed_count": point_count(
            truth_collision & raw_ray_coverage & ~raw_collision
        ),
        "truth_collision_point_outside_raw_ray_coverage_count": point_count(
            truth_collision & ~raw_ray_coverage
        ),
        "raw_depth_false_collision_point_count": point_count(
            truth_free & raw_collision
        ),
        "truth_collision_trajectory": truth_collision.any(dim=-1),
        "truth_collision_trajectory_recognized_by_raw_depth": (
            truth_collision & raw_collision
        ).any(dim=-1),
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
        "path_endpoint_goal_bearing_error_rad_mean": metrics[
            "path_endpoint_goal_bearing_error_rad"
        ].mean().item(),
        "reference_path_endpoint_goal_bearing_error_rad_mean": metrics[
            "reference_path_endpoint_goal_bearing_error_rad"
        ].mean().item(),
        "endpoint_goal_bearing_regret_rad_mean": metrics[
            "endpoint_goal_bearing_regret_rad"
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


def summarize_configuration_safety(metrics: dict[str, Tensor]) -> dict[str, float]:
    """Summarize complete configuration-space risk against the expert baseline."""
    predicted_finite = torch.isfinite(metrics["min_clearance_m"])
    reference_finite = torch.isfinite(
        metrics["reference_min_clearance_m"]
    )
    if not predicted_finite.any() or not reference_finite.any():
        raise RuntimeError("evaluated trajectories do not intersect observed geometry")
    return {
        "path_field_coverage_fraction_mean": metrics[
            "path_field_coverage_fraction"
        ].mean().item(),
        "min_clearance_m_mean": metrics["min_clearance_m"][
            predicted_finite
        ].mean().item(),
        "footprint_collision_fraction": metrics[
            "footprint_collision"
        ].float().mean().item(),
        "safety_margin_violation_fraction": metrics[
            "safety_margin_violation"
        ].float().mean().item(),
        "max_margin_violation_m_mean": metrics[
            "max_margin_violation_m"
        ].mean().item(),
        "arc_length_beyond_local_horizon_m_mean": metrics[
            "arc_length_beyond_local_horizon_m"
        ].mean().item(),
        "extra_footprint_collision_fraction_over_reference": (
            metrics["footprint_collision"]
            & ~metrics["reference_footprint_collision"]
        ).float().mean().item(),
        "extra_safety_margin_violation_fraction_over_reference": (
            metrics["safety_margin_violation"]
            & ~metrics["reference_safety_margin_violation"]
        ).float().mean().item(),
        "reference_path_field_coverage_fraction_mean": metrics[
            "reference_path_field_coverage_fraction"
        ].mean().item(),
        "reference_min_clearance_m_mean": metrics[
            "reference_min_clearance_m"
        ][reference_finite].mean().item(),
        "reference_footprint_collision_fraction": metrics[
            "reference_footprint_collision"
        ].float().mean().item(),
        "reference_safety_margin_violation_fraction": metrics[
            "reference_safety_margin_violation"
        ].float().mean().item(),
    }
