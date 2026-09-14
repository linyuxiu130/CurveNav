"""Model-independent metric geometry for local trajectory evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from curvenav.evaluation.configuration import (
    SAFETY_CLEARANCE_M,
    query_configuration_field,
)
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
EXECUTION_PREFIX_HORIZONS_M = (0.5, 1.0)
MPC_REFERENCE_LENGTH_M = 2.0
MPC_REFERENCE_SPEED_MPS = 0.5
MPC_MAX_LINEAR_SPEED_MPS = 0.5
MPC_MAX_ANGULAR_SPEED_RADPS = 0.5
MPC_HORIZON_S = 30 * 0.1
TERMINAL_COLLISION_WINDOW_M = 0.25


@dataclass(frozen=True)
class CollisionVisibilityAttribution:
    """Source collisions partitioned by deployed raw-depth evidence."""

    metrics: dict[str, Tensor]
    current_visible_points: Tensor
    history_only_visible_points: Tensor
    unrecognized_points: Tensor


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


def paired_path_change_m(
    path: Tensor,
    baseline_path: Tensor,
    comparison_horizon_m: float = COMPARISON_HORIZON_M,
) -> Tensor:
    """Measure one intervention against its own unmodified policy output."""
    _validate_path(path, "path")
    _validate_path(baseline_path, "baseline_path")
    if path.shape[0] != baseline_path.shape[0] or comparison_horizon_m <= 0:
        raise ValueError("paired paths must share a batch and positive horizon")
    evaluated_length = path_arc_length(baseline_path).clamp_max(
        comparison_horizon_m
    )
    query = _uniform_distance(evaluated_length, COMPARISON_PATH_SAMPLES)
    intervened = resample_path_at_distance(path, query)
    baseline = resample_path_at_distance(baseline_path, query)
    return torch.linalg.vector_norm(intervened - baseline, dim=-1).mean(dim=-1)


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
    sampled = query_configuration_field(
        configuration_field,
        dense_path,
        planning_horizon_m,
    )
    covered = active & sampled.support_observed
    clearance = sampled.signed_clearance_m
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


def collision_visibility_attribution(
    source_query: SourcePathQuery,
    full_history_field: Tensor,
    current_frame_field: Tensor,
    planning_horizon_m: float,
) -> CollisionVisibilityAttribution:
    """Partition identical source-collision points by current/history evidence."""
    dense_path = source_query.local_path
    full = query_configuration_field(
        full_history_field, dense_path, planning_horizon_m
    )
    current = query_configuration_field(
        current_frame_field, dense_path, planning_horizon_m
    )
    truth_collision = source_query.active & (source_query.clearance_m < 0.0)
    truth_free = source_query.active & ~truth_collision
    full_coverage = source_query.active & full.support_observed
    current_coverage = source_query.active & current.support_observed
    full_recognized = full_coverage & (full.signed_clearance_m < 0.0)
    current_recognized = current_coverage & (current.signed_clearance_m < 0.0)
    current_visible = truth_collision & current_recognized
    history_only_visible = truth_collision & full_recognized & ~current_recognized
    unrecognized = truth_collision & ~full_recognized

    def point_count(mask: Tensor) -> Tensor:
        return mask.sum(dim=-1)

    full_visible_trajectory = (truth_collision & full_recognized).any(dim=-1)
    current_visible_trajectory = current_visible.any(dim=-1)
    history_only_trajectory = full_visible_trajectory & ~current_visible_trajectory
    collision_trajectory = truth_collision.any(dim=-1)
    first_collision_index = truth_collision.to(torch.int64).argmax(dim=-1)

    def first_collision(mask: Tensor) -> Tensor:
        return collision_trajectory & mask.gather(
            1, first_collision_index[:, None]
        ).squeeze(1)

    last_active = source_query.active.sum(dim=-1).clamp_min(1) - 1

    def endpoint(mask: Tensor) -> Tensor:
        return mask.gather(1, last_active[:, None]).squeeze(1)

    metrics = {
        "truth_collision_point_count": point_count(truth_collision),
        "truth_collision_point_raw_depth_count": point_count(
            truth_collision & full_recognized
        ),
        "truth_collision_point_raw_ray_coverage_count": point_count(
            truth_collision & full_coverage
        ),
        "truth_collision_point_raw_ray_coverage_missed_count": point_count(
            truth_collision & full_coverage & ~full_recognized
        ),
        "truth_collision_point_outside_raw_ray_coverage_count": point_count(
            truth_collision & ~full_coverage
        ),
        "raw_depth_false_collision_point_count": point_count(
            truth_free & full_recognized
        ),
        "truth_collision_point_current_depth_count": point_count(current_visible),
        "truth_collision_point_history_only_depth_count": point_count(
            history_only_visible
        ),
        "truth_collision_point_unrecognized_by_full_depth_count": point_count(
            unrecognized
        ),
        "truth_collision_trajectory": collision_trajectory,
        "truth_collision_trajectory_recognized_by_raw_depth": (
            full_visible_trajectory
        ),
        "truth_collision_trajectory_recognized_by_current_depth": (
            current_visible_trajectory
        ),
        "truth_collision_trajectory_has_additional_history_evidence": (
            history_only_visible.any(dim=-1)
        ),
        "truth_collision_trajectory_recognized_only_with_history": (
            history_only_trajectory
        ),
        "truth_collision_trajectory_unrecognized_by_full_depth": (
            collision_trajectory & ~full_visible_trajectory
        ),
        "first_collision_current_depth_visible": first_collision(current_visible),
        "first_collision_history_only_depth_visible": first_collision(
            history_only_visible
        ),
        "first_collision_unrecognized_by_full_depth": first_collision(unrecognized),
        "path_endpoint_collision_current_depth_visible": endpoint(
            current_visible
        ),
        "path_endpoint_collision_history_only_depth_visible": endpoint(
            history_only_visible
        ),
        "path_endpoint_collision_unrecognized_by_full_depth": endpoint(
            unrecognized
        ),
    }
    return CollisionVisibilityAttribution(
        metrics=metrics,
        current_visible_points=current_visible,
        history_only_visible_points=history_only_visible,
        unrecognized_points=unrecognized,
    )


def source_execution_prefix_metrics(
    query: SourcePathQuery,
) -> dict[str, Tensor]:
    """Measure the source-truth portion that a receding-horizon policy executes.

    The source query visits crossed cells and endpoints with at most 2.5 cm spacing, so these
    metrics add no map queries and use exactly the same physical truth as the
    full-path collision report.
    """
    local = query.local_path
    segment = torch.linalg.vector_norm(local[:, 1:] - local[:, :-1], dim=-1)
    distance = torch.cat(
        (torch.zeros_like(segment[:, :1]), segment.cumsum(dim=-1)), dim=-1
    )
    evaluated_length = distance.masked_fill(~query.active, 0.0).amax(dim=-1)
    collision = query.active & (query.clearance_m < 0.0)
    margin = query.active & (query.clearance_m < SAFETY_CLEARANCE_M)
    terminal_window = query.active & (
        distance >= (evaluated_length - TERMINAL_COLLISION_WINDOW_M)[:, None] - 1e-6
    )
    last_active = query.active.sum(dim=-1).clamp_min(1) - 1
    endpoint_collision = collision.gather(1, last_active[:, None]).squeeze(1)
    terminal_collision = (collision & terminal_window).any(dim=-1)
    preterminal_collision = (collision & ~terminal_window).any(dim=-1)

    def first_violation(mask: Tensor) -> Tensor:
        first = distance.masked_fill(~mask, torch.inf).amin(dim=-1)
        return torch.where(torch.isfinite(first), first, evaluated_length)

    result = {
        "distance_to_first_collision_m": first_violation(collision),
        "distance_to_first_margin_violation_m": first_violation(margin),
        "path_endpoint_collision": endpoint_collision,
        "terminal_0p25m_collision": terminal_collision,
        "collision_confined_to_terminal_0p25m": (
            terminal_collision & ~preterminal_collision
        ),
    }
    for horizon_m in EXECUTION_PREFIX_HORIZONS_M:
        key = str(horizon_m).replace(".", "p")
        inside_prefix = query.active & (distance <= horizon_m + 1e-6)
        result[f"execution_prefix_{key}m_collision"] = (
            collision & inside_prefix
        ).any(dim=-1)
        result[f"execution_prefix_{key}m_margin_violation"] = (
            margin & inside_prefix
        ).any(dim=-1)
    return result


def controller_tracking_metrics(path: Tensor) -> dict[str, Tensor]:
    """Measure the benchmark's remaining-polyline curvature and desired speed.

    Diagnostic only: float64 CPU geometry follows the online MPC reset, including
    repeated endpoints and projection of the current origin onto an old plan.
    """
    _validate_path(path, "path")
    rows = []
    for points in path.detach().cpu().double().numpy():
        points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        arc = np.r_[0.0, np.cumsum(lengths)]
        start = 0.0
        if len(segments):
            fraction = np.clip(-np.sum(points[:-1] * segments, axis=1) / lengths**2, 0, 1)
            projected = points[:-1] + fraction[:, None] * segments
            nearest = np.argmin(np.linalg.norm(projected, axis=1))
            start = arc[nearest] + fraction[nearest] * lengths[nearest]
        origin = np.array([np.interp(start, arc, points[:, axis]) for axis in range(2)])
        points = np.vstack((origin, points[arc > start]))
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        arc = np.r_[0.0, np.cumsum(lengths)]
        maximum = 0.0
        length_speed = min(MPC_MAX_LINEAR_SPEED_MPS,
                           MPC_REFERENCE_SPEED_MPS * min(arc[-1] / MPC_REFERENCE_LENGTH_M, 1.0))
        if len(segments) >= 2:
            before, after = segments[:-1], segments[1:]
            turn = np.abs(np.arctan2(
                before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0],
                np.sum(before * after, axis=1),
            ))
            curvature = turn / (0.5 * (lengths[:-1] + lengths[1:]))
            curvature = np.r_[curvature[0], curvature, curvature[-1]]
            end = np.searchsorted(arc, length_speed * MPC_HORIZON_S, side="right") + 1
            maximum = float(curvature[:end].max())
        curvature_speed = min(MPC_MAX_LINEAR_SPEED_MPS,
                              MPC_MAX_ANGULAR_SPEED_RADPS / max(maximum, 1e-6))
        rows.append((maximum, length_speed, curvature_speed))
    maximum, length_speed, curvature_speed = path.new_tensor(rows).unbind(dim=1)
    initial = path[:, 1] - path[:, 0]
    return {
        "mpc_max_curvature_lookahead_inv_m": maximum,
        "mpc_length_limited_speed_mps": length_speed,
        "mpc_curvature_limited_speed_mps": curvature_speed,
        "mpc_desired_speed_mps": torch.minimum(length_speed, curvature_speed),
        "mpc_curvature_is_active": curvature_speed < length_speed,
        "initial_tangent_heading_error_rad": torch.atan2(
            initial[:, 1], initial[:, 0]
        ).abs(),
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
    result = {
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
    for horizon_m in EXECUTION_PREFIX_HORIZONS_M:
        key = str(horizon_m).replace(".", "p")
        for suffix in ("collision", "margin_violation"):
            name = f"execution_prefix_{key}m_{suffix}"
            result[f"{name}_fraction"] = metrics[name].float().mean().item()
    result.update(
        path_endpoint_collision_fraction=metrics["path_endpoint_collision"]
        .float()
        .mean()
        .item(),
        terminal_0p25m_collision_fraction=metrics["terminal_0p25m_collision"]
        .float()
        .mean()
        .item(),
        collision_confined_to_terminal_0p25m_fraction=metrics[
            "collision_confined_to_terminal_0p25m"
        ]
        .float()
        .mean()
        .item(),
    )
    result.update(
        distance_to_first_collision_m_mean=metrics[
            "distance_to_first_collision_m"
        ].mean().item(),
        distance_to_first_margin_violation_m_mean=metrics[
            "distance_to_first_margin_violation_m"
        ].mean().item(),
        mpc_desired_speed_mps_mean=metrics["mpc_desired_speed_mps"].mean().item(),
        mpc_desired_speed_mps_p05=torch.quantile(
            metrics["mpc_desired_speed_mps"], 0.05
        ).item(),
        mpc_curvature_limited_fraction=metrics["mpc_curvature_is_active"]
        .float()
        .mean()
        .item(),
        mpc_max_curvature_lookahead_inv_m_p95=torch.quantile(
            metrics["mpc_max_curvature_lookahead_inv_m"], 0.95
        ).item(),
        initial_tangent_heading_error_rad_p95=torch.quantile(
            metrics["initial_tangent_heading_error_rad"], 0.95
        ).item(),
    )
    return result
