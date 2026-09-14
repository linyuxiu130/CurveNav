"""Held-out source-truth evaluation for CurveNav's single local policy."""

import argparse
from dataclasses import dataclass, fields
import json
from pathlib import Path
import time

import torch
from torch import Tensor

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data.batch import unpack_policy_batch
from curvenav.data.loader import build_policy_validation_loader
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.evaluation.metrics import (
    collision_visibility_attribution,
    configuration_space_safety_metrics,
    controller_tracking_metrics,
    paired_path_change_m,
    source_execution_prefix_metrics,
    summarize_configuration_safety,
    summarize_metrics,
    trajectory_metrics,
)
from curvenav.evaluation.protocol import (
    summarize_goal_bearing_strata,
    summarize_strata,
)
from curvenav.evaluation.report import write_case_report
from curvenav.factory import build_evaluation_projector, build_policy
from curvenav.models import CurveNavPolicy
from curvenav.models.policy import INFERENCE_CANDIDATES
from curvenav.training.critic import RouteUtilityTeacher
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.trajectory import local_terminal_goal
from curvenav.types import ConditionFeatures, TrajectoryPrediction


VALIDATION_BATCH_SIZE = 32
MODEL_LATENCY_REPEATS = 50


@dataclass(frozen=True)
class PolicyMeasurements:
    metrics: dict[str, Tensor]
    current_frame_metrics: dict[str, Tensor]
    depth_swap_metrics: dict[str, Tensor]
    point_goal_swap_metrics: dict[str, Tensor]
    batch_latency_ms: Tensor
    all_passes_latency_ms: Tensor
    model_latency_ms: Tensor
    wall_seconds: float
    samples: int
    sample_data: dict[str, Tensor]


def _sample(
    policy: CurveNavPolicy,
    batch: dict[str, Tensor],
):
    prepared = unpack_policy_batch(batch)
    prediction = policy.sample(prepared.condition)
    return prepared, prediction


def _cached_interventions(
    policy: CurveNavPolicy, encoded: ConditionFeatures, point_goal: Tensor,
) -> tuple[TrajectoryPrediction, TrajectoryPrediction]:
    """Decode each unique (scene, goal) pair once, including odd-sized tails."""
    batch = len(point_goal)
    index = torch.arange(batch, device=point_goal.device)
    permutation = index.roll(batch // 2)
    # For even B, p(p(i)) = i: (p(i), i) and (i, p(i)) are the same
    # scene/goal pairs in a different order. Odd tails retain both sets.
    pairs, inverse = torch.unique(
        torch.cat((permutation * batch + index, index * batch + permutation)),
        sorted=True, return_inverse=True,
    )
    predictions = []
    for pair in pairs.split(batch):
        scene_index, goal_index = pair // batch, pair % batch
        scene = ConditionFeatures(**{
            field.name: getattr(encoded, field.name)[scene_index]
            for field in fields(encoded)
        })
        goal = point_goal[goal_index]
        scene.terminal_goal = local_terminal_goal(goal, policy.planning_horizon_m)
        predictions.append(policy.sample_encoded(scene, goal))
    values = {
        field.name: torch.cat([getattr(p, field.name) for p in predictions])[inverse]
        for field in fields(TrajectoryPrediction)
    }
    depth_swap = TrajectoryPrediction(**{name: value[:batch] for name, value in values.items()})
    goal_swap = TrajectoryPrediction(**{name: value[batch:] for name, value in values.items()})
    return depth_swap, goal_swap


def _intervention_metrics(
    path: Tensor,
    baseline_path: Tensor,
    source_query: SourceConfigurationSpaceQuery,
    source_grid_index: Tensor,
    source_origin_xy: Tensor,
    source_yaw_rad: Tensor,
    planning_horizon_m: float,
) -> dict[str, Tensor]:
    metrics = {"path_change_from_policy_m": paired_path_change_m(path, baseline_path)}
    source = source_query.query(
        path,
        source_grid_index,
        source_origin_xy,
        source_yaw_rad,
        planning_horizon_m,
    )
    metrics.update(
        source_query.safety_metrics_from_query(path, source, planning_horizon_m)
    )
    metrics.update(source_execution_prefix_metrics(source))
    return metrics


def _collision_detection_summary(metrics: dict[str, Tensor]) -> dict[str, int | float]:
    """Report where source collisions occur and which frames reveal them."""
    truth_points = metrics["truth_collision_point_count"].sum()
    truth_trajectories = metrics["truth_collision_trajectory"].sum()

    def aggregate(name: str) -> int:
        return int(metrics[name].sum().item())

    def fraction(numerator: int, denominator: Tensor) -> float:
        return float(numerator / max(int(denominator.item()), 1))

    raw_points = aggregate("truth_collision_point_raw_depth_count")
    raw_ray_coverage_points = aggregate("truth_collision_point_raw_ray_coverage_count")
    raw_trajectories = aggregate("truth_collision_trajectory_recognized_by_raw_depth")
    current_points = aggregate("truth_collision_point_current_depth_count")
    history_only_points = aggregate("truth_collision_point_history_only_depth_count")
    unrecognized_points = aggregate(
        "truth_collision_point_unrecognized_by_full_depth_count"
    )
    current_trajectories = aggregate(
        "truth_collision_trajectory_recognized_by_current_depth"
    )
    history_only_trajectories = aggregate(
        "truth_collision_trajectory_recognized_only_with_history"
    )
    additional_history_trajectories = aggregate(
        "truth_collision_trajectory_has_additional_history_evidence"
    )
    unrecognized_trajectories = aggregate(
        "truth_collision_trajectory_unrecognized_by_full_depth"
    )
    first_current = aggregate("first_collision_current_depth_visible")
    first_history = aggregate("first_collision_history_only_depth_visible")
    first_unrecognized = aggregate("first_collision_unrecognized_by_full_depth")
    prefix_1m = metrics["execution_prefix_1p0m_collision"]
    prefix_1m_count = int(prefix_1m.sum().item())
    prefix_1m_first_current = int(
        (prefix_1m & metrics["first_collision_current_depth_visible"]).sum().item()
    )
    prefix_1m_first_history = int(
        (prefix_1m & metrics["first_collision_history_only_depth_visible"]).sum().item()
    )
    prefix_1m_first_unrecognized = int(
        (prefix_1m & metrics["first_collision_unrecognized_by_full_depth"]).sum().item()
    )

    def prefix_fraction(count: int) -> float:
        return float(count / max(prefix_1m_count, 1))

    return {
        "ground_truth_collision_trajectory_count": int(truth_trajectories.item()),
        "ground_truth_collision_point_count": int(truth_points.item()),
        "ground_truth_collision_points_recognized_by_raw_depth_count": raw_points,
        "ground_truth_collision_points_recognized_by_raw_depth_fraction": fraction(
            raw_points, truth_points
        ),
        "ground_truth_collision_points_inside_raw_ray_coverage_count": (
            raw_ray_coverage_points
        ),
        "ground_truth_collision_points_inside_raw_ray_coverage_fraction": fraction(
            raw_ray_coverage_points, truth_points
        ),
        "ground_truth_collision_points_raw_ray_coverage_but_missed_count": aggregate(
            "truth_collision_point_raw_ray_coverage_missed_count"
        ),
        "ground_truth_collision_points_outside_raw_ray_coverage_count": aggregate(
            "truth_collision_point_outside_raw_ray_coverage_count"
        ),
        "ground_truth_collision_trajectories_recognized_by_raw_depth_count": (
            raw_trajectories
        ),
        "ground_truth_collision_trajectories_recognized_by_raw_depth_fraction": (
            fraction(raw_trajectories, truth_trajectories)
        ),
        "collision_points_visible_in_current_frame_count": current_points,
        "collision_points_visible_in_current_frame_fraction": fraction(
            current_points, truth_points
        ),
        "collision_points_visible_only_through_history_count": history_only_points,
        "collision_points_visible_only_through_history_fraction": fraction(
            history_only_points, truth_points
        ),
        "collision_points_unrecognized_by_all_history_frames_count": (
            unrecognized_points
        ),
        "collision_points_unrecognized_by_all_history_frames_fraction": fraction(
            unrecognized_points, truth_points
        ),
        "collision_trajectories_recognized_in_current_frame_count": (
            current_trajectories
        ),
        "collision_trajectories_recognized_in_current_frame_fraction": fraction(
            current_trajectories, truth_trajectories
        ),
        "collision_trajectories_recognized_only_through_history_count": (
            history_only_trajectories
        ),
        "collision_trajectories_recognized_only_through_history_fraction": fraction(
            history_only_trajectories, truth_trajectories
        ),
        "collision_trajectories_with_additional_history_evidence_count": (
            additional_history_trajectories
        ),
        "collision_trajectories_unrecognized_by_all_history_frames_count": (
            unrecognized_trajectories
        ),
        "collision_trajectories_unrecognized_by_all_history_frames_fraction": fraction(
            unrecognized_trajectories, truth_trajectories
        ),
        "first_collision_visible_in_current_frame_count": first_current,
        "first_collision_visible_in_current_frame_fraction": fraction(
            first_current, truth_trajectories
        ),
        "first_collision_visible_only_through_history_count": first_history,
        "first_collision_visible_only_through_history_fraction": fraction(
            first_history, truth_trajectories
        ),
        "first_collision_unrecognized_by_all_history_frames_count": first_unrecognized,
        "first_collision_unrecognized_by_all_history_frames_fraction": fraction(
            first_unrecognized, truth_trajectories
        ),
        "execution_prefix_1p0m_collision_count": prefix_1m_count,
        "first_collision_within_1p0m_visible_in_current_frame_count": (
            prefix_1m_first_current
        ),
        "first_collision_within_1p0m_visible_in_current_frame_fraction": (
            prefix_fraction(prefix_1m_first_current)
        ),
        "first_collision_within_1p0m_visible_only_through_history_count": (
            prefix_1m_first_history
        ),
        "first_collision_within_1p0m_visible_only_through_history_fraction": (
            prefix_fraction(prefix_1m_first_history)
        ),
        "first_collision_within_1p0m_unrecognized_by_all_history_frames_count": (
            prefix_1m_first_unrecognized
        ),
        "first_collision_within_1p0m_unrecognized_by_all_history_frames_fraction": (
            prefix_fraction(prefix_1m_first_unrecognized)
        ),
        "path_endpoint_collision_count": aggregate("path_endpoint_collision"),
        "endpoint_collision_point_visible_in_current_frame_count": aggregate(
            "path_endpoint_collision_current_depth_visible"
        ),
        "endpoint_collision_point_visible_only_through_history_count": aggregate(
            "path_endpoint_collision_history_only_depth_visible"
        ),
        "endpoint_collision_point_unrecognized_by_all_history_frames_count": aggregate(
            "path_endpoint_collision_unrecognized_by_full_depth"
        ),
        "terminal_0p25m_collision_count": aggregate("terminal_0p25m_collision"),
        "collision_confined_to_terminal_0p25m_count": aggregate(
            "collision_confined_to_terminal_0p25m"
        ),
        "raw_depth_false_collision_point_count": aggregate(
            "raw_depth_false_collision_point_count"
        ),
    }


def _validate_reference_source_safety(metrics: dict[str, Tensor]) -> None:
    """Reject a split whose stored provenance disagrees with its experts."""
    collision = metrics["reference_footprint_collision"].float().mean().item()
    margin = metrics["reference_safety_margin_violation"].float().mean().item()
    coverage = metrics["reference_path_field_coverage_fraction"].mean().item()
    if collision > 0.0 or margin > 0.0 or coverage < 1.0:
        raise RuntimeError(
            "source C-space expert contract failed: "
            f"collision={collision:.6f}, margin={margin:.6f}, coverage={coverage:.6f}"
        )


def _time_model(
    policy: CurveNavPolicy,
    batch: dict[str, Tensor],
    repeats: int,
) -> Tensor:
    first = {name: value[:1] for name, value in batch.items()}
    for _ in range(2):
        _sample(policy, first)
    torch.cuda.synchronize()
    latency = []
    for _ in range(repeats):
        started = time.perf_counter()
        _sample(policy, first)
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - started) * 1000.0)
    return torch.tensor(latency, dtype=torch.float64)


@torch.no_grad()
def measure_policy(
    policy: CurveNavPolicy,
    loader,
    device: torch.device,
    source_query: SourceConfigurationSpaceQuery,
    depth_projector,
    model_latency_repeats: int = MODEL_LATENCY_REPEATS,
) -> PolicyMeasurements:
    """Collect one deterministic path and one history ablation per observation."""
    policy.to(device).eval()
    teacher = RouteUtilityTeacher(source_query, policy.planning_horizon_m)
    depth_projector.to(device).eval()
    warmup = next(iter(loader))
    _sample(policy, warmup)
    torch.cuda.synchronize(device)

    values: dict[str, list[Tensor]] = {}
    current_frame_values: dict[str, list[Tensor]] = {}
    depth_swap_values: dict[str, list[Tensor]] = {}
    point_goal_swap_values: dict[str, list[Tensor]] = {}
    sample_data: dict[str, list[Tensor]] = {}
    batch_latency = []
    all_passes_latency = []
    samples = 0
    evaluation_started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prepared = unpack_policy_batch(batch)
        encoded = policy.encode_condition(prepared.condition)
        prediction = policy.sample_encoded(encoded, prepared.condition.point_goal)
        end.record()
        end.synchronize()
        batch_latency.append(start.elapsed_time(end))

        current_frame_batch = dict(batch)
        current_frame_valid = torch.zeros_like(batch["observation_valid"])
        current_frame_valid[:, -1] = True
        current_frame_batch["observation_valid"] = current_frame_valid
        current_frame_batch["obstacle_memory"] = torch.zeros_like(batch["obstacle_memory"])
        current_prepared, current_prediction = _sample(
            policy, current_frame_batch
        )
        depth_swap_prediction, point_goal_swap_prediction = _cached_interventions(
            policy, encoded, prepared.condition.point_goal
        )
        inference_end = torch.cuda.Event(enable_timing=True)
        inference_end.record()

        reference_path, _ = policy.curve_codec.decode_values(
            prepared.target.curve_values.float()
        )
        source_grid_index = batch["source_grid_index"]
        source_origin_xy = batch["source_origin_xy"]
        source_yaw_rad = batch["source_yaw_rad"]
        source_prediction = source_query.query(
            prediction.path.float(),
            source_grid_index,
            source_origin_xy,
            source_yaw_rad,
            policy.planning_horizon_m,
        )
        metrics = trajectory_metrics(
            prediction.path.float(),
            reference_path,
            prepared.condition.point_goal.float(),
        )
        candidate_labels = teacher(prediction.candidates, batch)
        candidate_truth = candidate_labels.clearance_m
        teacher_scores = candidate_labels.score
        rows = torch.arange(len(candidate_truth), device=candidate_truth.device)
        selected_truth = candidate_truth[rows, prediction.selected_index]
        safe_available = (candidate_truth >= 0).any(dim=1)
        metrics.update({
            "candidate_safe_available": safe_available,
            "candidate_collision_fraction": (candidate_truth < 0).float().mean(1),
            "selected_whole_curve_collision": selected_truth < 0,
            "selection_missed_safe_candidate": safe_available & (selected_truth < 0),
            "selection_score_regret": teacher_scores.max(1).values - teacher_scores[rows, prediction.selected_index],
            "critic_score_mae": (prediction.scores - teacher_scores).abs().mean(1),
        })
        metrics["valid_observation_frames"] = prepared.condition.observation_valid.sum(
            dim=-1
        )
        projection = depth_projector(prepared.condition)
        current_projection = depth_projector(current_prepared.condition)
        depth_safety = configuration_space_safety_metrics(
            prediction.path.float(),
            projection.configuration_field,
            policy.planning_horizon_m,
        )
        metrics.update({f"depth_{name}": value for name, value in depth_safety.items()})
        reference_depth_safety = configuration_space_safety_metrics(
            reference_path,
            projection.configuration_field,
            policy.planning_horizon_m,
        )
        metrics.update(
            {
                f"reference_depth_{name}": value
                for name, value in reference_depth_safety.items()
            }
        )
        metrics.update(
            source_query.safety_metrics_from_query(
                prediction.path.float(),
                source_prediction,
                policy.planning_horizon_m,
            )
        )
        metrics.update(source_execution_prefix_metrics(source_prediction))
        metrics.update(controller_tracking_metrics(prediction.path.float()))
        visibility = collision_visibility_attribution(
            source_prediction,
            projection.configuration_field,
            current_projection.configuration_field,
            policy.planning_horizon_m,
        )
        metrics.update(visibility.metrics)
        reference_source = source_query.query(
            reference_path,
            source_grid_index,
            source_origin_xy,
            source_yaw_rad,
            policy.planning_horizon_m,
        )
        reference_obstacle_metrics = source_query.safety_metrics_from_query(
            reference_path,
            reference_source,
            policy.planning_horizon_m,
        )
        for name, value in reference_obstacle_metrics.items():
            metrics[f"reference_{name}"] = value
        for name, value in controller_tracking_metrics(reference_path).items():
            metrics[f"reference_{name}"] = value
        straight_progress = torch.linspace(
            0.0,
            1.0,
            reference_path.shape[1],
            device=reference_path.device,
        )[None, :, None]
        straight_path = reference_path[:, :1] + straight_progress * (
            reference_path[:, -1:] - reference_path[:, :1]
        )
        straight_obstacle_metrics = source_query.safety_metrics(
            straight_path,
            source_grid_index,
            source_origin_xy,
            source_yaw_rad,
            policy.planning_horizon_m,
        )
        for name, value in straight_obstacle_metrics.items():
            metrics[f"straight_{name}"] = value

        current_metrics = trajectory_metrics(
            current_prediction.path.float(),
            reference_path,
            current_prepared.condition.point_goal.float(),
        )
        current_metrics["valid_observation_frames"] = (
            current_prepared.condition.observation_valid.sum(dim=-1)
        )
        current_source_prediction = source_query.query(
            current_prediction.path.float(),
            source_grid_index,
            source_origin_xy,
            source_yaw_rad,
            policy.planning_horizon_m,
        )
        current_metrics.update(
            source_query.safety_metrics_from_query(
                current_prediction.path.float(),
                current_source_prediction,
                policy.planning_horizon_m,
            )
        )
        current_metrics.update(
            source_execution_prefix_metrics(current_source_prediction)
        )
        current_visibility = collision_visibility_attribution(
            current_source_prediction,
            projection.configuration_field,
            current_projection.configuration_field,
            policy.planning_horizon_m,
        )
        current_metrics.update(current_visibility.metrics)
        current_metrics["path_change_from_full_history_m"] = paired_path_change_m(
            current_prediction.path.float(), prediction.path.float()
        )
        depth_swap_metrics = _intervention_metrics(
            depth_swap_prediction.path.float(),
            prediction.path.float(),
            source_query,
            source_grid_index,
            source_origin_xy,
            source_yaw_rad,
            policy.planning_horizon_m,
        )
        point_goal_swap_metrics = _intervention_metrics(
            point_goal_swap_prediction.path.float(),
            prediction.path.float(),
            source_query,
            source_grid_index,
            source_origin_xy,
            source_yaw_rad,
            policy.planning_horizon_m,
        )
        for name, value in metrics.items():
            values.setdefault(name, []).append(value.cpu())
        for name, value in current_metrics.items():
            current_frame_values.setdefault(name, []).append(value.cpu())
        for name, value in depth_swap_metrics.items():
            depth_swap_values.setdefault(name, []).append(value.cpu())
        for name, value in point_goal_swap_metrics.items():
            point_goal_swap_values.setdefault(name, []).append(value.cpu())
        for name, value in {
            "predicted_path": prediction.path.float(),
            "candidate_scores": prediction.scores,
            "candidate_clearance_m": candidate_truth,
            "candidate_teacher_score": teacher_scores,
            "candidate_geodesic_progress_m": candidate_labels.progress_m,
            "selected_candidate_index": prediction.selected_index,
            "current_frame_predicted_path": current_prediction.path.float(),
            "reference_path": reference_path,
            "point_goal": prepared.condition.point_goal.float(),
            "configuration_field": projection.configuration_field[:, (0, 3, 4)].to(
                dtype=torch.float16
            ),
            "configuration_extent_m": torch.full(
                (len(prediction.path),),
                policy.planning_horizon_m,
                device=prediction.path.device,
            ),
            "current_configuration_ray_coverage": (
                current_projection.configuration_field[:, 3] > 0.5
            ),
            "current_configuration_forbidden": (
                current_projection.configuration_field[:, 4] > 0.5
            ),
            "source_grid_index": source_grid_index,
            "source_origin_xy": source_origin_xy,
            "source_yaw_rad": source_yaw_rad,
            "source_collision_path_points": source_prediction.local_path,
            "source_collision_current_visible": visibility.current_visible_points,
            "source_collision_history_only_visible": (
                visibility.history_only_visible_points
            ),
            "source_collision_unrecognized": visibility.unrecognized_points,
        }.items():
            sample_data.setdefault(name, []).append(value.cpu())
        samples += len(prediction.path)
        all_passes_latency.append(start.elapsed_time(inference_end))
        if batch_index % 100 == 0:
            elapsed = time.perf_counter() - evaluation_started
            print(json.dumps({"event": "evaluation_progress", "samples": samples,
                              "elapsed_seconds": elapsed,
                              "samples_per_second": samples / elapsed}), flush=True)

    wall_seconds = time.perf_counter() - evaluation_started
    return PolicyMeasurements(
        metrics={name: torch.cat(parts) for name, parts in values.items()},
        current_frame_metrics={
            name: torch.cat(parts) for name, parts in current_frame_values.items()
        },
        depth_swap_metrics={
            name: torch.cat(parts) for name, parts in depth_swap_values.items()
        },
        point_goal_swap_metrics={
            name: torch.cat(parts) for name, parts in point_goal_swap_values.items()
        },
        batch_latency_ms=torch.tensor(batch_latency, dtype=torch.float64),
        all_passes_latency_ms=torch.tensor(all_passes_latency, dtype=torch.float64),
        model_latency_ms=(
            _time_model(policy, warmup, model_latency_repeats)
            if model_latency_repeats
            else torch.empty(0, dtype=torch.float64)
        ),
        wall_seconds=wall_seconds,
        samples=samples,
        sample_data={
            name: (torch.nn.utils.rnn.pad_sequence(
                [row for part in parts for row in part], batch_first=True,
            ) if name.startswith("source_collision_") else torch.cat(parts))
            for name, parts in sample_data.items()
        },
    )


def evaluate_measurements(measurements: PolicyMeasurements) -> dict[str, object]:
    """Summarize one inference pass without model-dependent safety surrogates."""
    seconds = measurements.batch_latency_ms.sum().item() / 1000.0
    current_frame_summary = summarize_metrics(measurements.current_frame_metrics)
    metrics = measurements.metrics
    policy_summary = summarize_metrics(metrics)
    current_metrics = measurements.current_frame_metrics
    depth_swap = measurements.depth_swap_metrics
    point_goal_swap = measurements.point_goal_swap_metrics
    current_history_only_risk = current_metrics[
        "truth_collision_trajectory_recognized_only_with_history"
    ].bool()
    history_only_risk_count = int(current_history_only_risk.sum().item())
    full_safe_on_history_only_risk = (
        current_history_only_risk & ~metrics["footprint_collision"].bool()
    )
    current_collision = current_metrics["footprint_collision"].bool()
    full_collision = metrics["footprint_collision"].bool()
    full_avoids_current_collision = current_collision & ~full_collision
    full_introduces_collision = full_collision & ~current_collision
    return {
        "protocol": "curvenav_metric_local_validation_source_cspace",
        "samples": measurements.samples,
        "trajectories_per_observation": INFERENCE_CANDIDATES,
        "selected_trajectories_per_observation": 1,
        **policy_summary,
        "candidate_selection": {
            name: metrics[name].float().mean().item()
            for name in ("candidate_safe_available", "candidate_collision_fraction",
                         "selected_whole_curve_collision", "selection_missed_safe_candidate",
                         "selection_score_regret", "critic_score_mae")
        },
        **summarize_configuration_safety(metrics),
        "current_frame_ablation": {
            "fixed_horizon_ade_m_mean": current_frame_summary[
                "fixed_horizon_ade_m_mean"
            ],
            "fixed_horizon_ade_increase_m": current_frame_summary[
                "fixed_horizon_ade_m_mean"
            ]
            - policy_summary["fixed_horizon_ade_m_mean"],
            "fixed_horizon_fde_m_mean": current_frame_summary[
                "fixed_horizon_fde_m_mean"
            ],
            "goal_progress_regret_m_mean": current_frame_summary[
                "goal_progress_regret_m_mean"
            ],
            "footprint_collision_fraction": current_metrics["footprint_collision"]
            .float()
            .mean()
            .item(),
            "safety_margin_violation_fraction": current_metrics[
                "safety_margin_violation"
            ]
            .float()
            .mean()
            .item(),
            "path_change_from_full_history_m_mean": current_metrics[
                "path_change_from_full_history_m"
            ]
            .mean()
            .item(),
            "current_only_collision_with_history_only_evidence_count": (
                history_only_risk_count
            ),
            "full_history_avoids_current_only_history_visible_collision_count": int(
                full_safe_on_history_only_risk.sum().item()
            ),
            "full_history_avoidance_fraction_on_history_only_risk": float(
                full_safe_on_history_only_risk.sum().item()
                / max(history_only_risk_count, 1)
            ),
            "full_history_avoids_current_only_collision_count": int(
                full_avoids_current_collision.sum().item()
            ),
            "full_history_avoids_current_only_collision_fraction": float(
                full_avoids_current_collision.sum().item()
                / max(int(current_collision.sum().item()), 1)
            ),
            "full_history_introduces_collision_vs_current_only_count": int(
                full_introduces_collision.sum().item()
            ),
        },
        "condition_causality_audit": {
            "interpretation": "paired intervention only; not a navigation score",
            "depth_swap_path_change_m": depth_swap["path_change_from_policy_m"]
            .mean()
            .item(),
            "depth_swap_footprint_collision_fraction": depth_swap["footprint_collision"]
            .float()
            .mean()
            .item(),
            "depth_swap_execution_prefix_1p0m_collision_fraction": depth_swap[
                "execution_prefix_1p0m_collision"
            ]
            .float()
            .mean()
            .item(),
            "point_goal_swap_path_change_m": point_goal_swap[
                "path_change_from_policy_m"
            ]
            .mean()
            .item(),
            "point_goal_swap_footprint_collision_fraction": point_goal_swap[
                "footprint_collision"
            ]
            .float()
            .mean()
            .item(),
            "point_goal_swap_execution_prefix_1p0m_collision_fraction": (
                point_goal_swap["execution_prefix_1p0m_collision"].float().mean().item()
            ),
        },
        "raw_depth_path_support": {
            "interpretation": (
                "strict four-corner observed support on the same 0.025m path grid"
            ),
            "predicted_fraction_mean": metrics["depth_path_field_coverage_fraction"]
            .mean()
            .item(),
            "reference_fraction_mean": metrics[
                "reference_depth_path_field_coverage_fraction"
            ]
            .mean()
            .item(),
            "predicted_minus_reference_mean": (
                metrics["depth_path_field_coverage_fraction"]
                - metrics["reference_depth_path_field_coverage_fraction"]
            )
            .mean()
            .item(),
            "predicted_below_reference_fraction": (
                metrics["depth_path_field_coverage_fraction"]
                < metrics["reference_depth_path_field_coverage_fraction"] - 1e-6
            )
            .float()
            .mean()
            .item(),
        },
        "raw_depth_collision_attribution": _collision_detection_summary(metrics),
        "batch32_latency_ms_mean": measurements.batch_latency_ms.mean().item(),
        "base_policy_forward_observations_per_second": (measurements.samples / seconds),
        "full_evaluation_observations_per_second": (
            measurements.samples / measurements.wall_seconds
        ),
        "full_evaluation_wall_seconds": measurements.wall_seconds,
        "all_policy_passes_cuda_seconds": measurements.all_passes_latency_ms.sum().item() / 1000.0,
        "model_latency_batch1_ms_p50": torch.quantile(
            measurements.model_latency_ms, 0.50
        ).item(),
        "model_latency_batch1_ms_p95": torch.quantile(
            measurements.model_latency_ms, 0.95
        ).item(),
        "strata": summarize_strata(metrics),
        "goal_bearing_strata": summarize_goal_bearing_strata(metrics),
    }


def run_evaluation(
    config: CurveNavConfig,
    checkpoint_path: Path,
    artifact_dir: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CurveNav evaluation requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_policy_contract(checkpoint, config)
    policy = build_policy(config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy)
    policy.eval()
    bundle = build_policy_validation_loader(
        config.data,
        config.trajectory,
        batch_size=VALIDATION_BATCH_SIZE,
        num_workers=config.training.num_workers,
    )
    loader = CudaPrefetchLoader(bundle.loader, bundle.depth_bank, torch.device("cuda"))
    source_query = SourceConfigurationSpaceQuery.from_prepared_split(
        config.data.root,
        "validation",
    )
    depth_projector = build_evaluation_projector(config)
    print(json.dumps({"event": "evaluation_start", "checkpoint_step": int(checkpoint["step"]),
                      "samples": bundle.samples, "candidates": INFERENCE_CANDIDATES}), flush=True)
    measurements = measure_policy(
        policy,
        loader,
        torch.device("cuda"),
        source_query,
        depth_projector,
    )
    if measurements.samples != bundle.samples:
        raise RuntimeError(
            f"evaluated {measurements.samples} samples, expected {bundle.samples}"
        )
    _validate_reference_source_safety(measurements.metrics)
    result = evaluate_measurements(measurements)
    result.update(
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_step=int(checkpoint["step"]),
        weights="ema",
        evaluation_wall_seconds_including_setup=time.perf_counter() - started,
    )
    if artifact_dir is not None:
        artifact_dir = artifact_dir.expanduser().resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        case_path = write_case_report(
            artifact_dir,
            measurements.metrics,
            measurements.sample_data,
            source_query,
        )
        metrics_path = artifact_dir / "offline-metrics.json"
        result["case_report"] = str(case_path)
        result["case_visualization"] = str(case_path.with_suffix(".svg"))
        result["metrics_report"] = str(metrics_path)
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    run_evaluation(load_config(args.config), args.checkpoint, args.artifact_dir)


if __name__ == "__main__":
    main()
