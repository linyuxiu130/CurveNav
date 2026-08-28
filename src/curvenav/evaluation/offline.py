"""Held-out evaluation for CurveNav's deterministic local policy."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch import Tensor

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data.batch import unpack_policy_batch
from curvenav.data.loader import build_policy_validation_loader
from curvenav.factory import build_policy
from curvenav.models import CurveNavPolicy
from curvenav.models.safety import SAFETY_CLEARANCE_M
from curvenav.physical import ROBOT_FOOTPRINT_RADIUS_M
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.trajectory import path_arc_length


VALIDATION_BATCH_SIZE = 32
ONLINE_LATENCY_REPEATS = 50


@dataclass(frozen=True)
class PolicyMeasurements:
    metrics: dict[str, Tensor]
    current_frame_metrics: dict[str, Tensor]
    batch_latency_ms: Tensor
    online_latency_ms: Tensor
    samples: int


def trajectory_batch_metrics(
    path: Tensor,
    curvature: Tensor,
    reference_path: Tensor,
    reference_curvature: Tensor,
    point_goal: Tensor,
) -> dict[str, Tensor]:
    """Return aligned per-observation geometry for one predicted trajectory."""
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError("path must have shape [B,P,2]")
    batch_size, path_points, _ = path.shape
    if curvature.shape != (batch_size, path_points):
        raise ValueError("curvature shape must match path")
    if reference_path.shape != path.shape:
        raise ValueError("reference_path shape must match path")
    if reference_curvature.shape != curvature.shape:
        raise ValueError("reference_curvature shape must match reference_path")
    if point_goal.shape != (batch_size, 2):
        raise ValueError("point_goal must have shape [B,2]")

    difference = path - reference_path
    point_error = torch.linalg.vector_norm(difference, dim=-1)
    predicted_length = path_arc_length(path)
    reference_length = path_arc_length(reference_path)
    goal_distance = torch.linalg.vector_norm(point_goal, dim=-1)
    path_delta = path[:, 1:] - path[:, :-1]
    reference_delta = reference_path[:, 1:] - reference_path[:, :-1]
    path_segment_length = torch.linalg.vector_norm(path_delta, dim=-1)
    reference_segment_length = torch.linalg.vector_norm(reference_delta, dim=-1)
    total_abs_heading_change = (
        0.5 * (curvature[:, :-1].abs() + curvature[:, 1:].abs())
        * path_segment_length
    ).sum(dim=-1)
    reference_total_abs_heading_change = (
        0.5
        * (reference_curvature[:, :-1].abs() + reference_curvature[:, 1:].abs())
        * reference_segment_length
    ).sum(dim=-1)
    tangent_dot = (path_delta[:, 1:] * path_delta[:, :-1]).sum(dim=-1)
    final_predicted_direction = path_delta[:, -1] / torch.linalg.vector_norm(
        path_delta[:, -1], dim=-1, keepdim=True
    ).clamp_min(1e-6)
    final_reference_direction = reference_delta[:, -1] / torch.linalg.vector_norm(
        reference_delta[:, -1], dim=-1, keepdim=True
    ).clamp_min(1e-6)
    final_dot = (final_predicted_direction * final_reference_direction).sum(dim=-1)
    final_cross = (
        final_predicted_direction[:, 0] * final_reference_direction[:, 1]
        - final_predicted_direction[:, 1] * final_reference_direction[:, 0]
    )
    return {
        "ade_m": point_error.mean(dim=-1),
        "rmse_m": difference.square().mean(dim=(-1, -2)).sqrt(),
        "arc_length_m": predicted_length,
        "arc_length_error_m": (predicted_length - reference_length).abs(),
        "reference_arc_length_m": reference_length,
        "goal_progress_m": goal_distance
        - torch.linalg.vector_norm(point_goal - path[:, -1], dim=-1),
        "reference_goal_progress_m": goal_distance
        - torch.linalg.vector_norm(point_goal - reference_path[:, -1], dim=-1),
        "point_goal_distance_m": goal_distance,
        "point_goal_is_behind": point_goal[:, 0] < 0,
        "max_abs_curvature_inv_m": curvature.abs().amax(dim=-1),
        "reference_max_abs_curvature_inv_m": reference_curvature.abs().amax(dim=-1),
        "total_abs_heading_change_rad": total_abs_heading_change,
        "reference_total_abs_heading_change_rad": (
            reference_total_abs_heading_change
        ),
        "terminal_heading_error_rad": torch.atan2(final_cross, final_dot).abs(),
        "has_tangent_reversal": (tangent_dot < 0).any(dim=-1),
    }


def observed_obstacle_metrics(
    path: Tensor,
    obstacle_points: Tensor,
    obstacle_valid: Tensor,
) -> dict[str, Tensor]:
    """Measure path clearance against body-height surfaces in current depth."""
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError("path must have shape [B,P,2]")
    if obstacle_points.ndim != 3 or obstacle_points.shape[-1] != 2:
        raise ValueError("obstacle_points must have shape [B,O,2]")
    if obstacle_valid.shape != obstacle_points.shape[:2]:
        raise ValueError("obstacle_valid must have shape [B,O]")
    if obstacle_valid.dtype != torch.bool or path.shape[0] != obstacle_points.shape[0]:
        raise ValueError("obstacle mask and batch dimensions must match")

    distance = torch.cdist(path.float(), obstacle_points.float())
    distance = distance.masked_fill(~obstacle_valid[:, None], torch.inf)
    nearest = distance.amin(dim=(-1, -2))
    has_obstacle = obstacle_valid.any(dim=-1)
    return {
        "observed_obstacle_available": has_obstacle,
        "observed_min_clearance_m": nearest,
        "observed_footprint_collision": has_obstacle
        & (nearest < ROBOT_FOOTPRINT_RADIUS_M),
        "observed_safety_margin_violation": has_obstacle
        & (nearest < SAFETY_CLEARANCE_M),
    }


def _sample(
    policy: CurveNavPolicy,
    batch: dict[str, Tensor],
):
    prepared = unpack_policy_batch(batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = policy.sample(prepared.condition)
    return prepared, prediction


def _time_online(
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
    online_latency_repeats: int = ONLINE_LATENCY_REPEATS,
) -> PolicyMeasurements:
    """Collect one deterministic prediction for every aligned observation."""
    policy.to(device).eval()
    warmup = next(iter(loader))
    _sample(policy, warmup)
    torch.cuda.synchronize(device)

    values: dict[str, list[Tensor]] = {}
    current_frame_values: dict[str, list[Tensor]] = {}
    batch_latency = []
    samples = 0
    for batch in loader:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prepared, prediction = _sample(policy, batch)
        end.record()
        end.synchronize()
        batch_latency.append(start.elapsed_time(end))
        current_frame_batch = dict(batch)
        current_frame_valid = torch.zeros_like(batch["observation_valid"])
        current_frame_valid[:, -1] = True
        current_frame_batch["observation_valid"] = current_frame_valid
        current_prepared, current_prediction = _sample(policy, current_frame_batch)
        goal_shuffle_batch = dict(batch)
        goal_shuffle_batch["point_goal"] = torch.roll(
            batch["point_goal"], shifts=1, dims=0
        )
        _, goal_shuffle_prediction = _sample(policy, goal_shuffle_batch)
        depth_shuffle_batch = dict(batch)
        shuffled_depth = batch["depth"].clone()
        shuffled_depth[:, -1] = torch.roll(batch["depth"][:, -1], shifts=1, dims=0)
        depth_shuffle_batch["depth"] = shuffled_depth
        _, depth_shuffle_prediction = _sample(policy, depth_shuffle_batch)
        reference_path, _, reference_curvature = policy.curve_codec.decode_values(
            prepared.target.curve_values.float(),
        )
        metrics = trajectory_batch_metrics(
            prediction.path.float(),
            prediction.curvature.float(),
            reference_path,
            reference_curvature,
            prepared.condition.point_goal.float(),
        )
        metrics["valid_observation_frames"] = (
            prepared.condition.observation_valid.sum(dim=-1)
        )
        projection = policy.depth_encoder.metric_projector(
            prepared.condition.depth,
            prepared.condition.observation_to_current.float(),
        )
        current_obstacle_valid = (
            projection.obstacle_valid[:, -1]
            & prepared.condition.observation_valid[:, -1, None]
        )
        metrics.update(
            observed_obstacle_metrics(
                prediction.path.float(),
                projection.obstacle_points[:, -1],
                current_obstacle_valid,
            )
        )
        goal_shuffle_error = torch.linalg.vector_norm(
            goal_shuffle_prediction.path.float() - reference_path,
            dim=-1,
        ).mean(dim=-1)
        depth_shuffle_error = torch.linalg.vector_norm(
            depth_shuffle_prediction.path.float() - reference_path,
            dim=-1,
        ).mean(dim=-1)
        metrics["point_goal_shuffle_ade_m"] = goal_shuffle_error
        metrics["point_goal_shuffle_path_change_m"] = torch.linalg.vector_norm(
            goal_shuffle_prediction.path.float() - prediction.path.float(),
            dim=-1,
        ).mean(dim=-1)
        metrics["current_depth_shuffle_ade_m"] = depth_shuffle_error
        metrics["current_depth_shuffle_path_change_m"] = torch.linalg.vector_norm(
            depth_shuffle_prediction.path.float() - prediction.path.float(),
            dim=-1,
        ).mean(dim=-1)
        current_metrics = trajectory_batch_metrics(
            current_prediction.path.float(),
            current_prediction.curvature.float(),
            reference_path,
            reference_curvature,
            current_prepared.condition.point_goal.float(),
        )
        current_metrics["valid_observation_frames"] = (
            current_prepared.condition.observation_valid.sum(dim=-1)
        )
        for name, value in metrics.items():
            values.setdefault(name, []).append(value.cpu())
        for name, value in current_metrics.items():
            current_frame_values.setdefault(name, []).append(value.cpu())
        samples += len(prediction.path)

    return PolicyMeasurements(
        metrics={name: torch.cat(parts) for name, parts in values.items()},
        current_frame_metrics={
            name: torch.cat(parts) for name, parts in current_frame_values.items()
        },
        batch_latency_ms=torch.tensor(batch_latency, dtype=torch.float64),
        online_latency_ms=(
            _time_online(
                policy,
                warmup,
                online_latency_repeats,
            )
            if online_latency_repeats
            else torch.empty(0, dtype=torch.float64)
        ),
        samples=samples,
    )


def summarize_policy_metrics(metrics: dict[str, Tensor]) -> dict[str, float | int]:
    reference_curvature = metrics["reference_max_abs_curvature_inv_m"]
    reference_turn = metrics["reference_total_abs_heading_change_rad"]
    high_turn_threshold = torch.quantile(reference_turn, 0.9)
    high_turn = reference_turn >= high_turn_threshold
    predicted_high_turn = metrics["total_abs_heading_change_rad"][high_turn]
    reference_high_turn = reference_turn[high_turn]
    result = {
        "ade_m": metrics["ade_m"].mean().item(),
        "rmse_m": metrics["rmse_m"].mean().item(),
        "arc_length_m": metrics["arc_length_m"].mean().item(),
        "arc_length_error_m": metrics["arc_length_error_m"].mean().item(),
        "reference_arc_length_m": metrics["reference_arc_length_m"].mean().item(),
        "goal_progress_m": metrics["goal_progress_m"].mean().item(),
        "reference_goal_progress_m": metrics["reference_goal_progress_m"].mean().item(),
        "negative_progress_fraction": (metrics["goal_progress_m"] < 0)
        .float()
        .mean()
        .item(),
        "max_abs_curvature_inv_m_p95": torch.quantile(
            metrics["max_abs_curvature_inv_m"], 0.95
        ).item(),
        "reference_max_abs_curvature_inv_m_p95": torch.quantile(
            metrics["reference_max_abs_curvature_inv_m"], 0.95
        ).item(),
        "terminal_heading_error_rad": metrics["terminal_heading_error_rad"]
        .mean()
        .item(),
        "total_abs_heading_change_rad": metrics["total_abs_heading_change_rad"]
        .mean()
        .item(),
        "reference_total_abs_heading_change_rad": reference_turn.mean().item(),
        "high_turn_threshold_rad": high_turn_threshold.item(),
        "high_turn_samples": int(high_turn.sum().item()),
        "ade_m_high_turn_10pct": metrics["ade_m"][high_turn].mean().item(),
        "terminal_heading_error_rad_high_turn_10pct": metrics[
            "terminal_heading_error_rad"
        ][high_turn]
        .mean()
        .item(),
        "predicted_total_abs_heading_change_rad_high_turn_10pct": (
            predicted_high_turn.mean().item()
        ),
        "reference_total_abs_heading_change_rad_high_turn_10pct": (
            reference_high_turn.mean().item()
        ),
        "predicted_to_reference_turn_ratio_high_turn_10pct": (
            predicted_high_turn.mean() / reference_high_turn.mean().clamp_min(1e-6)
        ).item(),
        "max_abs_curvature_correlation": torch.corrcoef(
            torch.stack(
                (
                    metrics["max_abs_curvature_inv_m"],
                    reference_curvature,
                )
            )
        )[0, 1].item(),
        "tangent_reversal_fraction": metrics["has_tangent_reversal"]
        .float()
        .mean()
        .item(),
    }
    valid_frames = metrics["valid_observation_frames"]
    for frame_count in valid_frames.unique(sorted=True).tolist():
        selected = valid_frames == frame_count
        result[f"samples_with_{frame_count}_frames"] = int(selected.sum().item())
        result[f"ade_m_with_{frame_count}_frames"] = (
            metrics["ade_m"][selected].mean().item()
        )
    goal_distance = metrics["point_goal_distance_m"]
    goal_distance_bands = (
        ("lt_3m", goal_distance < 3.0),
        ("3_to_6m", (goal_distance >= 3.0) & (goal_distance < 6.0)),
        ("6_to_8_5m", (goal_distance >= 6.0) & (goal_distance < 8.5)),
        ("ge_8_5m", goal_distance >= 8.5),
    )
    for name, selected in goal_distance_bands:
        count = int(selected.sum().item())
        result[f"samples_point_goal_{name}"] = count
        if count:
            result[f"ade_m_point_goal_{name}"] = metrics["ade_m"][selected].mean().item()
    behind = metrics["point_goal_is_behind"]
    result["point_goal_behind_samples"] = int(behind.sum().item())
    if behind.any():
        result["ade_m_point_goal_behind"] = metrics["ade_m"][behind].mean().item()
    if not all(torch.isfinite(torch.tensor(value)) for value in result.values()):
        raise FloatingPointError(f"validation metrics are non-finite: {result}")
    return result


def evaluate_policy(
    policy: CurveNavPolicy,
    loader,
    device: torch.device,
    expected_samples: int,
) -> dict[str, float | int | str]:
    measurements = measure_policy(policy, loader, device)
    if measurements.samples != expected_samples:
        raise RuntimeError(
            f"evaluated {measurements.samples} samples, expected {expected_samples}"
        )
    seconds = measurements.batch_latency_ms.sum().item() / 1000.0
    current_frame_summary = summarize_policy_metrics(
        measurements.current_frame_metrics
    )
    metrics = measurements.metrics
    observed = metrics["observed_obstacle_available"]
    observed_count = int(observed.sum().item())
    if observed_count == 0:
        raise RuntimeError("validation contains no current-depth body obstacles")
    return {
        "protocol": "curvenav_local_validation",
        "samples": measurements.samples,
        "trajectories_per_observation": 1,
        **summarize_policy_metrics(measurements.metrics),
        "current_frame_only_ade_m": current_frame_summary["ade_m"],
        "current_frame_only_ade_m_high_turn_10pct": (
            current_frame_summary["ade_m_high_turn_10pct"]
        ),
        "current_frame_only_terminal_heading_error_rad_high_turn_10pct": (
            current_frame_summary[
                "terminal_heading_error_rad_high_turn_10pct"
            ]
        ),
        "current_frame_only_turn_ratio_high_turn_10pct": (
            current_frame_summary[
                "predicted_to_reference_turn_ratio_high_turn_10pct"
            ]
        ),
        "current_frame_only_negative_progress_fraction": current_frame_summary[
            "negative_progress_fraction"
        ],
        "point_goal_shuffle_ade_m": metrics["point_goal_shuffle_ade_m"].mean().item(),
        "point_goal_shuffle_ade_increase_m": (
            metrics["point_goal_shuffle_ade_m"] - metrics["ade_m"]
        ).mean().item(),
        "point_goal_shuffle_path_change_m": metrics[
            "point_goal_shuffle_path_change_m"
        ].mean().item(),
        "current_depth_shuffle_ade_m": metrics[
            "current_depth_shuffle_ade_m"
        ].mean().item(),
        "current_depth_shuffle_ade_increase_m": (
            metrics["current_depth_shuffle_ade_m"] - metrics["ade_m"]
        ).mean().item(),
        "current_depth_shuffle_path_change_m": metrics[
            "current_depth_shuffle_path_change_m"
        ].mean().item(),
        "observed_obstacle_samples": observed_count,
        "observed_min_clearance_m_mean": metrics["observed_min_clearance_m"][
            observed
        ].mean().item(),
        "observed_footprint_collision_fraction": metrics[
            "observed_footprint_collision"
        ][observed].float().mean().item(),
        "observed_safety_margin_violation_fraction": metrics[
            "observed_safety_margin_violation"
        ][observed].float().mean().item(),
        "batch32_latency_ms_mean": measurements.batch_latency_ms.mean().item(),
        "throughput_observations_per_second": measurements.samples / seconds,
        "online_latency_ms_p50": torch.quantile(
            measurements.online_latency_ms, 0.50
        ).item(),
        "online_latency_ms_p95": torch.quantile(
            measurements.online_latency_ms, 0.95
        ).item(),
    }


def run_evaluation(config: CurveNavConfig, checkpoint_path: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CurveNav evaluation requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_policy_contract(checkpoint, config)
    policy = build_policy(config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy)
    bundle = build_policy_validation_loader(
        config.data,
        config.trajectory,
        batch_size=VALIDATION_BATCH_SIZE,
        num_workers=config.training.num_workers,
    )
    loader = CudaPrefetchLoader(bundle.loader, bundle.depth_bank, torch.device("cuda"))
    result = evaluate_policy(
        policy,
        loader,
        torch.device("cuda"),
        bundle.samples,
    )
    result.update(
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_step=int(checkpoint["step"]),
        weights="ema",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    run_evaluation(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
