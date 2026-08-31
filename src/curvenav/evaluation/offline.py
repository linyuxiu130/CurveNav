"""Held-out source-truth evaluation for CurveNav's single local policy."""

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
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.evaluation.metrics import (
    configuration_space_collision_attribution,
    configuration_space_safety_metrics,
    summarize_configuration_safety,
    summarize_metrics,
    trajectory_metrics,
)
from curvenav.evaluation.protocol import (
    summarize_goal_bearing_strata,
    summarize_strata,
)
from curvenav.evaluation.report import write_case_report
from curvenav.factory import build_policy
from curvenav.models import CurveNavPolicy
from curvenav.precision import CudaPrecision, cuda_precision
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.prefetch import CudaPrefetchLoader


VALIDATION_BATCH_SIZE = 32
MODEL_LATENCY_REPEATS = 50


@dataclass(frozen=True)
class PolicyMeasurements:
    metrics: dict[str, Tensor]
    current_frame_metrics: dict[str, Tensor]
    batch_latency_ms: Tensor
    model_latency_ms: Tensor
    samples: int
    sample_data: dict[str, Tensor]


def _sample(
    policy: CurveNavPolicy,
    batch: dict[str, Tensor],
    precision: CudaPrecision,
):
    prepared = unpack_policy_batch(batch)
    with torch.autocast(device_type="cuda", dtype=precision.autocast_dtype):
        prediction = policy.sample(prepared.condition)
    return prepared, prediction


def _collision_detection_summary(metrics: dict[str, Tensor]) -> dict[str, int | float]:
    """Report raw-depth observability of source-truth collision points."""
    truth_points = metrics["truth_collision_point_count"].sum()
    truth_trajectories = metrics["truth_collision_trajectory"].sum()

    def aggregate(name: str) -> int:
        return int(metrics[name].sum().item())

    def fraction(numerator: int, denominator: Tensor) -> float:
        return float(numerator / max(int(denominator.item()), 1))

    raw_points = aggregate("truth_collision_point_raw_depth_count")
    raw_ray_coverage_points = aggregate(
        "truth_collision_point_raw_ray_coverage_count"
    )
    raw_trajectories = aggregate(
        "truth_collision_trajectory_recognized_by_raw_depth"
    )
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
    precision: CudaPrecision,
) -> Tensor:
    first = {name: value[:1] for name, value in batch.items()}
    for _ in range(2):
        _sample(policy, first, precision)
    torch.cuda.synchronize()
    latency = []
    for _ in range(repeats):
        started = time.perf_counter()
        _sample(policy, first, precision)
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - started) * 1000.0)
    return torch.tensor(latency, dtype=torch.float64)


@torch.no_grad()
def measure_policy(
    policy: CurveNavPolicy,
    loader,
    device: torch.device,
    source_query: SourceConfigurationSpaceQuery,
    model_latency_repeats: int = MODEL_LATENCY_REPEATS,
) -> PolicyMeasurements:
    """Collect one deterministic path and one history ablation per observation."""
    policy.to(device).eval()
    precision = cuda_precision(device)
    warmup = next(iter(loader))
    _sample(policy, warmup, precision)
    torch.cuda.synchronize(device)

    values: dict[str, list[Tensor]] = {}
    current_frame_values: dict[str, list[Tensor]] = {}
    sample_data: dict[str, list[Tensor]] = {}
    batch_latency = []
    samples = 0
    for batch in loader:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prepared, prediction = _sample(policy, batch, precision)
        end.record()
        end.synchronize()
        batch_latency.append(start.elapsed_time(end))

        current_frame_batch = dict(batch)
        current_frame_valid = torch.zeros_like(batch["observation_valid"])
        current_frame_valid[:, -1] = True
        current_frame_batch["observation_valid"] = current_frame_valid
        current_prepared, current_prediction = _sample(
            policy, current_frame_batch, precision
        )

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
        metrics["valid_observation_frames"] = (
            prepared.condition.observation_valid.sum(dim=-1)
        )
        projection = policy.depth_encoder.metric_projector(
            prepared.condition.depth,
            prepared.condition.observation_to_current.float(),
            prepared.condition.observation_valid,
        )
        depth_safety = configuration_space_safety_metrics(
            prediction.path.float(),
            projection.configuration_field,
            policy.planning_horizon_m,
        )
        metrics.update({f"depth_{name}": value for name, value in depth_safety.items()})
        metrics.update(
            source_query.safety_metrics_from_query(
                prediction.path.float(),
                source_prediction,
                policy.planning_horizon_m,
            )
        )
        metrics.update(
            configuration_space_collision_attribution(
                source_prediction,
                projection.configuration_field,
                policy.planning_horizon_m,
            )
        )
        reference_obstacle_metrics = source_query.safety_metrics(
            reference_path,
            source_grid_index,
            source_origin_xy,
            source_yaw_rad,
            policy.planning_horizon_m,
        )
        for name, value in reference_obstacle_metrics.items():
            metrics[f"reference_{name}"] = value
        straight_progress = torch.linspace(
            0.0,
            1.0,
            reference_path.shape[1],
            device=reference_path.device,
        )[None, :, None]
        straight_path = (
            reference_path[:, :1]
            + straight_progress * (reference_path[:, -1:] - reference_path[:, :1])
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
        current_metrics.update(
            source_query.safety_metrics(
                current_prediction.path.float(),
                source_grid_index,
                source_origin_xy,
                source_yaw_rad,
                policy.planning_horizon_m,
            )
        )
        for name, value in metrics.items():
            values.setdefault(name, []).append(value.cpu())
        for name, value in current_metrics.items():
            current_frame_values.setdefault(name, []).append(value.cpu())
        for name, value in {
            "predicted_path": prediction.path.float(),
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
        }.items():
            sample_data.setdefault(name, []).append(value.cpu())
        samples += len(prediction.path)

    return PolicyMeasurements(
        metrics={name: torch.cat(parts) for name, parts in values.items()},
        current_frame_metrics={
            name: torch.cat(parts) for name, parts in current_frame_values.items()
        },
        batch_latency_ms=torch.tensor(batch_latency, dtype=torch.float64),
        model_latency_ms=(
            _time_model(policy, warmup, model_latency_repeats, precision)
            if model_latency_repeats
            else torch.empty(0, dtype=torch.float64)
        ),
        samples=samples,
        sample_data={name: torch.cat(parts) for name, parts in sample_data.items()},
    )


def evaluate_measurements(measurements: PolicyMeasurements) -> dict[str, object]:
    """Summarize one inference pass without model-dependent safety surrogates."""
    seconds = measurements.batch_latency_ms.sum().item() / 1000.0
    current_frame_summary = summarize_metrics(measurements.current_frame_metrics)
    metrics = measurements.metrics
    policy_summary = summarize_metrics(metrics)
    current_metrics = measurements.current_frame_metrics
    return {
        "protocol": "curvenav_metric_local_validation_source_cspace",
        "samples": measurements.samples,
        "trajectories_per_observation": 1,
        **policy_summary,
        **summarize_configuration_safety(metrics),
        "current_frame_ablation": {
            "fixed_horizon_ade_m_mean": current_frame_summary[
                "fixed_horizon_ade_m_mean"
            ],
            "fixed_horizon_ade_increase_m": current_frame_summary[
                "fixed_horizon_ade_m_mean"
            ] - policy_summary["fixed_horizon_ade_m_mean"],
            "fixed_horizon_fde_m_mean": current_frame_summary[
                "fixed_horizon_fde_m_mean"
            ],
            "goal_progress_regret_m_mean": current_frame_summary[
                "goal_progress_regret_m_mean"
            ],
            "footprint_collision_fraction": current_metrics[
                "footprint_collision"
            ].float().mean().item(),
            "safety_margin_violation_fraction": current_metrics[
                "safety_margin_violation"
            ].float().mean().item(),
        },
        "raw_depth_collision_attribution": _collision_detection_summary(metrics),
        "batch32_latency_ms_mean": measurements.batch_latency_ms.mean().item(),
        "throughput_observations_per_second": measurements.samples / seconds,
        "model_latency_batch1_ms_p50": torch.quantile(
            measurements.model_latency_ms, 0.50
        ).item(),
        "model_latency_batch1_ms_p95": torch.quantile(
            measurements.model_latency_ms, 0.95
        ).item(),
        "strata": summarize_strata(metrics),
        "goal_bearing_strata": summarize_goal_bearing_strata(metrics),
    }


def evaluate_policy(
    policy: CurveNavPolicy,
    loader,
    device: torch.device,
    expected_samples: int,
    source_query: SourceConfigurationSpaceQuery,
) -> dict[str, object]:
    measurements = measure_policy(policy, loader, device, source_query)
    if measurements.samples != expected_samples:
        raise RuntimeError(
            f"evaluated {measurements.samples} samples, expected {expected_samples}"
        )
    _validate_reference_source_safety(measurements.metrics)
    return evaluate_measurements(measurements)


def run_evaluation(
    config: CurveNavConfig,
    checkpoint_path: Path,
    artifact_dir: Path | None = None,
) -> dict[str, object]:
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
    source_query = SourceConfigurationSpaceQuery.from_prepared_split(
        config.data.root,
        "validation",
    )
    measurements = measure_policy(
        policy,
        loader,
        torch.device("cuda"),
        source_query,
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
    )
    if artifact_dir is not None:
        artifact_dir = artifact_dir.expanduser().resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        case_path = write_case_report(
            artifact_dir,
            measurements.metrics,
            measurements.sample_data,
        )
        metrics_path = artifact_dir / "offline-metrics.json"
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result["case_report"] = str(case_path)
        result["metrics_report"] = str(metrics_path)
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
