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
from curvenav.evaluation.metrics import (
    observed_safety_metrics,
    summarize_metrics,
    summarize_paired_safety,
    trajectory_metrics,
)
from curvenav.evaluation.protocol import summarize_strata
from curvenav.evaluation.report import write_case_report
from curvenav.factory import build_policy
from curvenav.models import CurveNavPolicy
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
):
    prepared = unpack_policy_batch(batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = policy.sample(prepared.condition)
    return prepared, prediction


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
    model_latency_repeats: int = MODEL_LATENCY_REPEATS,
) -> PolicyMeasurements:
    """Collect one deterministic prediction for every aligned observation."""
    policy.to(device).eval()
    warmup = next(iter(loader))
    _sample(policy, warmup)
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
        prepared, prediction = _sample(policy, batch)
        end.record()
        end.synchronize()
        batch_latency.append(start.elapsed_time(end))
        current_frame_batch = dict(batch)
        current_frame_valid = torch.zeros_like(batch["observation_valid"])
        current_frame_valid[:, -1] = True
        current_frame_batch["observation_valid"] = current_frame_valid
        current_prepared, current_prediction = _sample(policy, current_frame_batch)
        reference_path, _ = policy.curve_codec.decode_values(
            prepared.target.curve_values.float(),
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
        metrics.update(
            observed_safety_metrics(
                prediction.path.float(),
                projection.configuration_field,
                policy.planning_horizon_m,
            )
        )
        reference_obstacle_metrics = observed_safety_metrics(
            reference_path,
            projection.configuration_field,
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
        straight_obstacle_metrics = observed_safety_metrics(
            straight_path,
            projection.configuration_field,
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
            observed_safety_metrics(
                current_prediction.path.float(),
                projection.configuration_field,
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
            "configuration_field": projection.configuration_field[
                :, (0, 3, 4)
            ].to(dtype=torch.float16),
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
            _time_model(
                policy,
                warmup,
                model_latency_repeats,
            )
            if model_latency_repeats
            else torch.empty(0, dtype=torch.float64)
        ),
        samples=samples,
        sample_data={name: torch.cat(parts) for name, parts in sample_data.items()},
    )


def evaluate_measurements(
    measurements: PolicyMeasurements,
) -> dict[str, object]:
    """Summarize a single inference pass without evaluating the policy twice."""
    seconds = measurements.batch_latency_ms.sum().item() / 1000.0
    current_frame_summary = summarize_metrics(measurements.current_frame_metrics)
    metrics = measurements.metrics
    policy_summary = summarize_metrics(metrics)
    current_metrics = measurements.current_frame_metrics
    return {
        "protocol": "curvenav_metric_local_validation",
        "samples": measurements.samples,
        "trajectories_per_observation": 1,
        **policy_summary,
        **summarize_paired_safety(metrics),
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
            "observed_footprint_collision_fraction": current_metrics[
                "observed_footprint_collision"
            ].float().mean().item(),
            "observed_safety_margin_violation_fraction": current_metrics[
                "observed_safety_margin_violation"
            ].float().mean().item(),
        },
        "batch32_latency_ms_mean": measurements.batch_latency_ms.mean().item(),
        "throughput_observations_per_second": measurements.samples / seconds,
        "model_latency_batch1_ms_p50": torch.quantile(
            measurements.model_latency_ms, 0.50
        ).item(),
        "model_latency_batch1_ms_p95": torch.quantile(
            measurements.model_latency_ms, 0.95
        ).item(),
        "strata": summarize_strata(metrics),
    }


def evaluate_policy(
    policy: CurveNavPolicy,
    loader,
    device: torch.device,
    expected_samples: int,
) -> dict[str, object]:
    measurements = measure_policy(policy, loader, device)
    if measurements.samples != expected_samples:
        raise RuntimeError(
            f"evaluated {measurements.samples} samples, expected {expected_samples}"
        )
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
    measurements = measure_policy(policy, loader, torch.device("cuda"))
    if measurements.samples != bundle.samples:
        raise RuntimeError(
            f"evaluated {measurements.samples} samples, expected {bundle.samples}"
        )
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
