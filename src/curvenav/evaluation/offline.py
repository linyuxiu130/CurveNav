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
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.trajectory import path_arc_length


VALIDATION_BATCH_SIZE = 32
ONLINE_LATENCY_REPEATS = 50


@dataclass(frozen=True)
class PolicyMeasurements:
    metrics: dict[str, Tensor]
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
        "max_abs_curvature_inv_m": curvature.abs().amax(dim=-1),
        "reference_max_abs_curvature_inv_m": reference_curvature.abs().amax(dim=-1),
        "terminal_heading_error_rad": torch.atan2(final_cross, final_dot).abs(),
        "has_tangent_reversal": (tangent_dot < 0).any(dim=-1),
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
        smoothed_reference_path = policy.target_codec.decode_equal_arc(
            prepared.target.control_points.float()
        )
        _, _, reference_curvature = policy.curve_codec.path_geometry(
            smoothed_reference_path
        )
        metrics = trajectory_batch_metrics(
            prediction.path.float(),
            prediction.curvature.float(),
            prepared.target.reference_path.float(),
            reference_curvature,
            prepared.condition.point_goal.float(),
        )
        metrics["valid_observation_frames"] = (
            prepared.condition.observation_valid.sum(dim=-1)
        )
        for name, value in metrics.items():
            values.setdefault(name, []).append(value.cpu())
        samples += len(prediction.path)

    return PolicyMeasurements(
        metrics={name: torch.cat(parts) for name, parts in values.items()},
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
    high_curvature_threshold = torch.quantile(reference_curvature, 0.9)
    high_curvature = reference_curvature >= high_curvature_threshold
    predicted_high_curvature = metrics["max_abs_curvature_inv_m"][high_curvature]
    reference_high_curvature = reference_curvature[high_curvature]
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
        "high_curvature_threshold_inv_m": high_curvature_threshold.item(),
        "high_curvature_samples": int(high_curvature.sum().item()),
        "ade_m_high_curvature_10pct": metrics["ade_m"][high_curvature].mean().item(),
        "terminal_heading_error_rad_high_curvature_10pct": metrics[
            "terminal_heading_error_rad"
        ][high_curvature]
        .mean()
        .item(),
        "predicted_max_abs_curvature_inv_m_high_curvature_10pct": (
            predicted_high_curvature.mean().item()
        ),
        "reference_max_abs_curvature_inv_m_high_curvature_10pct": (
            reference_high_curvature.mean().item()
        ),
        "predicted_to_reference_curvature_ratio_high_curvature_10pct": (
            predicted_high_curvature.mean() / reference_high_curvature.mean()
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
    return {
        "protocol": "curvenav_local_validation",
        "samples": measurements.samples,
        "trajectories_per_observation": 1,
        **summarize_policy_metrics(measurements.metrics),
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
