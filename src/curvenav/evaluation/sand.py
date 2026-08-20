"""Run-disjoint CurveNav evaluation with one immutable sampling protocol."""

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data import build_sand_validation_loader, unpack_prepared_sand_batch
from curvenav.deployment.runtime import DepthSafetySelector
from curvenav.factory import build_policy
from curvenav.models import CurveNavPolicy
from curvenav.training import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.trajectory import path_arc_length


VALIDATION_SAMPLES = 1024
VALIDATION_BATCH_SIZE = 32
VALIDATION_CANDIDATES = 8
ONLINE_LATENCY_REPEATS = 50


def _select_candidate(value: Tensor, indices: Tensor) -> Tensor:
    return value.gather(1, indices[:, None]).squeeze(1)


def trajectory_batch_metrics(
    dense_path: Tensor,
    curvature: Tensor,
    target_path: Tensor,
    target_curvature: Tensor,
    selected_indices: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return per-observation metrics for ``[B, K, P, 2]`` candidates."""
    if dense_path.ndim != 4 or dense_path.shape[-1] != 2:
        raise ValueError("dense_path must have shape [B, K, P, 2]")
    batch_size, candidates, path_points, _ = dense_path.shape
    if candidates < 2:
        raise ValueError("evaluation requires at least two candidates")
    if curvature.shape != (batch_size, candidates, path_points):
        raise ValueError("curvature shape must match dense_path")
    if target_path.shape != (batch_size, path_points, 2):
        raise ValueError("target_path shape must match dense_path")
    if target_curvature.shape != (batch_size, path_points):
        raise ValueError("target_curvature shape must match target_path")
    difference = dense_path - target_path[:, None]
    point_error = torch.linalg.vector_norm(difference, dim=-1)
    ade = point_error.mean(dim=-1)
    rmse = difference.square().mean(dim=(-1, -2)).sqrt()
    best = ade.argmin(dim=1)

    flat_path = dense_path.reshape(batch_size * candidates, path_points, 2)
    predicted_length = path_arc_length(flat_path).reshape(batch_size, candidates)
    target_length = path_arc_length(target_path)
    best_length = _select_candidate(predicted_length, best)

    pair_index = torch.triu_indices(candidates, candidates, offset=1, device=dense_path.device)
    pair_difference = dense_path[:, pair_index[0]] - dense_path[:, pair_index[1]]
    diversity = torch.linalg.vector_norm(pair_difference, dim=-1).mean(dim=(-1, -2))
    endpoint_error = torch.linalg.vector_norm(
        dense_path[:, :, -1] - target_path[:, None, -1],
        dim=-1,
    )
    best_max_curvature = _select_candidate(curvature.abs().amax(dim=-1), best)

    metrics = {
        "single_ade_m": ade[:, 0],
        "single_rmse_m": rmse[:, 0],
        "oracle_ade_m": _select_candidate(ade, best),
        "oracle_rmse_m": _select_candidate(rmse, best),
        "oracle_arc_length_error_m": (best_length - target_length).abs(),
        "oracle_arc_length_m": best_length,
        "target_arc_length_m": target_length,
        "candidate_pairwise_ade_m": diversity,
        "target_endpoint_error_m": endpoint_error.amax(dim=1),
        "oracle_max_abs_curvature_inv_m": best_max_curvature,
        "target_max_abs_curvature_inv_m": target_curvature.abs().amax(dim=-1),
    }
    if selected_indices is not None:
        if selected_indices.shape != (batch_size,):
            raise ValueError("selected_indices must have shape [B]")
        if selected_indices.dtype != torch.int64:
            raise ValueError("selected_indices must be int64")
        if torch.any((selected_indices < 0) | (selected_indices >= candidates)):
            raise ValueError("selected_indices contains an invalid candidate index")
        selected_length = _select_candidate(predicted_length, selected_indices)
        metrics.update(
            {
                "selector_ade_m": _select_candidate(ade, selected_indices),
                "selector_rmse_m": _select_candidate(rmse, selected_indices),
                "selector_arc_length_error_m": (selected_length - target_length).abs(),
                "selector_arc_length_m": selected_length,
                "selector_max_abs_curvature_inv_m": _select_candidate(
                    curvature.abs().amax(dim=-1), selected_indices
                ),
            }
        )
    return metrics


def _sample(policy: CurveNavPolicy, batch: dict[str, Tensor]):
    prepared = unpack_prepared_sand_batch(batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = policy.sample(
            prepared.condition,
            num_samples=VALIDATION_CANDIDATES,
        )
    return prepared, prediction


def _first_observation(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value[:1] for name, value in batch.items()}


def _time_policy_batches(
    policy: CurveNavPolicy,
    batch: dict[str, Tensor],
    repeats: int,
    random_seed: int,
) -> Tensor:
    for offset in range(2):
        torch.manual_seed(random_seed + offset)
        _sample(policy, batch)
    torch.cuda.synchronize(batch["depth"].device)
    latency = []
    torch.manual_seed(random_seed)
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _sample(policy, batch)
        end.record()
        end.synchronize()
        latency.append(start.elapsed_time(end))
    return torch.tensor(latency, dtype=torch.float64)


@torch.no_grad()
def evaluate_policy(
    policy: CurveNavPolicy,
    loader,
    device: torch.device,
    random_seed: int,
    maximum_depth_m: float,
) -> dict[str, float | int | str]:
    """Evaluate one already-loaded policy under the fixed source protocol."""
    policy.to(device).eval()
    policy.compile(mode="default")

    warmup = next(iter(loader))
    for offset in range(2):
        torch.manual_seed(random_seed + offset)
        _sample(policy, warmup)
    torch.cuda.synchronize(device)
    torch.manual_seed(random_seed)

    values: dict[str, list[Tensor]] = {}
    batch_latencies_ms = []
    evaluated = 0
    selector = DepthSafetySelector(maximum_depth_m)
    for batch in loader:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prepared, prediction = _sample(policy, batch)
        end.record()
        end.synchronize()
        batch_latencies_ms.append(start.elapsed_time(end))

        batch_size = prepared.canonical_path.shape[0]
        dense_path = prediction.dense_path.reshape(
            batch_size,
            VALIDATION_CANDIDATES,
            -1,
            2,
        ).float()
        curvature = prediction.curvature.reshape(
            batch_size,
            VALIDATION_CANDIDATES,
            -1,
        ).float()
        _, _, target_curvature = policy.codec(prepared.target.control_points.float())
        depth_m = (
            prepared.condition.depth[:, -1]
            .permute(0, 2, 3, 1)
            .float()
            .mul(maximum_depth_m)
            .cpu()
            .numpy()
        )
        selector_indices, _ = selector.select_indices(
            dense_path.cpu().numpy(),
            depth_m,
            prepared.condition.task_goal.float().cpu().numpy(),
        )
        selected_indices = torch.from_numpy(selector_indices).to(
            device=dense_path.device,
            dtype=torch.int64,
        )
        metrics = trajectory_batch_metrics(
            dense_path,
            curvature,
            prepared.canonical_path.float(),
            target_curvature,
            selected_indices,
        )
        for name, value in metrics.items():
            values.setdefault(name, []).append(value.cpu())
        evaluated += batch_size

    if evaluated != VALIDATION_SAMPLES:
        raise RuntimeError(f"evaluated {evaluated} samples, expected {VALIDATION_SAMPLES}")
    merged = {name: torch.cat(parts) for name, parts in values.items()}
    latency = torch.tensor(batch_latencies_ms, dtype=torch.float64)
    online_latency = _time_policy_batches(
        policy,
        _first_observation(warmup),
        repeats=ONLINE_LATENCY_REPEATS,
        random_seed=random_seed,
    )
    total_seconds = latency.sum().item() / 1000.0
    result: dict[str, float | int | str] = {
        "protocol": "curvenav_source_weighted_run_disjoint_val_v3_free_endpoint",
        "samples": evaluated,
        "candidates": VALIDATION_CANDIDATES,
        "inference_steps": policy.rectified_flow.inference_steps,
        "single_ade_m": merged["single_ade_m"].mean().item(),
        "single_rmse_m": merged["single_rmse_m"].mean().item(),
        "oracle_best_of_8_ade_m": merged["oracle_ade_m"].mean().item(),
        "oracle_best_of_8_rmse_m": merged["oracle_rmse_m"].mean().item(),
        "oracle_arc_length_error_m": merged["oracle_arc_length_error_m"].mean().item(),
        "oracle_arc_length_m": merged["oracle_arc_length_m"].mean().item(),
        "selector_ade_m": merged["selector_ade_m"].mean().item(),
        "selector_rmse_m": merged["selector_rmse_m"].mean().item(),
        "selector_arc_length_error_m": merged["selector_arc_length_error_m"].mean().item(),
        "selector_arc_length_m": merged["selector_arc_length_m"].mean().item(),
        "selector_max_abs_curvature_inv_m_p95": torch.quantile(
            merged["selector_max_abs_curvature_inv_m"], 0.95
        ).item(),
        "target_arc_length_m": merged["target_arc_length_m"].mean().item(),
        "candidate_pairwise_ade_m": merged["candidate_pairwise_ade_m"].mean().item(),
        "target_endpoint_error_m_max": merged["target_endpoint_error_m"].amax().item(),
        "oracle_max_abs_curvature_inv_m_p95": torch.quantile(
            merged["oracle_max_abs_curvature_inv_m"], 0.95
        ).item(),
        "target_max_abs_curvature_inv_m_p95": torch.quantile(
            merged["target_max_abs_curvature_inv_m"], 0.95
        ).item(),
        "batched_policy_latency_ms_batch32_mean": latency.mean().item(),
        "batched_policy_throughput_observations_per_second": evaluated / total_seconds,
        "online_policy_latency_ms_batch1_p50": torch.quantile(
            online_latency, 0.50
        ).item(),
        "online_policy_latency_ms_batch1_p95": torch.quantile(
            online_latency, 0.95
        ).item(),
    }
    if not all(
        isinstance(value, (int, str)) or torch.isfinite(torch.tensor(value)).item()
        for value in result.values()
    ):
        raise FloatingPointError(f"validation metrics contain non-finite values: {result}")
    return result


def run_evaluation(config: CurveNavConfig, checkpoint_path: Path) -> dict[str, object]:
    """Load EMA weights and evaluate the configured validation sources."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("SanD policy evaluation requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_policy_contract(checkpoint, config)
    policy = build_policy(config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy)
    loader_bundle = build_sand_validation_loader(
        config.data,
        config.trajectory,
        batch_size=VALIDATION_BATCH_SIZE,
        samples=VALIDATION_SAMPLES,
        num_workers=config.training.num_workers,
        random_seed=config.training.seed,
    )
    loader = CudaPrefetchLoader(
        loader_bundle.loader,
        loader_bundle.depth_bank,
        torch.device("cuda"),
    )
    result = evaluate_policy(
        policy,
        loader,
        device=torch.device("cuda"),
        random_seed=config.training.seed,
        maximum_depth_m=config.data.max_depth_m,
    )
    result.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "depth_encoder_revision": checkpoint["policy_contract"][
                "depth_encoder_revision"
            ],
            "weights": "ema",
            "validation_sources": [
                {
                    "root": source.root,
                    "split": source.split,
                    "weight": source.weight,
                }
                for source in config.data.validation_sources
            ],
        }
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CurveNav on held-out runs")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    run_evaluation(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
