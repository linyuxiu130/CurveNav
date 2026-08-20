"""Fixed-batch overfit gate for the only CurveNav training chain."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data import (
    PreparedSandBatch,
    build_sand_overfit_loader,
    unpack_prepared_sand_batch,
)
from curvenav.factory import build_policy
from curvenav.models import CurveNavPolicy
from curvenav.training.checkpoint import checkpoint_state
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.optimizer import build_optimizer
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.training.runtime import configure_cuda_training_backend
from curvenav.trajectory import path_scale_summary


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _mean_flow_loss(
    policy: CurveNavPolicy,
    prepared: PreparedSandBatch,
    base_seed: int,
) -> Tensor:
    """Evaluate the same batch over a fixed eight-draw flow distribution."""
    was_training = policy.training
    policy.eval()
    losses = []
    for draw in range(8):
        _seed_everything(base_seed + draw)
        with torch.autocast(
            device_type=prepared.condition.depth.device.type,
            dtype=torch.float16,
            enabled=prepared.condition.depth.is_cuda,
        ):
            losses.append(policy.training_loss(prepared.condition, prepared.target).loss.float())
    policy.train(was_training)
    return torch.stack(losses).mean()


@torch.no_grad()
def _fixed_flow_loss(
    policy: CurveNavPolicy,
    prepared: PreparedSandBatch,
    seed: int,
) -> Tensor:
    """Evaluate one frozen source/time draw for a true memorization gate."""
    was_training = policy.training
    policy.eval()
    _seed_everything(seed)
    with torch.autocast(
        device_type=prepared.condition.depth.device.type,
        dtype=torch.float16,
        enabled=prepared.condition.depth.is_cuda,
    ):
        loss = policy(prepared.condition, prepared.target).float()
    policy.train(was_training)
    return loss


def run_overfit(config: CurveNavConfig) -> dict[str, float]:
    """Overfit one immutable SanD batch and save a diagnostic checkpoint."""
    config.validate()
    if config.training.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CurveNav overfit validation requires CUDA")
    device = torch.device("cuda")
    configure_cuda_training_backend()
    _seed_everything(config.training.seed)

    loader_bundle = build_sand_overfit_loader(
        config.data,
        config.trajectory,
        batch_size=config.training.overfit_batch_size,
        random_seed=config.training.seed,
    )
    loader = CudaPrefetchLoader(
        loader_bundle.loader,
        loader_bundle.depth_bank,
        device,
    )
    model_batch = next(iter(loader))

    policy = build_policy(config).to(device)
    policy.compile(mode="default")
    prepared = unpack_prepared_sand_batch(model_batch)
    optimizer = build_optimizer(
        policy,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    policy.train()

    evaluation_seed = config.training.seed + 1
    training_draw_seed = config.training.seed + 100
    initial_loss = _fixed_flow_loss(policy, prepared, training_draw_seed).item()
    initial_distribution_loss = _mean_flow_loss(policy, prepared, evaluation_seed).item()
    for step in range(1, config.training.overfit_steps + 1):
        _seed_everything(training_draw_seed)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss = policy(prepared.condition, prepared.target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), config.training.grad_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update()
        if step == 1 or step % config.training.log_every_steps == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss.detach().float().item(),
                        "grad_norm": grad_norm.detach().float().item(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final_loss = _fixed_flow_loss(policy, prepared, training_draw_seed).item()
    final_distribution_loss = _mean_flow_loss(policy, prepared, evaluation_seed).item()
    policy.eval()
    loss_ratio = final_loss / max(initial_loss, 1e-12)
    _seed_everything(evaluation_seed)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        prediction = policy.sample(prepared.condition)
    with torch.no_grad():
        reconstructed_target = policy.codec.decode(prepared.target.control_points)
        target_scale = path_scale_summary(reconstructed_target)
        prediction_scale = path_scale_summary(prediction.dense_path)
        fit_rmse = torch.sqrt((reconstructed_target - prepared.canonical_path).square().mean())
    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "initial_distribution_loss": initial_distribution_loss,
        "final_distribution_loss": final_distribution_loss,
        "target_fit_rmse_m": fit_rmse.item(),
        "target_mean_arc_length_m": target_scale["arc_length_m"].mean().item(),
        "target_max_arc_length_m": target_scale["arc_length_m"].max().item(),
        "prediction_mean_arc_length_m": prediction_scale["arc_length_m"].mean().item(),
        "prediction_max_arc_length_m": prediction_scale["arc_length_m"].max().item(),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"fixed-batch diagnostics contain non-finite values: {metrics}")
    if loss_ratio > config.training.overfit_max_loss_ratio:
        raise RuntimeError(
            "fixed-batch loss did not reach the required ratio "
            f"<= {config.training.overfit_max_loss_ratio}: {metrics}"
        )
    state = checkpoint_state(
        policy,
        config,
        step=config.training.overfit_steps,
        optimizer=optimizer,
        extra=metrics,
    )
    state["ema"] = ema.state_dict()
    checkpoint_path = Path(config.training.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    print(json.dumps({"checkpoint": str(checkpoint_path), **metrics}, ensure_ascii=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit the single CurveNav training batch")
    parser.add_argument("config", type=Path, help="CurveNav YAML configuration")
    args = parser.parse_args()
    run_overfit(load_config(args.config))


if __name__ == "__main__":
    main()
