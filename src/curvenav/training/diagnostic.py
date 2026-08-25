"""Fixed-batch fitting diagnostic for CurveNav's one policy graph."""

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
from curvenav.data.batch import PreparedPolicyBatch, unpack_policy_batch
from curvenav.data.loader import build_fixed_batch_loader
from curvenav.factory import build_policy
from curvenav.models import CurveNavPolicy
from curvenav.training.optimizer import build_optimizer
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.training.runtime import configure_cuda_training_backend
from curvenav.trajectory import path_scale_summary


FIXED_BATCH_SIZE = 8
FIXED_BATCH_STEPS = 1_500
FIXED_LOSS_DRAWS = 8


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _fixed_loss(
    policy: CurveNavPolicy,
    prepared: PreparedPolicyBatch,
    seed: int,
) -> Tensor:
    """Evaluate a fixed eight-draw mean for the stochastic Flow objective."""
    was_training = policy.training
    policy.eval()
    losses = []
    for draw in range(FIXED_LOSS_DRAWS):
        _seed_everything(seed + draw)
        with torch.autocast(
            device_type=prepared.condition.depth.device.type,
            dtype=torch.float16,
            enabled=prepared.condition.depth.is_cuda,
        ):
            losses.append(policy(prepared.condition, prepared.target).loss.float())
    policy.train(was_training)
    return torch.stack(losses).mean()


def run_fixed_batch_diagnostic(config: CurveNavConfig) -> dict[str, float]:
    """Fit one immutable prepared batch and report the resulting diagnostics."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CurveNav fixed-batch diagnostic requires CUDA")
    device = torch.device("cuda")
    configure_cuda_training_backend()
    _seed_everything(config.training.seed)

    loader_bundle = build_fixed_batch_loader(
        config.data,
        config.trajectory,
        batch_size=FIXED_BATCH_SIZE,
        seed=config.training.seed,
    )
    loader = CudaPrefetchLoader(
        loader_bundle.loader,
        loader_bundle.depth_bank,
        device,
    )
    model_batch = next(iter(loader))

    policy = build_policy(config).to(device)
    policy.compile(mode="default")
    prepared = unpack_policy_batch(model_batch)
    optimizer = build_optimizer(
        policy,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    policy.train()

    training_draw_seed = config.training.seed + 100
    initial_loss = _fixed_loss(policy, prepared, training_draw_seed).item()
    for step in range(1, FIXED_BATCH_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            losses = policy(prepared.condition, prepared.target)
        scaler.scale(losses.loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), config.training.grad_clip_norm
        )
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_skipped = scaler.get_scale() < scale_before_step
        if step == 1 or step % config.training.log_every_steps == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses.loss.detach().float().item(),
                        "flow_loss": losses.flow_loss.detach().float().item(),
                        "path_loss": losses.path_loss.detach().float().item(),
                        "tangent_loss": losses.tangent_loss.detach().float().item(),
                        "grad_norm": grad_norm.detach().float().item(),
                        "optimizer_step_skipped": optimizer_step_skipped,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final_loss = _fixed_loss(policy, prepared, training_draw_seed).item()
    policy.eval()
    loss_ratio = final_loss / initial_loss
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = policy.sample(prepared.condition)
    with torch.no_grad():
        reference_path = prepared.target.reference_path.float()
        reference_scale = path_scale_summary(reference_path)
        prediction_scale = path_scale_summary(prediction.path)
        candidate_error = torch.linalg.vector_norm(
            prediction.candidate_paths - reference_path[:, None], dim=-1
        ).mean(dim=-1)
        selected_ade = torch.linalg.vector_norm(
            prediction.path - reference_path, dim=-1
        ).mean()
        segment = prediction.path[:, 1:] - prediction.path[:, :-1]
        tangent_dot = (segment[:, 1:] * segment[:, :-1]).sum(dim=-1)
    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "reference_mean_arc_length_m": reference_scale["arc_length_m"].mean().item(),
        "reference_max_arc_length_m": reference_scale["arc_length_m"].max().item(),
        "prediction_mean_arc_length_m": prediction_scale["arc_length_m"].mean().item(),
        "prediction_max_arc_length_m": prediction_scale["arc_length_m"].max().item(),
        "oracle_ade_m": candidate_error.min(dim=1).values.mean().item(),
        "selected_ade_m": selected_ade.item(),
        "tangent_reversal_fraction": (tangent_dot < 0).any(dim=1).float().mean().item(),
        "max_abs_curvature_inv_m": prediction.curvature.abs().max().item(),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"fixed-batch diagnostics contain non-finite values: {metrics}")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CurveNav's fixed-batch fitting diagnostic"
    )
    parser.add_argument("config", type=Path, help="CurveNav YAML configuration")
    args = parser.parse_args()
    run_fixed_batch_diagnostic(load_config(args.config))


if __name__ == "__main__":
    main()
