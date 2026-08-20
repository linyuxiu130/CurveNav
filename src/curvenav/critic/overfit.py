"""Fixed-state memorization gate for the standalone critic chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from curvenav.critic.config import CriticExperimentConfig, load_critic_config
from curvenav.critic.data import CriticHssdDataset, load_critic_sidecar_table
from curvenav.critic.loss import critic_loss
from curvenav.critic.model import TrajectoryCritic
from curvenav.critic.runtime import load_frozen_policy, policy_condition
from curvenav.data.hssd import CurveNavHssdV2Dataset
from curvenav.training.optimizer import build_optimizer
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.training.runtime import configure_cuda_training_backend


def run_critic_overfit(config: CriticExperimentConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("critic overfit requires CUDA")
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    configure_cuda_training_backend()
    device = torch.device("cuda")
    policy_bundle = load_frozen_policy(config)
    codec = policy_bundle.policy.codec
    table = load_critic_sidecar_table(config.paths.sidecar_root, codec)
    conditions = CurveNavHssdV2Dataset(
        config.paths.dataset_root,
        policy_bundle.config.data,
        split="train",
    )
    dataset = CriticHssdDataset(conditions, table)
    subset = Subset(dataset, range(config.training.overfit_batch_size))
    loader = DataLoader(
        subset,
        batch_size=config.training.overfit_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(CudaPrefetchLoader(loader, conditions.depth_bank, device)))
    policy = policy_bundle.policy.to(device)
    critic = TrajectoryCritic(
        num_control_points=codec.num_control_points,
        scale_xy=policy_bundle.config.trajectory.scale_xy,
        model_dim=policy_bundle.config.condition_encoder.model_dim,
    ).to(device)
    optimizer = build_optimizer(
        critic,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        condition_memory = policy.encode_condition(policy_condition(batch)).tokens

    def forward_loss() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = critic(batch["controls_m"], condition_memory)
            return critic_loss(
                prediction,
                preference=batch["preference"],
                candidate_valid=batch["candidate_valid"],
                collision=batch["collision"],
                margin_violation=batch["margin_violation"],
                progress_m=batch["progress_m"],
                progress_valid=batch["progress_valid"],
                clearance_m=batch["clearance_m"],
                auxiliary_weight=config.training.auxiliary_weight,
            ).loss

    critic.eval()
    with torch.no_grad():
        initial = float(forward_loss())
    critic.train()
    for step in range(1, config.training.overfit_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.training.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
    critic.eval()
    with torch.no_grad():
        final = float(forward_loss())
    ratio = final / initial
    result = {"initial_loss": initial, "final_loss": final, "loss_ratio": ratio}
    print(json.dumps(result), flush=True)
    if ratio > config.training.overfit_max_loss_ratio:
        raise RuntimeError(f"critic overfit gate failed: {result}")
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = config.paths.output_dir / ".overfit.tmp.pt"
    final_path = config.paths.output_dir / "overfit.pt"
    torch.save({"critic": critic.state_dict(), **result}, temporary)
    temporary.replace(final_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit CurveNav trajectory critic")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_critic_overfit(load_critic_config(args.config))


if __name__ == "__main__":
    main()
