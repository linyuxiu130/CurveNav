"""Two-GPU training for the frozen-policy CurveNav trajectory critic."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
from pathlib import Path
import time

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch import Tensor

from curvenav.critic.config import CriticExperimentConfig, load_critic_config
from curvenav.critic.data import (
    build_critic_condition_loader,
    build_critic_label_loader,
    load_critic_sidecar_table,
)
from curvenav.critic.evaluation import CriticMetricCounts, critic_metric_counts
from curvenav.critic.loss import CriticLoss, critic_loss
from curvenav.critic.model import CriticPrediction, TrajectoryCritic
from curvenav.critic.runtime import (
    FrozenPolicyBundle,
    critic_checkpoint,
    load_frozen_policy,
    policy_condition,
    sha256_file,
)
from curvenav.data.depth_bank import PackedDepthBankSpec
from curvenav.data.hssd import CurveNavHssdV2Dataset
from curvenav.training.optimizer import build_cosine_schedule, build_optimizer
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.training.runtime import configure_cuda_training_backend


_LOSS_NAMES = (
    "loss",
    "pairwise",
    "collision",
    "margin",
    "progress",
    "clearance",
)


def _loss_values(loss: CriticLoss) -> Tensor:
    return torch.stack(
        tuple(getattr(loss, name).detach().float() for name in _LOSS_NAMES)
    )


@torch.no_grad()
def _cache_condition_memory(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    loader: Iterable[dict[str, Tensor]],
    depth_bank: PackedDepthBankSpec,
    expected_sample_indices: tuple[int, ...],
    state_count: int,
    table: Tensor | None,
) -> Tensor:
    """Encode each immutable observation once and replicate the compact token table."""
    start = time.perf_counter()
    seen = torch.zeros(state_count, dtype=torch.bool, device=accelerator.device)
    prefetched = CudaPrefetchLoader(loader, depth_bank, accelerator.device)
    for batch in prefetched:
        with accelerator.autocast():
            memory = policy.encode_condition(policy_condition(batch)).tokens
        sample_index, memory = accelerator.gather((batch["sample_index"], memory))
        if table is None:
            table = torch.zeros(
                (state_count, memory.shape[1], memory.shape[2]),
                dtype=torch.float16,
                device=accelerator.device,
            )
        elif table.shape[1:] != memory.shape[1:]:
            raise RuntimeError("condition-memory shape changed between HSSD splits")
        table[sample_index] = memory.to(dtype=table.dtype)
        seen[sample_index] = True

    expected = torch.tensor(
        expected_sample_indices,
        dtype=torch.int64,
        device=accelerator.device,
    )
    missing = expected[~seen[expected]]
    if missing.numel():
        raise RuntimeError(
            f"condition-memory cache is missing sample indices: {missing[:8].tolist()}"
        )
    if table is None:
        raise RuntimeError("condition-memory loader produced no batches")
    accelerator.print(
        json.dumps(
            {
                "event": "condition_cache_complete",
                "states": len(expected_sample_indices),
                "seconds": time.perf_counter() - start,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return table


def _gather_validation_tensors(
    accelerator: Accelerator,
    prediction: CriticPrediction,
    batch: dict[str, Tensor],
) -> tuple[CriticPrediction, dict[str, Tensor]]:
    names = (
        "preference",
        "candidate_valid",
        "collision",
        "margin_violation",
        "progress_m",
        "progress_valid",
        "clearance_m",
    )
    gathered = accelerator.gather_for_metrics(
        (
            prediction.score,
            prediction.collision_logit,
            prediction.margin_logit,
            prediction.progress,
            prediction.clearance,
            *(batch[name] for name in names),
        )
    )
    gathered_prediction = CriticPrediction(*gathered[:5])
    gathered_batch = dict(zip(names, gathered[5:], strict=True))
    return gathered_prediction, gathered_batch


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    critic: torch.nn.Module,
    loader: Iterable[dict[str, Tensor]],
    condition_memory: Tensor,
    auxiliary_weight: float,
) -> dict[str, float]:
    critic.eval()
    loss_sums = torch.zeros(len(_LOSS_NAMES), device=accelerator.device)
    states = torch.zeros((), device=accelerator.device)
    metric_sums = torch.zeros(10, dtype=torch.float64, device=accelerator.device)
    for batch in loader:
        with accelerator.autocast():
            prediction = critic(
                batch["controls_m"], condition_memory[batch["sample_index"]]
            )
        prediction, gathered_batch = _gather_validation_tensors(
            accelerator, prediction, batch
        )
        loss = critic_loss(
            prediction,
            **gathered_batch,
            auxiliary_weight=auxiliary_weight,
        )
        batch_states = prediction.score.shape[0]
        loss_sums += _loss_values(loss) * batch_states
        states += batch_states
        metric_sums += critic_metric_counts(
            prediction,
            preference=gathered_batch["preference"],
            collision=gathered_batch["collision"],
            margin_violation=gathered_batch["margin_violation"],
        ).stacked()

    summary = CriticMetricCounts.from_stacked(metric_sums).summary()
    mean_losses = loss_sums / states.clamp_min(1)
    summary.update(
        {
            f"validation_{name}": float(value.item())
            for name, value in zip(_LOSS_NAMES, mean_losses, strict=True)
        }
    )
    return summary


def _save_best_checkpoint(
    accelerator: Accelerator,
    critic: torch.nn.Module,
    config: CriticExperimentConfig,
    policy_bundle: FrozenPolicyBundle,
    sidecar_schema_sha256: str,
    epoch: int,
    step: int,
    validation: dict[str, float],
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        config.paths.output_dir.mkdir(parents=True, exist_ok=True)
        state = critic_checkpoint(
            critic=accelerator.unwrap_model(critic),
            config=config,
            policy_bundle=policy_bundle,
            sidecar_schema_sha256=sidecar_schema_sha256,
            epoch=epoch,
            step=step,
            validation=validation,
        )
        temporary = config.paths.output_dir / ".checkpoint.tmp.pt"
        checkpoint = config.paths.output_dir / "checkpoint.pt"
        accelerator.save(state, temporary)
        temporary.replace(checkpoint)
    accelerator.wait_for_everyone()


def run_critic_training(config: CriticExperimentConfig) -> None:
    """Train only the critic against scene-isolated HSSD sidecar supervision."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("critic training requires CUDA")
    ddp = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    accelerator = Accelerator(
        mixed_precision="fp16",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp],
    )
    if accelerator.num_processes < 2:
        raise RuntimeError("critic production training requires at least two GPUs")
    configure_cuda_training_backend()
    set_seed(config.training.seed)

    policy_bundle = load_frozen_policy(config)
    policy = policy_bundle.policy.to(accelerator.device)
    codec = policy.codec
    sidecar = load_critic_sidecar_table(config.paths.sidecar_root, codec)
    train_conditions = CurveNavHssdV2Dataset(
        config.paths.dataset_root,
        policy_bundle.config.data,
        split="train",
    )
    validation_conditions = CurveNavHssdV2Dataset(
        config.paths.dataset_root,
        policy_bundle.config.data,
        split="validation",
    )
    train_condition_bundle = build_critic_condition_loader(
        train_conditions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    validation_condition_bundle = build_critic_condition_loader(
        validation_conditions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    train_loader = build_critic_label_loader(
        train_conditions,
        sidecar,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
        seed=config.training.seed,
    )
    validation_loader = build_critic_label_loader(
        validation_conditions,
        sidecar,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.training.seed,
    )

    critic = TrajectoryCritic(
        num_control_points=codec.num_control_points,
        scale_xy=policy_bundle.config.trajectory.scale_xy,
        model_dim=policy_bundle.config.condition_encoder.model_dim,
    )
    optimizer = build_optimizer(
        critic,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    global_batch_size = config.training.batch_size * accelerator.num_processes
    steps_per_epoch = math.ceil(len(train_conditions) / global_batch_size)
    total_steps = config.training.epochs * steps_per_epoch
    scheduler = build_cosine_schedule(
        optimizer,
        total_steps=total_steps,
        warmup_steps=config.training.warmup_epochs * steps_per_epoch,
        minimum_factor=config.training.minimum_learning_rate_factor,
    )
    (
        critic,
        optimizer,
        train_condition_loader,
        validation_condition_loader,
        train_loader,
        validation_loader,
        scheduler,
    ) = accelerator.prepare(
        critic,
        optimizer,
        train_condition_bundle.loader,
        validation_condition_bundle.loader,
        train_loader,
        validation_loader,
        scheduler,
        device_placement=[True, True, False, False, True, True, True],
    )
    if len(train_loader) != steps_per_epoch:
        raise RuntimeError(
            "unexpected distributed critic loader length: "
            f"{len(train_loader)} != {steps_per_epoch}"
        )

    accelerator.print(
        json.dumps(
            {
                "event": "critic_training_start",
                "world_size": accelerator.num_processes,
                "train_states": len(train_conditions),
                "validation_states": len(validation_conditions),
                "global_batch_size": global_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "epochs": config.training.epochs,
                "total_steps": total_steps,
                "precision": accelerator.mixed_precision,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    condition_memory: Tensor | None = None
    condition_memory = _cache_condition_memory(
        accelerator,
        policy,
        train_condition_loader,
        train_condition_bundle.depth_bank,
        train_conditions.sample_indices,
        len(sidecar),
        condition_memory,
    )
    condition_memory = _cache_condition_memory(
        accelerator,
        policy,
        validation_condition_loader,
        validation_condition_bundle.depth_bank,
        validation_conditions.sample_indices,
        len(sidecar),
        condition_memory,
    )
    policy.to("cpu")
    del train_condition_loader, validation_condition_loader, policy
    torch.cuda.empty_cache()

    sidecar_schema_sha256 = sha256_file(
        config.paths.sidecar_root / "critic_sidecar_schema_v1.json"
    )
    best_policy_pair_accuracy = float("-inf")
    step = 0
    for epoch in range(1, config.training.epochs + 1):
        critic.train()
        window_loss = torch.zeros(len(_LOSS_NAMES), device=accelerator.device)
        window_steps = 0
        window_start = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                prediction = critic(
                    batch["controls_m"], condition_memory[batch["sample_index"]]
                )
                loss = critic_loss(
                    prediction,
                    preference=batch["preference"],
                    candidate_valid=batch["candidate_valid"],
                    collision=batch["collision"],
                    margin_violation=batch["margin_violation"],
                    progress_m=batch["progress_m"],
                    progress_valid=batch["progress_valid"],
                    clearance_m=batch["clearance_m"],
                    auxiliary_weight=config.training.auxiliary_weight,
                )
            accelerator.backward(loss.loss)
            accelerator.clip_grad_norm_(
                critic.parameters(), config.training.grad_clip_norm
            )
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()
            step += 1
            window_steps += 1
            window_loss += _loss_values(loss)
            if step == 1 or step % config.training.log_every_steps == 0:
                elapsed = time.perf_counter() - window_start
                mean_loss = accelerator.reduce(
                    window_loss / window_steps, reduction="mean"
                )
                accelerator.print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            **{
                                name: float(value.item())
                                for name, value in zip(
                                    _LOSS_NAMES, mean_loss, strict=True
                                )
                            },
                            "learning_rate": scheduler.get_last_lr()[0],
                            "samples_per_second": (
                                global_batch_size * window_steps / elapsed
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                window_loss.zero_()
                window_steps = 0
                window_start = time.perf_counter()

        validation = _evaluate(
            accelerator,
            critic,
            validation_loader,
            condition_memory,
            config.training.auxiliary_weight,
        )
        accelerator.print(
            json.dumps(
                {"event": "critic_validation", "epoch": epoch, **validation},
                ensure_ascii=False,
            ),
            flush=True,
        )
        policy_pair_accuracy = validation["policy_pair_accuracy"]
        if policy_pair_accuracy > best_policy_pair_accuracy:
            best_policy_pair_accuracy = policy_pair_accuracy
            _save_best_checkpoint(
                accelerator,
                critic,
                config,
                policy_bundle,
                sidecar_schema_sha256,
                epoch,
                step,
                validation,
            )

    accelerator.print(
        json.dumps(
            {
                "event": "critic_training_complete",
                "step": step,
                "best_policy_pair_accuracy": best_policy_pair_accuracy,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CurveNav trajectory critic")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_critic_training(load_critic_config(args.config))


if __name__ == "__main__":
    main()
