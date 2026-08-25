"""Multi-GPU FP16 training for the single CurveNav policy route."""

import argparse
import json
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data import build_policy_training_loader, unpack_policy_batch
from curvenav.factory import build_policy
from curvenav.training.checkpoint import (
    checkpoint_state,
    restore_training_state,
    validate_policy_contract,
)
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.optimizer import build_cosine_schedule, build_optimizer
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.training.runtime import configure_cuda_training_backend


def _save_checkpoint(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    config: CurveNavConfig,
    epoch: int,
    step: int,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = checkpoint_state(
        accelerator.unwrap_model(policy),
        config,
        step=step,
        optimizer=optimizer,
        extra={
            "epoch": epoch,
            "world_size": accelerator.num_processes,
            "mixed_precision": accelerator.mixed_precision,
        },
    )
    state["scheduler"] = scheduler.state_dict()
    state["ema"] = ema.state_dict()
    if accelerator.scaler is not None:
        state["grad_scaler"] = accelerator.scaler.state_dict()
    checkpoint_path = output_dir / "checkpoint.pt"
    temporary_path = output_dir / ".checkpoint.tmp.pt"
    accelerator.save(state, temporary_path)
    temporary_path.replace(checkpoint_path)


def _advance_optimizer_state(
    accelerator: Accelerator,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
) -> bool:
    if accelerator.optimizer_step_was_skipped:
        return False
    scheduler.step()
    ema.update()
    return True


def run_training(
    config: CurveNavConfig,
    resume_path: Path | None = None,
    stop_after_steps: int | None = None,
) -> None:
    """Train with one process per GPU and one globally sharded policy loader."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CurveNav production training requires CUDA")

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
        raise RuntimeError("production training requires at least two GPU processes")

    configure_cuda_training_backend()
    set_seed(config.training.seed)
    global_batch_size = (
        config.training.per_device_batch_size * accelerator.num_processes
    )
    if config.training.samples_per_epoch % global_batch_size:
        raise ValueError("samples_per_epoch must be divisible by the global batch size")
    steps_per_epoch = config.training.samples_per_epoch // global_batch_size
    total_steps = config.training.epochs * steps_per_epoch
    checkpoint = None
    start_step = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_policy_contract(checkpoint, config)
        start_step = int(checkpoint["step"])
        if not 0 <= start_step < total_steps:
            raise ValueError(
                f"resume step must be in [0, {total_steps}), got {start_step}"
            )
    end_step = total_steps if stop_after_steps is None else stop_after_steps
    if not start_step < end_step <= total_steps:
        raise ValueError(
            f"stop_after_steps must be in ({start_step}, {total_steps}], got {end_step}"
        )
    remaining_steps = end_step - start_step
    loader_bundle = build_policy_training_loader(
        config.data,
        config.trajectory,
        batch_size=config.training.per_device_batch_size,
        samples_per_epoch=remaining_steps * global_batch_size,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
        seed=config.training.seed,
        sample_offset=start_step * global_batch_size,
    )
    loader = loader_bundle.loader
    policy = build_policy(config)
    policy.compile(mode="default")
    optimizer = build_optimizer(
        policy,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = build_cosine_schedule(
        optimizer,
        total_steps=total_steps,
        warmup_steps=config.training.warmup_epochs * steps_per_epoch,
        minimum_factor=config.training.min_learning_rate_factor,
    )
    policy, optimizer, loader, scheduler = accelerator.prepare(
        policy,
        optimizer,
        loader,
        scheduler,
        device_placement=[True, True, False, True],
    )
    loader = CudaPrefetchLoader(
        loader,
        loader_bundle.depth_bank,
        accelerator.device,
    )
    if len(loader) != remaining_steps:
        raise RuntimeError(
            f"unexpected distributed loader length: {len(loader)} != {remaining_steps}"
        )
    ema = ExponentialMovingAverage(
        accelerator.unwrap_model(policy), decay=config.training.ema_decay
    )
    if checkpoint is not None:
        start_step = restore_training_state(
            checkpoint,
            accelerator.unwrap_model(policy),
            optimizer,
            scheduler,
            ema,
            config,
            accelerator.scaler,
        )

    accelerator.print(
        json.dumps(
            {
                "event": "training_start",
                "world_size": accelerator.num_processes,
                "per_device_batch_size": config.training.per_device_batch_size,
                "global_batch_size": global_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "end_step": end_step,
                "resume_step": start_step,
                "precision": accelerator.mixed_precision,
            },
            ensure_ascii=False,
        )
    )

    policy.train()
    step = start_step
    window_losses = torch.zeros(5, device=accelerator.device)
    window_steps = 0
    window_start = time.perf_counter()
    checkpoint_interval = config.training.checkpoint_every_epochs * steps_per_epoch
    for batch in loader:
        prepared = unpack_policy_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            losses = policy(prepared.condition, prepared.target)
        accelerator.backward(losses.loss)
        accelerator.clip_grad_norm_(policy.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        _advance_optimizer_state(accelerator, scheduler, ema)

        step += 1
        epoch = (step - 1) // steps_per_epoch + 1
        window_steps += 1
        window_losses += torch.stack(
            (
                losses.loss,
                losses.flow_loss,
                losses.path_loss,
                losses.tangent_loss,
                losses.ranking_loss,
            )
        ).detach().float()
        if step == 1 or step % config.training.log_every_steps == 0:
            accelerator.wait_for_everyone()
            elapsed = time.perf_counter() - window_start
            mean_losses = accelerator.reduce(
                window_losses / window_steps, reduction="mean"
            ).tolist()
            accelerator.print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": mean_losses[0],
                        "flow_loss": mean_losses[1],
                        "path_loss": mean_losses[2],
                        "tangent_loss": mean_losses[3],
                        "ranking_loss": mean_losses[4],
                        "learning_rate": scheduler.get_last_lr()[0],
                        "samples_per_second": global_batch_size * window_steps / elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            window_losses.zero_()
            window_steps = 0
            window_start = time.perf_counter()

        if step % checkpoint_interval == 0:
            _save_checkpoint(
                accelerator,
                policy,
                optimizer,
                scheduler,
                ema,
                config,
                epoch=step // steps_per_epoch,
                step=step,
            )

    if step % checkpoint_interval != 0:
        _save_checkpoint(
            accelerator,
            policy,
            optimizer,
            scheduler,
            ema,
            config,
            epoch=(step - 1) // steps_per_epoch + 1,
            step=step,
        )
    accelerator.print(
        json.dumps(
            {
                "event": "training_complete",
                "step": step,
                "production_total_steps": total_steps,
            }
        )
    )
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CurveNav policy")
    parser.add_argument("config", type=Path, help="CurveNav YAML configuration")
    parser.add_argument("--resume", type=Path, help="complete checkpoint to continue")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="bounded diagnostic endpoint on the unchanged production schedule",
    )
    args = parser.parse_args()
    run_training(load_config(args.config), args.resume, args.stop_after_steps)


if __name__ == "__main__":
    main()
