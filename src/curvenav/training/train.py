"""Multi-GPU mixed-precision training for the single CurveNav policy route."""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data.batch import unpack_policy_batch
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.training.critic import CurveNavCriterion
from curvenav.data.loader import (
    build_policy_training_loader,
    build_policy_validation_loader,
)
from curvenav.factory import build_policy
from curvenav.models import TRAINING_LOSS_NAMES
from curvenav.precision import PRECISION_NAME
from curvenav.training.checkpoint import (
    build_training_contract,
    build_training_checkpoint,
    restore_process_rng_state,
    restore_training_state,
    validate_policy_contract,
    validate_training_resume,
)
from curvenav.training.batching import build_distributed_batch_layout
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.optimizer import build_cosine_schedule, build_optimizer
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.training.runtime import (
    compile_static_training_functions,
    configure_cuda_training_backend,
)


def _save_checkpoint(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    config: CurveNavConfig,
    step: int,
    best_validation_loss: float,
    training_contract: dict[str, int | str],
    filenames: tuple[str, ...],
) -> None:
    accelerator.wait_for_everyone()
    cpu_rng_states = accelerator.gather(
        torch.get_rng_state().to(accelerator.device).unsqueeze(0)
    ).cpu()
    cuda_rng_states = accelerator.gather(
        torch.cuda.get_rng_state(accelerator.device).to(accelerator.device).unsqueeze(0)
    ).cpu()
    if not accelerator.is_main_process:
        return
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_training_checkpoint(
        accelerator.unwrap_model(policy),
        optimizer,
        scheduler,
        ema,
        config,
        step,
        best_validation_loss,
        training_contract=training_contract,
        rng_states={"cpu": cpu_rng_states, "cuda": cuda_rng_states},
    )
    for filename in filenames:
        checkpoint_path = output_dir / filename
        temporary_path = output_dir / f".{filename}.tmp"
        accelerator.save(state, temporary_path)
        temporary_path.replace(checkpoint_path)


@torch.no_grad()
def _validate(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    ema: ExponentialMovingAverage,
    loader: CudaPrefetchLoader,
    criterion: CurveNavCriterion,
    seed: int,
) -> dict[str, float]:
    """Measure the EMA objective on each held-out sample without changing RNG."""
    model = accelerator.unwrap_model(policy)
    totals = torch.zeros(len(TRAINING_LOSS_NAMES) + 1, device=accelerator.device)
    devices = [accelerator.device.index]
    with torch.random.fork_rng(devices=devices), ema.average_parameters(model):
        torch.manual_seed(seed + accelerator.process_index)
        model.eval()
        for batch in loader:
            prepared = unpack_policy_batch(batch)
            source = torch.randn_like(prepared.target.curve_values)
            losses = criterion(model(prepared.condition, prepared.target, source), batch)
            count = len(prepared.target.curve_values)
            totals[:-1] += torch.stack(losses.logging_values()).float() * count
            totals[-1] += count
        model.train()
    totals = accelerator.reduce(totals, reduction="sum")
    means = (totals[:-1] / totals[-1]).tolist()
    return dict(zip(TRAINING_LOSS_NAMES, means, strict=True))


def run_training(
    config: CurveNavConfig,
    resume_path: Path | None = None,
) -> None:
    """Train with one process per GPU and one globally sharded policy loader."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CurveNav production training requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    ddp = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        bucket_cap_mb=64,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    accelerator = Accelerator(
        mixed_precision="no",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp],
    )
    configure_cuda_training_backend()
    set_seed(config.training.seed, device_specific=True)
    training_contract = build_training_contract(
        config,
        accelerator.num_processes,
        PRECISION_NAME,
    )
    global_batch_size = training_contract["global_batch_size"]
    per_device_batch_size = training_contract["per_device_batch_size"]
    batch_layout = build_distributed_batch_layout(
        global_batch_size,
        per_device_batch_size,
        accelerator.num_processes,
    )
    local_rank_batch_size = batch_layout.rank_batch_sizes[accelerator.process_index]
    micro_batches_per_step = batch_layout.micro_batches_per_step
    steps_per_epoch = training_contract["steps_per_epoch"]
    total_steps = training_contract["total_steps"]
    checkpoint = None
    start_step = 0
    best_validation_loss = float("inf")
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_policy_contract(checkpoint, config)
        validate_training_resume(
            checkpoint,
            config,
            accelerator.num_processes,
            PRECISION_NAME,
        )
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        if not 0 <= start_step < total_steps:
            raise ValueError(
                f"resume step must be in [0, {total_steps}), got {start_step}"
            )
    remaining_steps = total_steps - start_step
    loader_bundle = build_policy_training_loader(
        config.data,
        config.trajectory,
        optimizer_steps=remaining_steps,
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
        seed=config.training.seed,
        sample_offset=start_step * global_batch_size,
    )
    loader = loader_bundle.loader
    policy = build_policy(config)
    criterion = CurveNavCriterion(
        SourceConfigurationSpaceQuery.from_prepared_split(config.data.root, "train"),
        policy.planning_horizon_m,
    )
    compile_static_training_functions(policy)
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
    policy, optimizer, scheduler = accelerator.prepare(
        policy,
        optimizer,
        scheduler,
        device_placement=[True, True, True],
    )
    loader = CudaPrefetchLoader(
        loader,
        loader_bundle.depth_bank,
        accelerator.device,
    )
    validation_bundle = build_policy_validation_loader(
        config.data,
        config.trajectory,
        batch_size=local_rank_batch_size,
        num_workers=config.training.num_workers,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    validation_loader = CudaPrefetchLoader(
        validation_bundle.loader,
        validation_bundle.depth_bank,
        accelerator.device,
    )
    validation_criterion = CurveNavCriterion(
        SourceConfigurationSpaceQuery.from_prepared_split(
            config.data.root, "validation"
        ),
        accelerator.unwrap_model(policy).planning_horizon_m,
    )
    expected_loader_steps = remaining_steps * micro_batches_per_step
    if len(loader) != expected_loader_steps:
        raise RuntimeError(
            "unexpected distributed loader length: "
            f"{len(loader)} != {expected_loader_steps}"
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
        )
        restore_process_rng_state(checkpoint, accelerator.process_index)

    accelerator.print(
        json.dumps(
            {
                "event": "training_start",
                "world_size": accelerator.num_processes,
                "minimum_per_rank_batch_size": training_contract[
                    "minimum_per_rank_batch_size"
                ],
                "maximum_per_rank_batch_size": training_contract[
                    "maximum_per_rank_batch_size"
                ],
                "local_rank_batch_size": local_rank_batch_size,
                "per_device_batch_size": per_device_batch_size,
                "micro_batches_per_step": micro_batches_per_step,
                "global_batch_size": global_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "samples_per_epoch": training_contract["samples_per_epoch"],
                "total_steps": total_steps,
                "resume_step": start_step,
                "precision": PRECISION_NAME,
            },
            ensure_ascii=False,
        )
    )

    policy.train()
    step = start_step
    window_losses = torch.zeros(len(TRAINING_LOSS_NAMES), device=accelerator.device)
    window_steps = 0
    window_start = time.perf_counter()
    checkpoint_interval = config.training.checkpoint_every_epochs * steps_per_epoch
    loader_iterator = iter(loader)
    # DDP averages equal-size ranks; accumulation averages their micro-batches.
    batch_weight = 1.0 / micro_batches_per_step
    for _ in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        current_losses = torch.zeros_like(window_losses)
        for micro_step in range(micro_batches_per_step):
            batch = next(loader_iterator)
            prepared = unpack_policy_batch(batch)
            flow_source = torch.randn_like(prepared.target.curve_values)
            synchronize = micro_step + 1 == micro_batches_per_step
            synchronization_context = (
                nullcontext() if synchronize else accelerator.no_sync(policy)
            )
            with synchronization_context:
                output = policy(
                    prepared.condition,
                    prepared.target,
                    flow_source,
                )
                losses = criterion(output, batch)
                accelerator.backward(losses.loss * batch_weight)
            current_losses += (
                torch.stack(losses.logging_values()).detach().float() * batch_weight
            )
        grad_norm = accelerator.clip_grad_norm_(
            policy.parameters(), config.training.grad_clip_norm
        )
        finite_update = torch.isfinite(current_losses).all() & torch.isfinite(grad_norm)
        torch._assert_async(
            finite_update,
            "CurveNav mixed-precision update contains a non-finite loss or gradient norm",
        )
        optimizer.step()
        scheduler.step()
        ema.update()

        step += 1
        epoch = (step - 1) // steps_per_epoch + 1
        window_steps += 1
        window_losses += current_losses
        if step == 1 or step % config.training.log_every_steps == 0:
            mean_losses = accelerator.reduce(
                window_losses / window_steps, reduction="mean"
            ).tolist()
            # Reading the reduced losses waits for the completed CUDA updates.
            elapsed = time.perf_counter() - window_start
            accelerator.print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        **dict(zip(TRAINING_LOSS_NAMES, mean_losses, strict=True)),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "gradient_norm": grad_norm.detach().float().item(),
                        "samples_per_second": global_batch_size
                        * window_steps
                        / elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            window_losses.zero_()
            window_steps = 0
            window_start = time.perf_counter()

        if step % steps_per_epoch == 0:
            validation = _validate(
                accelerator,
                policy,
                ema,
                validation_loader,
                validation_criterion,
                config.training.seed,
            )
            improved = validation["loss"] < best_validation_loss
            best_validation_loss = min(best_validation_loss, validation["loss"])
            accelerator.print(json.dumps({
                "event": "validation",
                "epoch": epoch,
                "step": step,
                **{f"validation_{name}": value for name, value in validation.items()},
                "best_validation_loss": best_validation_loss,
            }), flush=True)
        else:
            improved = False

        filenames = (
            (("checkpoint.pt",) if step % checkpoint_interval == 0 else ())
            + (("best.pt",) if improved else ())
        )
        if filenames:
            _save_checkpoint(
                accelerator,
                policy,
                optimizer,
                scheduler,
                ema,
                config,
                step=step,
                best_validation_loss=best_validation_loss,
                training_contract=training_contract,
                filenames=filenames,
            )

    if step % checkpoint_interval != 0:
        _save_checkpoint(
            accelerator,
            policy,
            optimizer,
            scheduler,
            ema,
            config,
            step=step,
            best_validation_loss=best_validation_loss,
            training_contract=training_contract,
            filenames=("checkpoint.pt",),
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
    args = parser.parse_args()
    run_training(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
