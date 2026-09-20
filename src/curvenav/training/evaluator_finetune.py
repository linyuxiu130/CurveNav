"""Post-train the route evaluator on the deployment candidate distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from curvenav.config_io import load_config
from curvenav.data.batch import unpack_policy_batch
from curvenav.data.loader import (
    build_policy_training_loader,
    build_policy_validation_loader,
)
from curvenav.evaluation.metrics import candidate_selection_metrics
from curvenav.factory import build_policy
from curvenav.models.policy import INFERENCE_CANDIDATES
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.critic import RouteUtilityTeacher
from curvenav.training.prefetch import CudaPrefetchLoader


def _teacher(config, split: str, policy):
    from curvenav.data.privileged import SourceConfigurationSpaceQuery

    return RouteUtilityTeacher(
        SourceConfigurationSpaceQuery.from_prepared_split(config.data.root, split),
        policy.planning_horizon_m,
    )


@torch.no_grad()
def _evaluate(policy, loader, teacher, device) -> dict[str, float]:
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    count = 0
    policy.eval()
    for batch in loader:
        prepared = unpack_policy_batch(batch)
        encoded = policy.encode_condition(prepared.condition)
        candidates = policy.sample_candidate_paths(encoded)
        scores = policy.score_candidate_paths(
            encoded, prepared.condition.point_goal, candidates
        )
        labels = teacher(candidates, batch)
        selected = scores.argmax(dim=1)
        metrics = candidate_selection_metrics(
            scores, selected, labels.score, labels.clearance_m
        )
        totals += torch.stack(
            (
                metrics["selected_utility"].double().sum(),
                metrics["selection_score_regret"].double().sum(),
                metrics["critic_score_mae"].double().sum(),
                metrics["candidate_safe_available"].double().sum(),
                metrics["selection_missed_safe_candidate"].double().sum(),
            )
        )
        count += len(selected)
    totals /= count
    return {
        "selected_utility": float(totals[0]),
        "selection_score_regret": float(totals[1]),
        "critic_score_mae": float(totals[2]),
        "candidate_safe_available": float(totals[3]),
        "selection_missed_safe_candidate": float(totals[4]),
        "samples": count,
    }


def _save_checkpoint(base: dict, policy, output: Path) -> None:
    state = {name: value.detach().cpu() for name, value in policy.state_dict().items()}
    # 后训练产物用于部署，不携带已经与新权重不匹配的联合训练恢复状态。
    result = {name: base[name] for name in ("checkpoint_type", "config", "policy_contract")}
    result["model"] = state
    ema = dict(base["ema"])
    ema["shadow"] = {
        name: state[name] for name in base["ema"]["shadow"]
    }
    result["ema"] = ema
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.is_dir():
        raise IsADirectoryError("--output 必须是 checkpoint 文件路径，例如 outputs/evaluator/checkpoint.pt")
    if not torch.cuda.is_available():
        raise RuntimeError("evaluator post-training requires CUDA")
    device = torch.device(args.device)
    config = load_config(args.config)
    base = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_policy_contract(base, config)
    policy = build_policy(config).to(device)
    policy.load_state_dict(base["model"], strict=True)

    from curvenav.training.ema import ExponentialMovingAverage

    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    ema.load_state_dict(base["ema"])
    ema.copy_to(policy)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.trajectory_evaluator.parameters():
        parameter.requires_grad_(True)
    loader_bundle = build_policy_training_loader(
        config.data,
        config.trajectory,
        optimizer_steps=args.steps,
        global_batch_size=args.batch_size,
        per_device_batch_size=args.batch_size,
        rank=0,
        world_size=1,
        num_workers=args.num_workers,
        prefetch_factor=2,
        seed=args.seed,
    )
    loader = CudaPrefetchLoader(loader_bundle.loader, loader_bundle.depth_bank, device)
    teacher = _teacher(config, "train", policy)
    optimizer = torch.optim.AdamW(
        policy.trajectory_evaluator.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    validation_bundle = build_policy_validation_loader(
        config.data,
        config.trajectory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_source=args.validation_samples_per_source,
    )
    validation_loader = CudaPrefetchLoader(
        validation_bundle.loader, validation_bundle.depth_bank, device
    )
    validation_teacher = _teacher(config, "validation", policy)
    before = _evaluate(policy, validation_loader, validation_teacher, device)
    policy.eval()
    policy.trajectory_evaluator.train()

    running = 0.0
    window_steps = 0
    for step, batch in enumerate(loader, 1):
        prepared = unpack_policy_batch(batch)
        with torch.no_grad():
            encoded = policy.encode_condition(prepared.condition)
            candidates = policy.sample_candidate_paths(encoded)
            labels = teacher(candidates, batch).score
        scores = policy.score_candidate_paths(
            encoded, prepared.condition.point_goal, candidates
        )
        loss = F.smooth_l1_loss(scores.float(), labels.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.trajectory_evaluator.parameters(), 1.0)
        optimizer.step()
        running += float(loss.detach())
        window_steps += 1
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "event": "evaluator_finetune",
                "step": step,
                "steps": args.steps,
                "loss": running / window_steps,
                "candidates": INFERENCE_CANDIDATES,
            }), flush=True)
            running = 0.0
            window_steps = 0
        if step == args.steps:
            break

    validation = _evaluate(policy, validation_loader, validation_teacher, device)
    _save_checkpoint(base, policy, output)
    manifest = {
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "output_checkpoint": str(output.resolve()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "candidate_count": INFERENCE_CANDIDATES,
        "frozen_modules": ["depth_encoder", "condition_encoder", "trajectory_decoder"],
        "trainable_module": "trajectory_evaluator",
        "before_validation": before,
        "validation": validation,
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "evaluator_finetune_done", **manifest}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="输出 checkpoint 文件路径（不是目录）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validation-samples-per-source", type=int, default=128)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
