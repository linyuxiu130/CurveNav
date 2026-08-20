"""Frozen policy and checkpoint contracts shared by critic training stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.critic.config import CriticExperimentConfig
from curvenav.factory import build_policy
from curvenav.models import CurveNavPolicy
from curvenav.training.checkpoint import policy_contract, validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.types import PolicyCondition


CRITIC_FORMAT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenPolicyBundle:
    policy: CurveNavPolicy
    config: CurveNavConfig
    checkpoint_sha256: str


def load_frozen_policy(config: CriticExperimentConfig) -> FrozenPolicyBundle:
    """Strictly load the EMA weights used to produce the sidecar candidates."""
    policy_config = load_config(config.paths.policy_config)
    checkpoint = torch.load(
        config.paths.policy_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_policy_contract(checkpoint, policy_config)
    if "ema" not in checkpoint:
        raise ValueError("critic policy checkpoint has no EMA state")
    policy = build_policy(policy_config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(policy, decay=policy_config.training.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy)
    policy.requires_grad_(False)
    policy.eval()
    return FrozenPolicyBundle(
        policy=policy,
        config=policy_config,
        checkpoint_sha256=sha256_file(config.paths.policy_checkpoint),
    )


def policy_condition(batch: dict[str, torch.Tensor]) -> PolicyCondition:
    return PolicyCondition(
        depth=batch["depth"],
        task_goal=batch["task_goal"],
        motion_context=batch["motion_context"],
    )


def critic_checkpoint(
    *,
    critic: torch.nn.Module,
    config: CriticExperimentConfig,
    policy_bundle: FrozenPolicyBundle,
    sidecar_schema_sha256: str,
    epoch: int,
    step: int,
    validation: dict[str, float],
) -> dict[str, Any]:
    return {
        "format_version": CRITIC_FORMAT_VERSION,
        "epoch": epoch,
        "step": step,
        "critic": critic.state_dict(),
        "config": asdict(config),
        "policy_checkpoint_sha256": policy_bundle.checkpoint_sha256,
        "policy_contract": policy_contract(policy_bundle.config),
        "sidecar_schema_sha256": sidecar_schema_sha256,
        "validation": validation,
        "critic_contract": {
            "inputs": "frozen_condition_tokens_plus_metric_bspline_controls",
            "candidate_identity_input": False,
            "policy_candidates": 8,
            "training_candidates": 10,
            "ranking_loss": "state_balanced_explicit_pareto_softplus",
            "selection": "lexicographic_predicted_collision_margin_then_pareto_score",
        },
    }
