"""Checkpoint metadata that prevents silent policy-contract mismatches."""

from dataclasses import asdict
from typing import Any, Mapping

from torch import nn
from torch.optim import Optimizer

from curvenav.config import CurveNavConfig
from curvenav.encoders.depth import DEPTH_ENCODER_REVISION
from curvenav.training.ema import ExponentialMovingAverage


def policy_contract(config: CurveNavConfig) -> dict[str, Any]:
    data = config.data
    trajectory = config.trajectory
    return {
        "depth_sequence_length": data.sequence_length,
        "depth_frame_stride": data.frame_skip + 1,
        "depth_image_size": [data.image_height, data.image_width],
        "depth_units_per_m": data.depth_units_per_m,
        "max_depth_m": data.max_depth_m,
        "depth_encoder_revision": DEPTH_ENCODER_REVISION,
        "dimensions": 2,
        "goal_semantics": "sampled_future_point_goal_robot_xy",
        "sand_supervision": "target_endpoint_equals_point_goal",
        "arc_length_policy": "learned_unconstrained",
        "endpoint_policy": "learned_local_endpoint_origin_only",
        "flow_source": "task_goal_capped_greville_line_origin_conditioned_rbf_gp",
        "flow_path": "linear_source_to_data",
        "integration_method": "euler",
        "inference_steps": config.rectified_flow.inference_steps,
        "motion_context": "executed_unit_xy_plus_valid_v2a",
        "num_control_points": trajectory.num_control_points,
        "degree": trajectory.degree,
        "num_path_points": trajectory.num_path_points,
        "scale_xy": list(trajectory.scale_xy),
        "source_std_xy": list(config.rectified_flow.source_std_xy),
    }


def checkpoint_state(
    model: nn.Module,
    config: CurveNavConfig,
    step: int,
    optimizer: Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, but do not write, a complete portable training checkpoint."""
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    state: dict[str, Any] = {
        "format_version": 11,
        "step": step,
        "model": model.state_dict(),
        "config": asdict(config),
        "policy_contract": policy_contract(config),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra is not None:
        state["extra"] = dict(extra)
    return state


def validate_policy_contract(
    checkpoint: Mapping[str, Any],
    config: CurveNavConfig,
) -> None:
    """Fail before loading weights when units, shape, or scales differ."""
    if checkpoint.get("format_version") != 11:
        raise ValueError("CurveNav requires checkpoint format_version 11")
    saved = checkpoint.get("policy_contract")
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint has no CurveNav policy contract")
    expected = policy_contract(config)
    mismatches = {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(f"policy contract mismatch: {details}")


def restore_training_state(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    ema: ExponentialMovingAverage,
    config: CurveNavConfig,
    grad_scaler: Any | None = None,
) -> int:
    """Restore the complete state needed to continue optimizer updates."""
    validate_policy_contract(checkpoint, config)
    required = {"step", "model", "optimizer", "scheduler", "ema"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"training checkpoint is missing state: {missing}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    ema.load_state_dict(checkpoint["ema"])
    if grad_scaler is not None:
        if "grad_scaler" not in checkpoint:
            raise ValueError("FP16 training checkpoint has no gradient scaler state")
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])
    step = int(checkpoint["step"])
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    return step
