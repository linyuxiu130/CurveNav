"""Checkpoint metadata that prevents silent policy-contract mismatches."""

from dataclasses import asdict
from typing import Any, Mapping

from torch import nn
from torch.optim import Optimizer

from curvenav.config import CurveNavConfig
from curvenav.conditioning import CONDITION_ENCODER_TYPE
from curvenav.encoders.depth import DEPTH_ENCODER_TYPE
from curvenav.encoders import POINT_GOAL_ENCODER_TYPE
from curvenav.models import TRAJECTORY_EVALUATOR_TYPE, TRAJECTORY_FLOW_TYPE
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.trajectory import (
    ARC_LENGTH_OVERSAMPLE_FACTOR,
    BSPLINE_BENDING_REGULARIZATION_M4,
)


CHECKPOINT_TYPE = "curvenav_local_policy"


def policy_contract(config: CurveNavConfig) -> dict[str, Any]:
    data = config.data
    trajectory = config.trajectory
    return {
        "observation_frames": data.observation_frames,
        "frame_spacing_m": data.frame_spacing_m,
        "expert_waypoint_spacing_m": data.expert_waypoint_spacing_m,
        "future_steps": data.future_steps,
        "depth_image_size": [data.image_height, data.image_width],
        "max_depth_m": data.max_depth_m,
        "canonical_focal_px": [data.canonical_focal_x_px, data.canonical_focal_y_px],
        "camera_extrinsics": {
            "forward_offset_m": data.camera_forward_offset_m,
            "height_m": data.camera_height_m,
            "downward_pitch_degrees": data.camera_downward_pitch_degrees,
        },
        "depth_encoder_type": DEPTH_ENCODER_TYPE,
        "trajectory_dimensions": 2,
        "point_goal_semantics": "mission_destination_in_current_robot_xy",
        "point_goal_encoder_type": POINT_GOAL_ENCODER_TYPE,
        "point_goal_features": "direction_plus_log_range",
        "point_goal_clip_distance_m": config.point_goal_encoder.goal_clip_distance_m,
        "trajectory_supervision": "fixed_future_expert_waypoints_or_true_goal",
        "bspline_bending_regularization_m4": BSPLINE_BENDING_REGULARIZATION_M4,
        "arc_length_policy": "learned_metric_length_without_rescaling_or_meter_cap",
        "trajectory_endpoint_policy": "implicit_local_subgoal_origin_only",
        "trajectory_flow_type": TRAJECTORY_FLOW_TYPE,
        "trajectory_evaluator_type": TRAJECTORY_EVALUATOR_TYPE,
        "training_objective": "rectified_flow_plus_arc_path_and_tangent",
        "trajectory_selection": "analytic_current_depth_clearance_length_goal_cost",
        "trajectory_safe_center_distance_m": (
            config.trajectory_evaluator.robot_radius_m
            + config.trajectory_evaluator.safety_margin_m
        ),
        "trajectory_clearance_discount_factor": (
            config.trajectory_evaluator.discount_factor
        ),
        "trajectory_cost_weights": [
            config.trajectory_evaluator.clearance_weight,
            config.trajectory_evaluator.length_weight,
            config.trajectory_evaluator.goal_weight,
        ],
        "trajectory_flow_candidates": config.trajectory_flow.inference_candidates,
        "trajectory_flow_integration_steps": config.trajectory_flow.integration_steps,
        "condition_encoder_type": CONDITION_ENCODER_TYPE,
        "visual_context": "masked_spatial_tokens_with_frame_slots_and_planar_backprojection",
        "observation_to_current": "planar_rigid_transform_used_for_depth_token_alignment",
        "visual_compression": "learned_queries_16_tokens_per_depth_frame",
        "num_control_points": trajectory.num_control_points,
        "degree": trajectory.degree,
        "num_path_points": trajectory.num_path_points,
        "path_sampling": "uniform_metric_arc_progress",
        "arc_length_oversample_factor": ARC_LENGTH_OVERSAMPLE_FACTOR,
        "trajectory_scale_xy": [
            trajectory.normalization_scale_m,
            trajectory.normalization_scale_m,
        ],
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
        "checkpoint_type": CHECKPOINT_TYPE,
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
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(f"CurveNav requires checkpoint_type {CHECKPOINT_TYPE!r}")
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
