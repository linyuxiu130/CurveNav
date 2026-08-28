"""Resumable state for the single CurveNav production training route."""

from dataclasses import asdict
from typing import Any, Mapping

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from curvenav.config import CurveNavConfig
from curvenav.conditioning import (
    CONDITION_ENCODER_TYPE,
    HISTORY_GEOMETRY_QUERY_COUNT,
)
from curvenav.encoders.depth import DEPTH_ENCODER_TYPE
from curvenav.encoders import POINT_GOAL_ENCODER_TYPE
from curvenav.models import TRAJECTORY_DECODER_TYPE
from curvenav.physical import (
    EXTRA_CLEARANCE_M,
    MINIMUM_OBSTACLE_HEIGHT_M,
    ROBOT_RADIUS_M,
    ROBOT_HEIGHT_M,
)
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.history import HISTORY_TRAINING_DISTRIBUTION
from curvenav.training.batching import build_distributed_batch_layout
from curvenav.trajectory import (
    CURVATURE_PARAMETERIZATION_TYPE,
    CURVATURE_VARIATION_REGULARIZATION,
    CURVE_INTEGRATION_OVERSAMPLE_FACTOR,
)
from curvenav.models.policy import (
    FLOW_SOURCE_ENDPOINT_PROBABILITY,
    INFERENCE_SOURCE_SEED,
)


CHECKPOINT_TYPE = "curvenav_metric_curve_flow_policy"
PRODUCTION_WORLD_SIZES = tuple(range(1, 9))


def build_training_contract(
    config: CurveNavConfig,
    world_size: int,
) -> dict[str, int]:
    """Resolve the exact optimizer-step topology represented by a checkpoint."""
    if world_size not in PRODUCTION_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {PRODUCTION_WORLD_SIZES}, got {world_size}"
        )
    global_batch_size = config.training.global_batch_size
    per_device_batch_size = config.training.per_device_batch_size
    layout = build_distributed_batch_layout(
        global_batch_size,
        per_device_batch_size,
        world_size,
    )
    steps_per_epoch = config.training.samples_per_epoch // global_batch_size
    return {
        "world_size": world_size,
        "minimum_per_rank_batch_size": min(layout.rank_batch_sizes),
        "maximum_per_rank_batch_size": max(layout.rank_batch_sizes),
        "per_device_batch_size": per_device_batch_size,
        "micro_batches_per_step": layout.micro_batches_per_step,
        "global_batch_size": global_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": config.training.epochs * steps_per_epoch,
    }


def build_policy_contract(config: CurveNavConfig) -> dict[str, Any]:
    data = config.data
    trajectory = config.trajectory
    depth = config.depth_encoder
    point_goal = config.point_goal_encoder
    condition = config.condition_encoder
    decoder = config.trajectory_decoder
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
        "model_architecture": {
            "model_dim": depth.model_dim,
            "depth_token_grid": [
                depth.frame_tokens_height,
                depth.frame_tokens_width,
            ],
            "depth_dropout": depth.dropout,
            "point_goal_hidden_dim": point_goal.hidden_dim,
            "condition_heads": condition.transformer_heads,
            "condition_layers": condition.transformer_layers,
            "condition_dropout": condition.dropout,
            "trajectory_decoder_layers": decoder.transformer_layers,
            "trajectory_decoder_heads": decoder.transformer_heads,
            "trajectory_path_tokens": decoder.path_tokens,
            "trajectory_decoder_dropout": decoder.dropout,
        },
        "trajectory_dimensions": 2,
        "planar_axis_convention": "x_forward_y_left",
        "point_goal_semantics": "mission_destination_in_current_robot_xy",
        "point_goal_encoder_type": POINT_GOAL_ENCODER_TYPE,
        "point_goal_features": "direction_plus_log_range",
        "point_goal_clip_distance_m": config.point_goal_encoder.goal_clip_distance_m,
        "trajectory_supervision": (
            "fixed_future_expert_projected_into_production_curve_values"
        ),
        "expert_curve_projection": (
            "production_smooth_heading_regularized_least_squares"
        ),
        "arc_length_policy": "learned_positive_metric_arc_length_independent_of_point_goal_range",
        "trajectory_endpoint_policy": "heun_integrated_conditional_flow_curve",
        "curve_boundary_conditions": "origin_and_robot_forward_initial_tangent",
        "trajectory_decoder_type": TRAJECTORY_DECODER_TYPE,
        "flow_steps": decoder.flow_steps,
        "flow_source": "isotropic_gaussian_training_prior_with_fixed_typical_set_inference",
        "inference_source_seed": INFERENCE_SOURCE_SEED,
        "training_source_endpoint_probability": FLOW_SOURCE_ENDPOINT_PROBABILITY,
        "flow_path": "conditional_optimal_transport_displacement_interpolation",
        "flow_solver": "fixed_step_heun",
        "flow_time_embedding": "smooth_scalar_mlp",
        "training_objective": "conditional_flow_matching_euclidean_velocity_mse",
        "trajectory_prediction": "single_flow_generated_smooth_metric_curvature_trajectory",
        "condition_encoder_type": CONDITION_ENCODER_TYPE,
        "visual_context": (
            "goal_relative_uncompressed_current_metric_tokens_plus_"
            "aligned_compressed_history"
        ),
        "depth_token_pooling": (
            "nearest_body_height_obstacle_else_nearest_surface_metric_xyz"
        ),
        "observation_to_current": "planar_rigid_transform_used_for_metric_xyz_alignment",
        "visual_compression": "full_current_grid_plus_32_history_geometry_queries",
        "condition_context": "four_layer_joint_goal_current_history_transformer",
        "trajectory_condition_interaction": (
            "decoded_path_tokens_with_metric_current_obstacle_cross_attention"
        ),
        "goal_conditioning": "goal_relative_metric_geometry_without_codec_length_coupling",
        "temporal_modeling": (
            "history_transforms_only_align_geometry_without_a_route_state_token"
        ),
        "history_training_distribution": HISTORY_TRAINING_DISTRIBUTION,
        "state_token_count": 0,
        "condition_token_count": (
            1
            + depth.frame_tokens_height * depth.frame_tokens_width
            + HISTORY_GEOMETRY_QUERY_COUNT
        ),
        "body_obstacle_geometry": {
            "robot_radius_m": ROBOT_RADIUS_M,
            "extra_clearance_m": EXTRA_CLEARANCE_M,
            "robot_height_m": ROBOT_HEIGHT_M,
            "minimum_obstacle_height_m": MINIMUM_OBSTACLE_HEIGHT_M,
        },
        "num_curve_tokens": trajectory.num_curvature_control_points + 1,
        "num_curvature_control_points": trajectory.num_curvature_control_points,
        "curvature_spline_degree": trajectory.curvature_spline_degree,
        "num_path_points": trajectory.num_path_points,
        "path_sampling": "fixed_uniform_metric_arc_progress",
        "curve_integration_oversample_factor": CURVE_INTEGRATION_OVERSAMPLE_FACTOR,
        "curve_coordinates": CURVATURE_PARAMETERIZATION_TYPE,
        "visual_planning_scale_m": (
            data.future_steps * data.expert_waypoint_spacing_m
        ),
        "curve_value_semantics": "metric_arc_length_m_then_curvature_controls_inv_m",
        "flow_length_transform": "standardized_inverse_softplus",
        "length_pretransform_mean": trajectory.length_pretransform_mean,
        "length_pretransform_std": trajectory.length_pretransform_std,
        "curvature_control_mean_inv_m": trajectory.curvature_control_mean_inv_m,
        "curvature_control_std_inv_m": trajectory.curvature_control_std_inv_m,
        "curvature_variation_regularization": (
            CURVATURE_VARIATION_REGULARIZATION
        ),
    }


def build_training_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    ema: ExponentialMovingAverage,
    grad_scaler: Any,
    config: CurveNavConfig,
    step: int,
    *,
    training_contract: Mapping[str, int],
    rng_states: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Build the complete state required to resume FP16 optimizer updates."""
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    world_size = int(training_contract["world_size"])
    expected_training_contract = build_training_contract(config, world_size)
    if dict(training_contract) != expected_training_contract:
        raise ValueError("training contract does not match the CurveNav configuration")
    if step > expected_training_contract["total_steps"]:
        raise ValueError("checkpoint step exceeds the production training schedule")
    for name in ("cpu", "cuda"):
        state = rng_states.get(name)
        if (
            not isinstance(state, Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 2
            or state.shape[0] != world_size
        ):
            raise ValueError(f"{name} RNG state must be uint8 [world_size, N]")
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "grad_scaler": grad_scaler.state_dict(),
        "config": asdict(config),
        "policy_contract": build_policy_contract(config),
        "training_contract": expected_training_contract,
        "rng_states": {
            "cpu": rng_states["cpu"].cpu(),
            "cuda": rng_states["cuda"].cpu(),
        },
    }


def validate_training_resume(
    checkpoint: Mapping[str, Any],
    config: CurveNavConfig,
    world_size: int,
) -> None:
    """Reject any resume that would change samples, steps, or random streams."""
    if checkpoint.get("config") != asdict(config):
        raise ValueError("resume checkpoint configuration does not exactly match")
    expected = build_training_contract(config, world_size)
    if checkpoint.get("training_contract") != expected:
        raise ValueError("resume checkpoint training topology does not exactly match")
    rng_states = checkpoint.get("rng_states")
    if not isinstance(rng_states, Mapping):
        raise ValueError("resume checkpoint has no per-rank RNG states")
    for name in ("cpu", "cuda"):
        state = rng_states.get(name)
        if (
            not isinstance(state, Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 2
            or state.shape[0] != world_size
        ):
            raise ValueError(f"resume checkpoint has invalid {name} RNG state")


def restore_process_rng_state(
    checkpoint: Mapping[str, Any],
    process_index: int,
) -> None:
    """Restore this rank after all RNG-consuming model setup is complete."""
    rng_states = checkpoint["rng_states"]
    torch.set_rng_state(rng_states["cpu"][process_index].cpu())
    torch.cuda.set_rng_state(rng_states["cuda"][process_index].cpu())


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
    expected = build_policy_contract(config)
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
    grad_scaler: Any,
) -> int:
    """Restore the complete state needed to continue optimizer updates."""
    validate_policy_contract(checkpoint, config)
    required = {"step", "model", "optimizer", "scheduler", "ema", "grad_scaler"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"training checkpoint is missing state: {missing}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    ema.load_state_dict(checkpoint["ema"])
    grad_scaler.load_state_dict(checkpoint["grad_scaler"])
    step = int(checkpoint["step"])
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    return step
