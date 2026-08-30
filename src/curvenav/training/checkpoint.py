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
    HISTORICAL_STATE_FEATURES,
)
from curvenav.encoders.configuration import (
    CONFIGURATION_ENCODER_TYPE,
    CONFIGURATION_TOKEN_COUNT,
    CONFIGURATION_TOKEN_GRID_SIZE,
)
from curvenav.encoders.geometry import CONFIGURATION_GRID_SIZE
from curvenav.encoders.depth import DEPTH_ENCODER_TYPE
from curvenav.encoders import POINT_GOAL_ENCODER_TYPE
from curvenav.models import TRAJECTORY_DECODER_TYPE
from curvenav.physical import (
    BODY_OBSTACLE_MIN_Z_M,
    EXTRA_CLEARANCE_M,
    MAXIMUM_TRAVERSABLE_HEIGHT_M,
    MAXIMUM_TRAVERSABLE_SLOPE_DEGREES,
    ROBOT_COLLISION_BOTTOM_Z_M,
    ROBOT_COLLISION_HEIGHT_M,
    ROBOT_COLLISION_TOP_Z_M,
    ROBOT_FOOTPRINT_RADIUS_M,
)
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.batching import build_distributed_batch_layout
from curvenav.trajectory import (
    HEADING_PARAMETERIZATION_TYPE,
)
from curvenav.models.policy import (
    INFERENCE_SOURCE_SEED,
)
from curvenav.models.safety import SAFETY_CLEARANCE_M, SAFETY_OBJECTIVE_TYPE


CHECKPOINT_TYPE = "curvenav_metric_curve_mean_flow_policy"
TRAINING_PRECISION = "bf16_primal_fp32_detached_meanflow_jvp"
PRODUCTION_WORLD_SIZES = tuple(range(1, 9))


def build_training_contract(
    config: CurveNavConfig,
    world_size: int,
) -> dict[str, int | str]:
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
        "mixed_precision": TRAINING_PRECISION,
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
            "configuration_token_grid": [
                CONFIGURATION_TOKEN_GRID_SIZE,
                CONFIGURATION_TOKEN_GRID_SIZE,
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
        "trajectory_supervision": (
            "fixed_future_expert_projected_into_regular_heading_field"
        ),
        "expert_curve_projection": "equal_arc_heading_field_least_squares",
        "arc_length_policy": "positive_softplus_of_standardized_pre_activation",
        "trajectory_endpoint_policy": "single_evaluation_conditional_average_flow_curve",
        "curve_boundary_conditions": (
            "origin_and_robot_longitudinal_initial_heading"
        ),
        "trajectory_decoder_type": TRAJECTORY_DECODER_TYPE,
        "flow_source": "standard_gaussian_training_and_fixed_typical_set_inference",
        "inference_source_seed": INFERENCE_SOURCE_SEED,
        "flow_path": "data_anchored_linear_stochastic_interpolant",
        "flow_solver": "none_direct_average_velocity_transport",
        "flow_time_embedding": "end_time_and_interval_width_mlp",
        "flow_time_sampling": "closed_interval_deterministic_collocation",
        "mean_flow_identity": (
            "instantaneous_boundary_plus_data_anchored_improved_mean_flow_v_loss"
        ),
        "training_objective": (
            "standardized_boundary_complete_improved_mean_flow_mse_plus_"
            "pathwise_configuration_space_risk"
        ),
        "safety_objective_type": SAFETY_OBJECTIVE_TYPE,
        "safety_clearance_m": SAFETY_CLEARANCE_M,
        "trajectory_prediction": (
            "single_mean_flow_generated_regular_metric_heading_curve"
        ),
        "condition_encoder_type": CONDITION_ENCODER_TYPE,
        "visual_context": (
            "goal_independent_current_metric_tokens_plus_complete_"
            "configuration_space_tokens_plus_causal_motion_state"
        ),
        "depth_token_pooling": (
            "nearest_nontraversable_body_height_surface_else_nearest_surface_metric_xyz"
        ),
        "body_obstacle_selection": (
            "robot_collision_height_band_excluding_local_traversable_surface_triangles"
        ),
        "observation_to_current": (
            "planar_rigid_transform_used_for_metric_xyz_alignment_and_motion_state"
        ),
        "visual_compression": "current_frame_visual_grid_only",
        "condition_context": (
            "four_layer_joint_goal_current_motion_and_configuration_transformer"
        ),
        "trajectory_condition_interaction": (
            "coarse_flow_then_two_exact_path_configuration_queries"
        ),
        "goal_conditioning": "unbounded_goal_token_plus_path_anchor_goal_delta",
        "temporal_modeling": (
            "aligned_four_frame_configuration_field_plus_causal_se2_motion_tokens"
        ),
        "state_token_features": HISTORICAL_STATE_FEATURES,
        "state_translation_scale_m": (
            (data.observation_frames - 1) * data.frame_spacing_m
        ),
        "state_token_count": data.observation_frames - 1,
        "condition_token_count": (
            1
            + depth.frame_tokens_height * depth.frame_tokens_width
            + data.observation_frames
            - 1
            + CONFIGURATION_TOKEN_COUNT
        ),
        "decoder_refinement_stages": 3,
        "decoder_stage_supervision": (
            "shared_readout_improved_mean_flow_on_all_three_stages"
        ),
        "configuration_encoder_type": CONFIGURATION_ENCODER_TYPE,
        "configuration_token_grid": [
            CONFIGURATION_TOKEN_GRID_SIZE,
            CONFIGURATION_TOKEN_GRID_SIZE,
        ],
        "configuration_space_field": {
            "grid_size": CONFIGURATION_GRID_SIZE,
            "extent_m": data.future_steps * data.expert_waypoint_spacing_m,
            "channels": [
                "signed_clearance_m",
                "gradient_x",
                "gradient_y",
                "observed",
                "forbidden",
            ],
            "footprint_inflated": True,
        },
        "body_obstacle_geometry": {
            "footprint_radius_m": ROBOT_FOOTPRINT_RADIUS_M,
            "extra_clearance_m": EXTRA_CLEARANCE_M,
            "collision_bottom_z_m": ROBOT_COLLISION_BOTTOM_Z_M,
            "collision_top_z_m": ROBOT_COLLISION_TOP_Z_M,
            "collision_height_m": ROBOT_COLLISION_HEIGHT_M,
            "body_obstacle_min_z_m": BODY_OBSTACLE_MIN_Z_M,
            "maximum_traversable_height_m": MAXIMUM_TRAVERSABLE_HEIGHT_M,
            "maximum_traversable_slope_degrees": (
                MAXIMUM_TRAVERSABLE_SLOPE_DEGREES
            ),
        },
        "num_curve_tokens": trajectory.num_heading_control_points,
        "num_heading_control_points": trajectory.num_heading_control_points,
        "spline_degree": trajectory.spline_degree,
        "num_path_points": trajectory.num_path_points,
        "path_sampling": "fixed_uniform_arc_progress",
        "curve_coordinates": HEADING_PARAMETERIZATION_TYPE,
        "visual_planning_scale_m": (
            data.future_steps * data.expert_waypoint_spacing_m
        ),
        "curve_value_semantics": (
            "metric_arc_length_then_seven_cubic_heading_control_increments_rad"
        ),
        "length_pre_activation_mean": trajectory.length_pre_activation_mean,
        "length_pre_activation_std": trajectory.length_pre_activation_std,
        "heading_increment_mean_rad": list(trajectory.heading_increment_mean_rad),
        "heading_increment_std_rad": list(trajectory.heading_increment_std_rad),
        "flow_coordinate_transform": (
            "standardized_softplus_length_pre_activation_and_heading_increments"
        ),
    }


def build_training_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    ema: ExponentialMovingAverage,
    config: CurveNavConfig,
    step: int,
    *,
    training_contract: Mapping[str, int | str],
    rng_states: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Build the complete state required to resume BF16 optimizer updates."""
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
) -> int:
    """Restore the complete state needed to continue optimizer updates."""
    validate_policy_contract(checkpoint, config)
    expected_keys = {
        "checkpoint_type",
        "step",
        "model",
        "optimizer",
        "scheduler",
        "ema",
        "config",
        "policy_contract",
        "training_contract",
        "rng_states",
    }
    if set(checkpoint) != expected_keys:
        raise ValueError("training checkpoint does not match the unique BF16 state")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    ema.load_state_dict(checkpoint["ema"])
    step = int(checkpoint["step"])
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    return step
