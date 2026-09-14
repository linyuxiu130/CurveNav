"""Resumable state for the single CurveNav production training route."""

from dataclasses import asdict
from typing import Any, Mapping

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from curvenav.data.history import history_contract
from curvenav.data.obstacle_memory import MEMORY_CONTRACT
from curvenav.config import CurveNavConfig
from curvenav.conditioning import (
    CONDITION_ENCODER_TYPE,
    HISTORICAL_STATE_FEATURES,
)
from curvenav.encoders.depth import DEPTH_ENCODER_TYPE
from curvenav.encoders.configuration import CONFIGURATION_ENCODER_TYPE
from curvenav.models import TRAJECTORY_DECODER_TYPE
from curvenav.models.evaluator import EVALUATOR_TYPE, EVALUATOR_LAYERS
from curvenav.training.critic import CRITIC_TARGET
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.batching import build_distributed_batch_layout
from curvenav.precision import PRECISION_NAME
from curvenav.trajectory import INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE
from curvenav.models.policy import (
    INFERENCE_SOURCE_SEED,
    INFERENCE_CANDIDATES,
    FLOW_TIME_SAMPLING,
)


CHECKPOINT_TYPE = "curvenav_metric_curve_flow_policy"
PRODUCTION_WORLD_SIZES = tuple(range(1, 9))


def build_training_contract(
    config: CurveNavConfig,
    world_size: int,
    mixed_precision: str,
) -> dict[str, int | str]:
    """Resolve the exact optimizer-step topology represented by a checkpoint."""
    if world_size not in PRODUCTION_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {PRODUCTION_WORLD_SIZES}, got {world_size}"
        )
    if mixed_precision != PRECISION_NAME:
        raise ValueError(f"unsupported CurveNav precision: {mixed_precision}")
    per_device_batch_size = config.training.per_device_batch_size
    global_batch_size = (
        per_device_batch_size * world_size * config.training.gradient_accumulation_steps
    )
    layout = build_distributed_batch_layout(
        global_batch_size,
        per_device_batch_size,
        world_size,
    )
    steps_per_epoch = config.training.samples_per_epoch // global_batch_size
    if steps_per_epoch < 1:
        raise ValueError("samples_per_epoch must cover at least one global batch")
    return {
        "mixed_precision": mixed_precision,
        "world_size": world_size,
        "minimum_per_rank_batch_size": min(layout.rank_batch_sizes),
        "maximum_per_rank_batch_size": max(layout.rank_batch_sizes),
        "per_device_batch_size": per_device_batch_size,
        "micro_batches_per_step": layout.micro_batches_per_step,
        "global_batch_size": global_batch_size,
        "samples_per_epoch": steps_per_epoch * global_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": config.training.epochs * steps_per_epoch,
    }


def build_policy_contract(config: CurveNavConfig) -> dict[str, Any]:
    data = config.data
    trajectory = config.trajectory
    depth = config.depth_encoder
    decoder = config.trajectory_decoder
    condition = config.condition_encoder
    return {
        "observation_schema": "depth_sensor10hz_halfpixel_sample4_v5",
        "history": history_contract(),
        "precision": PRECISION_NAME,
        "trajectory_math": "fp32_metric_math_bf16_neural_regions",
        "observation_frames": data.observation_frames,
        "expert_waypoint_spacing_m": data.expert_waypoint_spacing_m,
        "future_steps": data.future_steps,
        "depth_image_size": [data.image_height, data.image_width],
        "max_depth_m": data.max_depth_m,
        "camera_calibration": "per_frame_K_and_optical_camera_to_body_SE3",
        "depth_encoder_type": DEPTH_ENCODER_TYPE,
        "model_architecture": {
            "model_dim": depth.model_dim,
            "depth_token_grid": [
                depth.frame_tokens_height,
                depth.frame_tokens_width,
            ],
            "configuration_bev_grid": [
                condition.bev_grid_size,
                condition.bev_grid_size,
            ],
            "depth_dropout": depth.dropout,
            "trajectory_decoder_layers": decoder.transformer_layers,
            "flow_integration_steps": decoder.integration_steps,
            "trajectory_decoder_heads": decoder.transformer_heads,
            "trajectory_decoder_dropout": decoder.dropout,
        },
        "trajectory_dimensions": 2,
        "planar_axis_convention": "x_forward_y_left",
        "point_goal_semantics": (
            "mission_destination_in_current_robot_xy"
        ),
        "point_goal_conditioning": ("generator_local_terminal_evaluator_mission_direction_log_distance"),
        "trajectory_supervision": (
            "source_cspace_gated_fixed_future_expert_planar_bspline_imitation"
        ),
        "expert_curve_projection": "equal_arc_bspline_forward_tangent_endpoint_fit",
        "trajectory_endpoint_policy": "conditional_flow_curve",
        "curve_boundary_conditions": "fixed_robot_origin",
        "trajectory_decoder_type": TRAJECTORY_DECODER_TYPE,
        "flow_source": "standard_gaussian_training_fixed_iid_gaussian_candidate_bank",
        "inference_source_seed": INFERENCE_SOURCE_SEED,
        "inference_candidates": INFERENCE_CANDIDATES,
        "candidate_selection": "single_goal_conditioned_route_utility_argmax",
        "critic_target": CRITIC_TARGET,
        "trajectory_evaluator_type": EVALUATOR_TYPE,
        "trajectory_evaluator_layers": EVALUATOR_LAYERS,
        "critic_candidates": "one_expert_three_two_step_flow_proposals",
        "flow_path": "data_anchored_linear_stochastic_interpolant",
        "flow_solver": "explicit_euler_noise_to_data",
        "flow_time_embedding": "flow_time_mlp",
        "flow_time_sampling": FLOW_TIME_SAMPLING,
        "training_objective": (
            "conditional_flow_matching_plus_calibrated_route_utility_regression"
        ),
        "trajectory_prediction": ("conditional_flow_planar_cubic_bspline"),
        "condition_encoder_type": CONDITION_ENCODER_TYPE,
        "visual_context": (
            "goal_independent_temporal_visual_observed_configuration_bev"
        ),
        "depth_token_pooling": "body_hit_else_surface_same_pixel_stride16_feature_sample",
        "observation_to_current": (
            "se3_rigid_transform_used_for_metric_xyz_alignment_and_motion_state"
        ),
        "configuration_encoder_type": CONFIGURATION_ENCODER_TYPE,
        "visual_compression": "metric_splat_and_observed_cspace_16x16_bev",
        "condition_context": "target_independent_metric_bev_plus_motion_tokens",
        "trajectory_condition_interaction": (
            "cached_bev_kv_state_dependent_curve_geometry_attention"
        ),
        "path_relative_geometry": (
            "increment_effect_weighted_current_curve_to_bev_attention_bias"
        ),
        "goal_conditioning": (
            "terminal_local_goal_vector_without_straight_template_matching"
        ),
        "obstacle_memory": MEMORY_CONTRACT,
        "temporal_modeling": (
            "depth_pixel_sample_stride16_se3_all_newer_consistency"
        ),
        "state_token_features": HISTORICAL_STATE_FEATURES,
        "state_translation_scale_m": (
            data.future_steps * data.expert_waypoint_spacing_m
        ),
        "state_token_count": data.observation_frames - 1,
        "condition_token_count": (
            condition.bev_grid_size**2 + data.observation_frames - 1
        ),
        "decoder_flow_fields": 1,
        "decoder_stage_supervision": (
            "single_conditional_velocity_field"
        ),
        "source_configuration_space_truth": (
            "native_navigation_grid_endpoint_inclusive_dense_0.025m_"
            "oob_non_executable"
        ),
        "source_configuration_space_role": "dataset_certificate_critic_supervision_and_evaluation",
        "depth_configuration_space_role": (
            "target_independent_observed_bev_plus_current_curve_query"
        ),
        "num_curve_tokens": trajectory.num_control_points - 1,
        "curve_coordinate_dim": 2 * (trajectory.num_control_points - 1),
        "num_control_points": trajectory.num_control_points,
        "spline_degree": trajectory.spline_degree,
        "num_path_points": trajectory.num_path_points,
        "path_sampling": "fixed_uniform_bspline_parameter",
        "curve_coordinates": INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE,
        "visual_planning_scale_m": (data.future_steps * data.expert_waypoint_spacing_m),
        "curve_value_semantics": ("seven_planar_cubic_bspline_control_points_xy_m"),
        "control_increment_mean_xy_m": list(trajectory.control_increment_mean_xy_m),
        "control_increment_std_xy_m": list(trajectory.control_increment_std_xy_m),
        "flow_coordinate_transform": (
            "per_dimension_standardized_physical_control_increments"
        ),
    }


def build_training_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    ema: ExponentialMovingAverage,
    config: CurveNavConfig,
    step: int,
    best_validation_loss: float,
    *,
    training_contract: Mapping[str, int | str],
    rng_states: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Build the complete state required to resume mixed-precision updates."""
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    world_size = int(training_contract["world_size"])
    mixed_precision = str(training_contract["mixed_precision"])
    expected_training_contract = build_training_contract(
        config, world_size, mixed_precision
    )
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
        "best_validation_loss": best_validation_loss,
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
    mixed_precision: str,
) -> None:
    """Reject any resume that would change samples, steps, or random streams."""
    expected_config = asdict(config)
    # Moving artifacts does not change the optimizer, samples or random stream.
    expected_config["training"]["output_dir"] = checkpoint["config"]["training"]["output_dir"]
    if checkpoint["config"] != expected_config:
        raise ValueError("resume checkpoint configuration does not exactly match")
    expected = build_training_contract(config, world_size, mixed_precision)
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
        "best_validation_loss",
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
        raise ValueError(
            "training checkpoint does not match the unique mixed-precision state"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    ema.load_state_dict(checkpoint["ema"])
    step = int(checkpoint["step"])
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    return step
