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
from curvenav.encoders.depth import DEPTH_ENCODER_TYPE
from curvenav.encoders.configuration import CONFIGURATION_ENCODER_TYPE
from curvenav.models import TRAJECTORY_DECODER_TYPE
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.batching import build_distributed_batch_layout
from curvenav.trajectory import INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE
from curvenav.models.policy import (
    INFERENCE_SOURCE_SEED,
    MEAN_FLOW_TIME_SAMPLING,
)


CHECKPOINT_TYPE = "curvenav_metric_curve_mean_flow_policy"
SUPPORTED_TRAINING_PRECISIONS = frozenset(
    {
        "bf16_neural_fp32_geometry_flow_jvp",
        "fp16_neural_fp32_geometry_flow_jvp",
    }
)
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
    if mixed_precision not in SUPPORTED_TRAINING_PRECISIONS:
        raise ValueError(f"unsupported CurveNav precision: {mixed_precision}")
    global_batch_size = config.training.global_batch_size
    per_device_batch_size = config.training.per_device_batch_size
    layout = build_distributed_batch_layout(
        global_batch_size,
        per_device_batch_size,
        world_size,
    )
    steps_per_epoch = config.training.samples_per_epoch // global_batch_size
    return {
        "mixed_precision": mixed_precision,
        "world_size": world_size,
        "minimum_per_rank_batch_size": min(layout.rank_batch_sizes),
        "maximum_per_rank_batch_size": max(layout.rank_batch_sizes),
        "per_device_batch_size": per_device_batch_size,
        "micro_batches_per_step": layout.micro_batches_per_step,
        "global_batch_size": global_batch_size,
        "flow_interval_assignment": "global_sample_stream_index_mod_four",
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
            "configuration_bev_grid": [
                condition.bev_grid_size,
                condition.bev_grid_size,
            ],
            "depth_dropout": depth.dropout,
            "trajectory_decoder_layers": decoder.transformer_layers,
            "trajectory_decoder_heads": decoder.transformer_heads,
            "trajectory_decoder_dropout": decoder.dropout,
        },
        "trajectory_dimensions": 2,
        "planar_axis_convention": "x_forward_y_left",
        "point_goal_semantics": "mission_destination_in_current_robot_xy",
        "point_goal_conditioning": (
            "metric_goal_reference_retrieval_plus_candidate_to_terminal_local_goal"
        ),
        "trajectory_supervision": (
            "source_cspace_gated_fixed_future_expert_planar_bspline_imitation"
        ),
        "expert_curve_projection": "equal_arc_planar_bspline_least_squares",
        "trajectory_endpoint_policy": "one_step_improved_mean_flow_curve",
        "curve_boundary_conditions": "fixed_robot_origin",
        "trajectory_decoder_type": TRAJECTORY_DECODER_TYPE,
        "flow_source": (
            "standard_gaussian_flow_training_plus_exact_fixed_typical_"
            "deployment_boundary"
        ),
        "inference_source_seed": INFERENCE_SOURCE_SEED,
        "flow_path": "data_anchored_linear_stochastic_interpolant",
        "flow_solver": "none_direct_average_velocity",
        "flow_time_embedding": "end_time_and_interval_width_mlp",
        "flow_time_sampling": MEAN_FLOW_TIME_SAMPLING,
        "mean_flow_identity": "instantaneous_proposal_plus_average_velocity_jvp_target",
        "training_objective": (
            "standardized_euclidean_mean_flow_plus_deployed_"
            "strict_observed_clearance_risk"
        ),
        "trajectory_prediction": ("one_step_improved_mean_flow_planar_cubic_bspline"),
        "condition_encoder_type": CONDITION_ENCODER_TYPE,
        "visual_context": (
            "goal_independent_four_frame_visual_observed_configuration_bev"
        ),
        "depth_token_pooling": "nearest_metric_surface_per_image_token",
        "observation_to_current": (
            "planar_rigid_transform_used_for_metric_xyz_alignment_and_motion_state"
        ),
        "configuration_encoder_type": CONFIGURATION_ENCODER_TYPE,
        "visual_compression": "metric_splat_and_observed_cspace_16x16_bev",
        "condition_context": "target_independent_metric_bev_plus_motion_tokens",
        "trajectory_condition_interaction": (
            "goal_reference_geometry_then_clean_estimate_query_observed_cspace_and_bev"
        ),
        "path_relative_geometry": (
            "goal_reference_then_learned_clean_control_to_bev_metric_attention_bias"
        ),
        "goal_conditioning": (
            "terminal_local_goal_vector_without_straight_template_matching"
        ),
        "temporal_modeling": (
            "shared_learned_four_frame_depth_tokens_with_metric_se2_alignment"
        ),
        "state_token_features": HISTORICAL_STATE_FEATURES,
        "state_translation_scale_m": (
            (data.observation_frames - 1) * data.frame_spacing_m
        ),
        "state_token_count": data.observation_frames - 1,
        "condition_token_count": (
            condition.bev_grid_size**2 + data.observation_frames - 1
        ),
        "decoder_flow_fields": 2,
        "decoder_stage_supervision": (
            "instantaneous_clean_proposal_and_interval_average_velocity"
        ),
        "source_configuration_space_truth": (
            "native_navigation_grid_endpoint_inclusive_dense_0.025m_"
            "oob_non_executable"
        ),
        "source_configuration_space_role": "dataset_certificate_and_evaluation_only",
        "depth_configuration_space_role": (
            "target_independent_observed_bev_plus_flow_candidate_curve_query_"
            "plus_deployed_curve_training_risk"
        ),
        "num_curve_tokens": trajectory.num_control_points - 1,
        "curve_coordinate_dim": 2 * (trajectory.num_control_points - 1),
        "num_control_points": trajectory.num_control_points,
        "spline_degree": trajectory.spline_degree,
        "num_path_points": trajectory.num_path_points,
        "path_sampling": "fixed_uniform_arc_progress",
        "curve_coordinates": INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE,
        "visual_planning_scale_m": (data.future_steps * data.expert_waypoint_spacing_m),
        "curve_value_semantics": ("seven_planar_cubic_bspline_control_points_xy_m"),
        "control_increment_mean_xy_m": list(
            trajectory.control_increment_mean_xy_m
        ),
        "control_increment_std_xy_m": list(
            trajectory.control_increment_std_xy_m
        ),
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
    *,
    training_contract: Mapping[str, int | str],
    rng_states: Mapping[str, Tensor],
    amp_state: Mapping[str, Any] | None,
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
    uses_scaler = mixed_precision.startswith("fp16_")
    if uses_scaler != (amp_state is not None):
        raise ValueError("checkpoint AMP state does not match mixed precision")
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
        "amp_state": dict(amp_state) if amp_state is not None else None,
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
    if checkpoint.get("config") != asdict(config):
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
    amp_state = checkpoint.get("amp_state")
    if mixed_precision.startswith("fp16_"):
        if not isinstance(amp_state, Mapping):
            raise ValueError("FP16 resume checkpoint has no scaler state")
    elif amp_state is not None:
        raise ValueError("BF16 resume checkpoint unexpectedly has scaler state")


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
    scaler: Any | None,
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
        "amp_state",
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
    amp_state = checkpoint["amp_state"]
    if scaler is None:
        if amp_state is not None:
            raise ValueError("checkpoint scaler state requires FP16 runtime")
    else:
        if not isinstance(amp_state, Mapping):
            raise ValueError("FP16 runtime requires checkpoint scaler state")
        scaler.load_state_dict(dict(amp_state))
    step = int(checkpoint["step"])
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    return step
