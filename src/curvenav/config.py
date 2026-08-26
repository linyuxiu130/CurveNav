"""Typed configuration for the single CurveNav policy contract."""

import math
from dataclasses import dataclass, field as dataclass_field


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/policy_dataset"
    observation_frames: int = 4
    frame_spacing_m: float = 0.45
    expert_waypoint_spacing_m: float = 0.15
    future_steps: int = 24
    image_height: int = 126
    image_width: int = 224
    max_depth_m: float = 5.0
    canonical_focal_x_px: float = 166.80851063829786
    canonical_focal_y_px: float = 166.80851063829786
    camera_forward_offset_m: float = 0.28618
    camera_height_m: float = 0.62532
    camera_downward_pitch_degrees: float = 10.0

    def validate(self) -> None:
        if not self.root:
            raise ValueError("data.root cannot be empty")
        if self.observation_frames < 1 or self.future_steps < 1:
            raise ValueError("observation frames and future steps must be positive")
        if (self.image_height, self.image_width) != (126, 224):
            raise ValueError("CurveNav uses one calibrated 224x126 depth camera")
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                self.frame_spacing_m,
                self.expert_waypoint_spacing_m,
                self.max_depth_m,
            )
        ):
            raise ValueError("data spatial scales must be positive")


@dataclass(frozen=True)
class TrajectoryConfig:
    num_target_control_points: int = 8
    target_spline_degree: int = 3
    num_curvature_control_points: int = 7
    curvature_spline_degree: int = 3
    num_path_points: int = 64
    maximum_curvature_inv_m: float = 8.0

    def validate(self) -> None:
        if self.target_spline_degree != 3:
            raise ValueError("expert targets use one fixed cubic B-spline degree")
        if self.num_target_control_points != 8:
            raise ValueError("expert targets use exactly eight spline control points")
        if self.num_curvature_control_points != 7:
            raise ValueError("CurveNav uses exactly seven curvature controls")
        if self.curvature_spline_degree != 3:
            raise ValueError("CurveNav uses one cubic curvature B-spline")
        if self.num_path_points < self.num_target_control_points:
            raise ValueError("num_path_points must cover the target control points")
        if (
            not math.isfinite(self.maximum_curvature_inv_m)
            or self.maximum_curvature_inv_m <= 0
        ):
            raise ValueError("maximum_curvature_inv_m must be positive")


@dataclass(frozen=True)
class DepthEncoderConfig:
    model_dim: int = 384
    frame_tokens_height: int = 8
    frame_tokens_width: int = 12
    dropout: float = 0.0


@dataclass(frozen=True)
class PointGoalEncoderConfig:
    model_dim: int = 384
    hidden_dim: int = 384
    goal_clip_distance_m: float = 25.0


@dataclass(frozen=True)
class ConditionEncoderConfig:
    model_dim: int = 384
    transformer_layers: int = 4
    transformer_heads: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class TrajectoryFlowConfig:
    model_dim: int = 384
    transformer_layers: int = 8
    transformer_heads: int = 8
    integration_steps: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    global_batch_size: int = 1_024
    per_device_batch_size: int = 128
    samples_per_epoch: int = 40_960
    epochs: int = 200
    num_workers: int = 2
    prefetch_factor: int = 2
    warmup_epochs: int = 5
    min_learning_rate_factor: float = 0.01
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 20
    output_dir: str = "outputs/train_policy"
    learning_rate: float = 4e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999


@dataclass(frozen=True)
class CurveNavConfig:
    data: DataConfig = dataclass_field(default_factory=DataConfig)
    trajectory: TrajectoryConfig = dataclass_field(default_factory=TrajectoryConfig)
    depth_encoder: DepthEncoderConfig = dataclass_field(
        default_factory=DepthEncoderConfig
    )
    point_goal_encoder: PointGoalEncoderConfig = dataclass_field(
        default_factory=PointGoalEncoderConfig
    )
    condition_encoder: ConditionEncoderConfig = dataclass_field(
        default_factory=ConditionEncoderConfig
    )
    trajectory_flow: TrajectoryFlowConfig = dataclass_field(
        default_factory=TrajectoryFlowConfig
    )
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.data.validate()
        self.trajectory.validate()
        if self.data.observation_frames != 4:
            raise ValueError(
                "CurveNav uses four depth observations: three past and one current"
            )
        if (
            min(
                self.depth_encoder.frame_tokens_height,
                self.depth_encoder.frame_tokens_width,
            )
            < 1
        ):
            raise ValueError("depth encoder token grid dimensions must be positive")
        if self.point_goal_encoder.hidden_dim < 1:
            raise ValueError("point_goal_encoder.hidden_dim must be positive")
        if (
            not math.isfinite(self.point_goal_encoder.goal_clip_distance_m)
            or self.point_goal_encoder.goal_clip_distance_m <= 0
        ):
            raise ValueError("point_goal_encoder.goal_clip_distance_m must be positive")
        dims = {
            self.depth_encoder.model_dim,
            self.point_goal_encoder.model_dim,
            self.condition_encoder.model_dim,
            self.trajectory_flow.model_dim,
        }
        if len(dims) != 1:
            raise ValueError("all policy model dimensions must match")
        model_dim = next(iter(dims))
        if model_dim < 4 or model_dim % 4:
            raise ValueError("model_dim must be positive and divisible by four")
        for name, heads in (
            ("condition_encoder", self.condition_encoder.transformer_heads),
            ("trajectory_flow", self.trajectory_flow.transformer_heads),
        ):
            if heads < 1:
                raise ValueError(f"{name}.transformer_heads must be positive")
            if model_dim % heads != 0:
                raise ValueError(
                    f"model_dim must be divisible by {name}.transformer_heads"
                )
        for name, layers in (
            ("condition_encoder", self.condition_encoder.transformer_layers),
            ("trajectory_flow", self.trajectory_flow.transformer_layers),
        ):
            if layers < 1:
                raise ValueError(f"{name}.transformer_layers must be positive")
        for name, dropout in (
            ("depth_encoder", self.depth_encoder.dropout),
            ("condition_encoder", self.condition_encoder.dropout),
            ("trajectory_flow", self.trajectory_flow.dropout),
        ):
            if not 0 <= dropout < 1:
                raise ValueError(f"{name}.dropout must be in [0, 1)")
        if self.trajectory_flow.integration_steps < 1:
            raise ValueError("trajectory_flow.integration_steps must be positive")
        if not 0 <= self.training.seed < 2**32:
            raise ValueError("training.seed must be in [0, 2**32)")
        positive_integers = {
            "global_batch_size": self.training.global_batch_size,
            "per_device_batch_size": self.training.per_device_batch_size,
            "samples_per_epoch": self.training.samples_per_epoch,
            "epochs": self.training.epochs,
            "num_workers": self.training.num_workers,
            "prefetch_factor": self.training.prefetch_factor,
            "log_every_steps": self.training.log_every_steps,
            "checkpoint_every_epochs": self.training.checkpoint_every_epochs,
        }
        invalid = [name for name, value in positive_integers.items() if value < 1]
        if invalid:
            raise ValueError(f"training values must be positive: {invalid}")
        if self.training.per_device_batch_size > self.training.global_batch_size:
            raise ValueError("per_device_batch_size cannot exceed global_batch_size")
        if self.training.samples_per_epoch % self.training.global_batch_size:
            raise ValueError("samples_per_epoch must be divisible by global_batch_size")
        if not 0 <= self.training.warmup_epochs < self.training.epochs:
            raise ValueError("training.warmup_epochs must be in [0, epochs)")
        if not 0 < self.training.min_learning_rate_factor <= 1:
            raise ValueError("training.min_learning_rate_factor must be in (0, 1]")
        if not self.training.output_dir:
            raise ValueError("training.output_dir cannot be empty")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.training.weight_decay < 0:
            raise ValueError("training.weight_decay cannot be negative")
        if self.training.grad_clip_norm <= 0:
            raise ValueError("training.grad_clip_norm must be positive")
        if not 0 < self.training.ema_decay < 1:
            raise ValueError("training.ema_decay must be in (0, 1)")
