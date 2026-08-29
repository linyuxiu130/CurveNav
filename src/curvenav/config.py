"""Typed configuration for the single CurveNav policy contract."""

import math
from dataclasses import dataclass, field as dataclass_field

from curvenav.physical import (
    DINGO_CAMERA_DOWNWARD_PITCH_DEGREES,
    DINGO_CAMERA_FORWARD_OFFSET_M,
    DINGO_CAMERA_HEIGHT_M,
)


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
    camera_forward_offset_m: float = DINGO_CAMERA_FORWARD_OFFSET_M
    camera_height_m: float = DINGO_CAMERA_HEIGHT_M
    camera_downward_pitch_degrees: float = DINGO_CAMERA_DOWNWARD_PITCH_DEGREES

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
        camera = (
            self.camera_forward_offset_m,
            self.camera_height_m,
            self.camera_downward_pitch_degrees,
        )
        expected_camera = (
            DINGO_CAMERA_FORWARD_OFFSET_M,
            DINGO_CAMERA_HEIGHT_M,
            DINGO_CAMERA_DOWNWARD_PITCH_DEGREES,
        )
        if camera != expected_camera:
            raise ValueError("data camera must match the benchmark Dingo")


@dataclass(frozen=True)
class TrajectoryConfig:
    num_heading_control_points: int = 8
    spline_degree: int = 3
    num_path_points: int = 64
    log_length_mean: float = 0.9411997728025253
    log_length_std: float = 0.6578984994694861
    heading_increment_mean_rad: tuple[float, ...] = (
        0.0027091927181629527,
        0.005207439937511474,
        0.006545409534199333,
        0.005466179133750451,
        0.0027312263638442787,
        0.0029715759159046357,
        -0.0006079559277851468,
    )
    heading_increment_std_rad: tuple[float, ...] = (
        0.1039597669108465,
        0.2211581749264293,
        0.2887267459379414,
        0.2893491499417256,
        0.30020639618179806,
        0.26251125436301076,
        0.14751645522709783,
    )

    def validate(self) -> None:
        if self.num_heading_control_points != 8:
            raise ValueError("CurveNav uses exactly eight heading control points")
        if self.spline_degree != 3:
            raise ValueError("CurveNav uses one clamped cubic heading spline")
        if self.num_path_points < self.num_heading_control_points:
            raise ValueError("num_path_points must cover the heading controls")
        if not math.isfinite(self.log_length_mean):
            raise ValueError("log-length mean must be finite")
        if not math.isfinite(self.log_length_std) or self.log_length_std <= 0:
            raise ValueError("log-length standard deviation must be positive")
        if len(self.heading_increment_mean_rad) != 7 or not all(
            math.isfinite(value) for value in self.heading_increment_mean_rad
        ):
            raise ValueError("heading-increment mean must contain seven finite values")
        if len(self.heading_increment_std_rad) != 7 or not all(
            math.isfinite(value) and value > 0
            for value in self.heading_increment_std_rad
        ):
            raise ValueError(
                "heading-increment standard deviation must contain seven positive values"
            )


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


@dataclass(frozen=True)
class ConditionEncoderConfig:
    model_dim: int = 384
    transformer_layers: int = 4
    transformer_heads: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class TrajectoryDecoderConfig:
    model_dim: int = 384
    transformer_layers: int = 12
    transformer_heads: int = 8
    path_tokens: int = 16
    dropout: float = 0.0

    def validate(self) -> None:
        if self.path_tokens != 16:
            raise ValueError("CurveNav uses exactly sixteen path tokens")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    global_batch_size: int = 1_024
    per_device_batch_size: int = 256
    samples_per_epoch: int = 40_960
    epochs: int = 200
    num_workers: int = 2
    prefetch_factor: int = 2
    warmup_epochs: int = 5
    min_learning_rate_factor: float = 0.01
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 20
    output_dir: str = "outputs/train_policy"
    learning_rate: float = 2e-4
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
    trajectory_decoder: TrajectoryDecoderConfig = dataclass_field(
        default_factory=TrajectoryDecoderConfig
    )
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.data.validate()
        self.trajectory.validate()
        self.trajectory_decoder.validate()
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
        dims = {
            self.depth_encoder.model_dim,
            self.point_goal_encoder.model_dim,
            self.condition_encoder.model_dim,
            self.trajectory_decoder.model_dim,
        }
        if len(dims) != 1:
            raise ValueError("all policy model dimensions must match")
        model_dim = next(iter(dims))
        if model_dim < 4 or model_dim % 4:
            raise ValueError("model_dim must be positive and divisible by four")
        for name, heads in (
            ("condition_encoder", self.condition_encoder.transformer_heads),
            ("trajectory_decoder", self.trajectory_decoder.transformer_heads),
        ):
            if heads < 1:
                raise ValueError(f"{name}.transformer_heads must be positive")
            if model_dim % heads != 0:
                raise ValueError(
                    f"model_dim must be divisible by {name}.transformer_heads"
                )
        for name, layers in (
            ("condition_encoder", self.condition_encoder.transformer_layers),
            ("trajectory_decoder", self.trajectory_decoder.transformer_layers),
        ):
            if layers < 1:
                raise ValueError(f"{name}.transformer_layers must be positive")
        for name, dropout in (
            ("depth_encoder", self.depth_encoder.dropout),
            ("condition_encoder", self.condition_encoder.dropout),
            ("trajectory_decoder", self.trajectory_decoder.dropout),
        ):
            if not 0 <= dropout < 1:
                raise ValueError(f"{name}.dropout must be in [0, 1)")
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
