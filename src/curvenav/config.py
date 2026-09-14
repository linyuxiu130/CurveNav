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
    root: str = "data/policy_dataset-depth-memory"
    observation_frames: int = 4
    expert_waypoint_spacing_m: float = 0.15
    future_steps: int = 24
    image_height: int = 126
    image_width: int = 224
    max_depth_m: float = 5.0
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
    num_control_points: int = 8
    spline_degree: int = 3
    num_path_points: int = 64
    control_increment_mean_xy_m: tuple[float, ...] = (
        0.19642330136755862,
        0.0,
        0.3915094997185144,
        0.00021861691377818637,
        0.5476898505751266,
        -0.0014516595218387449,
        0.5035989823812841,
        -0.006019467515492705,
        0.4582797710610274,
        -0.01078719027836341,
        0.27792310350004706,
        -0.010353431965547815,
        0.13242820373850755,
        -0.005304436046028459,
    )
    control_increment_std_xy_m: tuple[float, ...] = (
        0.050220991216198226,
        0.050220991216198226,
        0.11429249515751656,
        0.11429249515751656,
        0.2282398416282557,
        0.2282398416282557,
        0.27447920007578686,
        0.27447920007578686,
        0.31187694698071,
        0.31187694698071,
        0.22779275189818496,
        0.22779275189818496,
        0.1147950114801321,
        0.1147950114801321,
    )

    def validate(self) -> None:
        if self.num_control_points != 8:
            raise ValueError("CurveNav uses exactly eight B-spline controls")
        if self.spline_degree != 3:
            raise ValueError("CurveNav uses one clamped cubic B-spline")
        if self.num_path_points != 64:
            raise ValueError("CurveNav uses exactly sixty-four metric path samples")
        if len(self.control_increment_mean_xy_m) != 14 or not all(
            math.isfinite(value) for value in self.control_increment_mean_xy_m
        ):
            raise ValueError(
                "control increment mean must contain fourteen finite values"
            )
        if len(self.control_increment_std_xy_m) != 14 or not all(
            math.isfinite(value) and value > 0
            for value in self.control_increment_std_xy_m
        ):
            raise ValueError(
                "control increment standard deviation must contain fourteen positives"
            )


@dataclass(frozen=True)
class DepthEncoderConfig:
    model_dim: int = 384
    frame_tokens_height: int = 8
    frame_tokens_width: int = 12
    dropout: float = 0.0


@dataclass(frozen=True)
class ConditionEncoderConfig:
    model_dim: int = 384
    bev_grid_size: int = 16


@dataclass(frozen=True)
class TrajectoryDecoderConfig:
    model_dim: int = 384
    transformer_layers: int = 12
    integration_steps: int = 2
    transformer_heads: int = 8
    dropout: float = 0.0

    def validate(self) -> None:
        if self.transformer_layers < 1 or self.integration_steps < 1:
            raise ValueError("decoder layers and integration steps must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    gradient_accumulation_steps: int = 1
    per_device_batch_size: int = 128
    samples_per_epoch: int = 40_960
    epochs: int = 50
    num_workers: int = 2
    prefetch_factor: int = 2
    warmup_epochs: int = 5
    min_learning_rate_factor: float = 0.01
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 1
    output_dir: str = "outputs/train_policy-depth-memory"
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
        if self.condition_encoder.bev_grid_size != 16:
            raise ValueError("CurveNav uses one 16x16 metric BEV memory")
        dims = {
            self.depth_encoder.model_dim,
            self.condition_encoder.model_dim,
            self.trajectory_decoder.model_dim,
        }
        if len(dims) != 1:
            raise ValueError("all policy model dimensions must match")
        model_dim = next(iter(dims))
        if model_dim < 4 or model_dim % 4:
            raise ValueError("model_dim must be positive and divisible by four")
        heads = self.trajectory_decoder.transformer_heads
        if heads < 1:
            raise ValueError("trajectory_decoder.transformer_heads must be positive")
        if model_dim % heads != 0:
            raise ValueError(
                "model_dim must be divisible by trajectory_decoder.transformer_heads"
            )
        for name, dropout in (
            ("depth_encoder", self.depth_encoder.dropout),
            ("trajectory_decoder", self.trajectory_decoder.dropout),
        ):
            if not 0 <= dropout < 1:
                raise ValueError(
                    f"{name}.dropout must be in [0, 1)"
                )
        if not 0 <= self.training.seed < 2**32:
            raise ValueError("training.seed must be in [0, 2**32)")
        positive_integers = {
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
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
