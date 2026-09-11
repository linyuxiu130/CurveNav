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
    root: str = "data/policy_dataset-depth-forward"
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
        0.19575905874489494,
        0.000013560529621743297,
        0.3902212095715023,
        0.002008237219028404,
        0.558451868792498,
        0.007029782353195491,
        0.5213567234781906,
        0.009539148157356049,
        0.4803115149944964,
        0.009542148952045124,
        0.29207692008915925,
        0.005783167418347843,
        0.13902845983512357,
        0.003241675585678419,
    )
    control_increment_std_xy_m: tuple[float, ...] = (
        0.06963118493486234,
        0.009753948216963312,
        0.1396785939096103,
        0.06463728684735078,
        0.20765432388349703,
        0.1990670224631178,
        0.22015131716511072,
        0.27216707621942426,
        0.24343845568601458,
        0.32393372700831924,
        0.18216146996698057,
        0.24085929622333027,
        0.09459704496084137,
        0.12324686272239309,
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
    transformer_heads: int = 8
    dropout: float = 0.0

    def validate(self) -> None:
        if self.transformer_layers < 2 or self.transformer_layers % 2:
            raise ValueError(
                "trajectory decoder layers must split evenly into proposal and average phases"
            )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    gradient_accumulation_steps: int = 1
    per_device_batch_size: int = 416
    samples_per_epoch: int = 40_960
    epochs: int = 200
    num_workers: int = 2
    prefetch_factor: int = 2
    warmup_epochs: int = 5
    min_learning_rate_factor: float = 0.01
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 20
    output_dir: str = "outputs/train_policy-depth-forward"
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
        if (
            self.trajectory_decoder.transformer_layers < 2
            or self.trajectory_decoder.transformer_layers % 2
        ):
            raise ValueError(
                "trajectory_decoder.transformer_layers must be positive and even"
            )
        for name, dropout in (
            ("depth_encoder", self.depth_encoder.dropout),
            ("trajectory_decoder", self.trajectory_decoder.dropout),
        ):
            if dropout != 0:
                raise ValueError(
                    f"{name}.dropout must be zero for deterministic MeanFlow"
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
        if self.training.per_device_batch_size % 4:
            raise ValueError(
                "per_device_batch_size must contain an exact quarter of deployment intervals"
            )
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
