"""Typed configuration for CurveNav model components."""

import math
from dataclasses import dataclass, field as dataclass_field


@dataclass(frozen=True)
class DataSourceConfig:
    root: str = "dataset"
    split: str = "train"
    weight: float = 1.0

    def validate(self) -> None:
        if not self.root:
            raise ValueError("data source root cannot be empty")
        if self.split not in {"all", "train", "val"}:
            raise ValueError("data source split must be 'all', 'train', or 'val'")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("data source weight must be positive")


def _default_training_sources() -> tuple[DataSourceConfig, ...]:
    return (DataSourceConfig(),)


def _default_validation_sources() -> tuple[DataSourceConfig, ...]:
    return (DataSourceConfig(split="val"),)


@dataclass(frozen=True)
class DataConfig:
    training_sources: tuple[DataSourceConfig, ...] = dataclass_field(
        default_factory=_default_training_sources
    )
    validation_sources: tuple[DataSourceConfig, ...] = dataclass_field(
        default_factory=_default_validation_sources
    )
    sequence_length: int = 4
    frame_skip: int = 2
    min_gap: int = 5
    max_gap: int = 42
    image_height: int = 168
    image_width: int = 224
    depth_units_per_m: float = 1000.0
    max_depth_m: float = 8.0

    def validate(self) -> None:
        if not self.training_sources:
            raise ValueError("data.training_sources cannot be empty")
        if not self.validation_sources:
            raise ValueError("data.validation_sources cannot be empty")
        for source in (*self.training_sources, *self.validation_sources):
            source.validate()
        if self.sequence_length < 1:
            raise ValueError("data.sequence_length must be positive")
        if self.frame_skip < 0:
            raise ValueError("data.frame_skip cannot be negative")
        if not 1 <= self.min_gap <= self.max_gap:
            raise ValueError("data gap must satisfy 1 <= min_gap <= max_gap")
        if self.image_height < 16 or self.image_width < 16:
            raise ValueError("data image dimensions must be at least 16")
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.depth_units_per_m, self.max_depth_m)
        ):
            raise ValueError("data depth scales must be positive")


@dataclass(frozen=True)
class TrajectoryConfig:
    num_control_points: int = 12
    degree: int = 3
    num_path_points: int = 64
    scale_xy: tuple[float, float] = (5.6, 2.5)

    def validate(self) -> None:
        if self.degree != 3:
            raise ValueError("CurveNav uses one fixed cubic B-spline degree")
        if self.num_control_points < self.degree + 1:
            raise ValueError("num_control_points must be at least degree + 1")
        if self.num_path_points < self.num_control_points:
            raise ValueError("num_path_points must be at least num_control_points")
        if len(self.scale_xy) != 2 or any(
            not math.isfinite(scale) or scale <= 0 for scale in self.scale_xy
        ):
            raise ValueError("scale_xy values must be positive")


@dataclass(frozen=True)
class DepthEncoderConfig:
    model_dim: int = 256
    frame_tokens_per_side: int = 4
    dropout: float = 0.0


@dataclass(frozen=True)
class GoalEncoderConfig:
    model_dim: int = 256
    hidden_dim: int = 256


@dataclass(frozen=True)
class MotionEncoderConfig:
    model_dim: int = 256
    hidden_dim: int = 256


@dataclass(frozen=True)
class ConditionEncoderConfig:
    model_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 4
    dropout: float = 0.0


@dataclass(frozen=True)
class FieldConfig:
    model_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 4
    dropout: float = 0.0


@dataclass(frozen=True)
class RectifiedFlowConfig:
    inference_steps: int = 8
    source_std_xy: tuple[float, float] = (0.04, 0.12)


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda"
    seed: int = 42
    batch_size: int = 256
    samples_per_epoch: int = 38_400
    epochs: int = 200
    num_workers: int = 8
    prefetch_factor: int = 2
    warmup_epochs: int = 5
    min_learning_rate_factor: float = 0.01
    overfit_steps: int = 1000
    overfit_batch_size: int = 8
    overfit_max_loss_ratio: float = 0.1
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 20
    output_dir: str = "outputs/train_v3_sand_goal_aligned"
    checkpoint_path: str = "outputs/train_v3_sand_goal_aligned/overfit.pt"
    learning_rate: float = 4e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999


@dataclass(frozen=True)
class CurveNavConfig:
    data: DataConfig = dataclass_field(default_factory=DataConfig)
    trajectory: TrajectoryConfig = dataclass_field(default_factory=TrajectoryConfig)
    depth_encoder: DepthEncoderConfig = dataclass_field(default_factory=DepthEncoderConfig)
    goal_encoder: GoalEncoderConfig = dataclass_field(default_factory=GoalEncoderConfig)
    motion_encoder: MotionEncoderConfig = dataclass_field(default_factory=MotionEncoderConfig)
    condition_encoder: ConditionEncoderConfig = dataclass_field(default_factory=ConditionEncoderConfig)
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    rectified_flow: RectifiedFlowConfig = dataclass_field(default_factory=RectifiedFlowConfig)
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.data.validate()
        self.trajectory.validate()
        if self.data.sequence_length != 4:
            raise ValueError("CurveNav uses exactly four ordered depth frames")
        if self.depth_encoder.frame_tokens_per_side < 1:
            raise ValueError("depth_encoder.frame_tokens_per_side must be positive")
        if self.goal_encoder.hidden_dim < 1:
            raise ValueError("goal_encoder.hidden_dim must be positive")
        if self.motion_encoder.hidden_dim < 1:
            raise ValueError("motion_encoder.hidden_dim must be positive")
        dims = {
            self.depth_encoder.model_dim,
            self.goal_encoder.model_dim,
            self.motion_encoder.model_dim,
            self.condition_encoder.model_dim,
            self.field.model_dim,
        }
        if len(dims) != 1:
            raise ValueError("all encoder, condition, and field model dimensions must match")
        model_dim = next(iter(dims))
        if model_dim < 2:
            raise ValueError("model_dim must be at least 2")
        for name, heads in (
            ("condition_encoder", self.condition_encoder.transformer_heads),
            ("field", self.field.transformer_heads),
        ):
            if heads < 1:
                raise ValueError(f"{name}.transformer_heads must be positive")
            if model_dim % heads != 0:
                raise ValueError(f"model_dim must be divisible by {name}.transformer_heads")
        for name, layers in (
            ("condition_encoder", self.condition_encoder.transformer_layers),
            ("field", self.field.transformer_layers),
        ):
            if layers < 1:
                raise ValueError(f"{name}.transformer_layers must be positive")
        if self.rectified_flow.inference_steps < 1:
            raise ValueError("rectified_flow.inference_steps must be positive")
        source_std_xy = self.rectified_flow.source_std_xy
        if len(source_std_xy) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in source_std_xy
        ):
            raise ValueError("rectified_flow.source_std_xy values must be positive")
        for name, dropout in (
            ("depth_encoder", self.depth_encoder.dropout),
            ("condition_encoder", self.condition_encoder.dropout),
            ("field", self.field.dropout),
        ):
            if not 0 <= dropout < 1:
                raise ValueError(f"{name}.dropout must be in [0, 1)")
        if self.training.device not in {"cuda", "cpu"}:
            raise ValueError("training.device must be 'cuda' or 'cpu'")
        if not 0 <= self.training.seed < 2**32:
            raise ValueError("training.seed must be in [0, 2**32)")
        positive_integers = {
            "batch_size": self.training.batch_size,
            "samples_per_epoch": self.training.samples_per_epoch,
            "epochs": self.training.epochs,
            "num_workers": self.training.num_workers,
            "prefetch_factor": self.training.prefetch_factor,
            "overfit_steps": self.training.overfit_steps,
            "overfit_batch_size": self.training.overfit_batch_size,
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
        if not math.isfinite(self.training.overfit_max_loss_ratio) or not (
            0 < self.training.overfit_max_loss_ratio < 1
        ):
            raise ValueError("training.overfit_max_loss_ratio must be in (0, 1)")
        if not self.training.output_dir:
            raise ValueError("training.output_dir cannot be empty")
        if not self.training.checkpoint_path:
            raise ValueError("training.checkpoint_path cannot be empty")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.training.weight_decay < 0:
            raise ValueError("training.weight_decay cannot be negative")
        if self.training.grad_clip_norm <= 0:
            raise ValueError("training.grad_clip_norm must be positive")
        if not 0 < self.training.ema_decay < 1:
            raise ValueError("training.ema_decay must be in (0, 1)")
