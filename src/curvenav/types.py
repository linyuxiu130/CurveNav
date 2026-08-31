"""Tensor contracts shared across CurveNav modules."""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class PolicyCondition:
    """Robot-centric observations used to condition a local trajectory."""

    depth: Tensor
    point_goal: Tensor
    observation_to_current: Tensor
    observation_valid: Tensor

    def validate(self) -> None:
        if self.depth.ndim != 5 or self.depth.shape[2] != 1:
            raise ValueError("depth must have shape [B, F, 1, H, W]")
        if self.point_goal.ndim != 2 or self.point_goal.shape[-1] != 2:
            raise ValueError("point_goal must have shape [B, 2]")
        if self.observation_to_current.shape != (*self.depth.shape[:2], 4):
            raise ValueError("observation_to_current must have shape [B, F, 4]")
        if self.observation_valid.shape != self.depth.shape[:2]:
            raise ValueError("observation_valid must have shape [B, F]")
        if self.observation_valid.dtype != torch.bool:
            raise TypeError("observation_valid must be boolean")
        if not (
            self.depth.shape[0]
            == self.point_goal.shape[0]
            == self.observation_to_current.shape[0]
            == self.observation_valid.shape[0]
        ):
            raise ValueError("condition batch sizes must match")


@dataclass
class DepthFeatures:
    """Current scene tokens and aligned four-frame metric visual evidence."""

    tokens: Tensor
    configuration_tokens: Tensor
    configuration_points: Tensor
    configuration_visual_valid: Tensor
    configuration_field: Tensor


@dataclass
class ConfigurationFeatures:
    """Metric spatial memory derived only from measured depth geometry."""

    tokens: Tensor
    measured_field: Tensor


@dataclass
class ConditionFeatures:
    """PointGoal scene context and its measured configuration-space field."""

    tokens: Tensor
    path_configuration_field: Tensor


@dataclass
class TrajectoryTarget:
    """Expert metric length and heading-field values used by the codec."""

    curve_values: Tensor

    def validate(self) -> None:
        if self.curve_values.ndim != 2:
            raise ValueError("curve_values must have shape [B,C]")


@dataclass
class TrajectoryPrediction:
    path: Tensor
