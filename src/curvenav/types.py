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
    """Per-frame visual and metric geometry tokens."""

    tokens: Tensor
    points: Tensor
    depth: Tensor
    obstacle_valid: Tensor


@dataclass
class ConditionFeatures:
    """Contextual condition memory plus current metric visual geometry."""

    tokens: Tensor
    current_tokens: Tensor
    current_points: Tensor
    current_obstacle_valid: Tensor
    point_goal: Tensor


@dataclass
class TrajectoryTarget:
    """Expert metric length and curvature controls used by the production codec."""

    curve_values: Tensor

    def validate(self) -> None:
        if self.curve_values.ndim != 2:
            raise ValueError("curve_values must have shape [B,C]")


@dataclass
class TrajectoryPrediction:
    path: Tensor
    heading: Tensor
    curvature: Tensor
