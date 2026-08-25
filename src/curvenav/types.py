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
        if not self.observation_valid[:, -1].all():
            raise ValueError("the current observation must always be valid")
        if not (
            self.depth.shape[0]
            == self.point_goal.shape[0]
            == self.observation_to_current.shape[0]
            == self.observation_valid.shape[0]
        ):
            raise ValueError("condition batch sizes must match")


@dataclass
class DepthFeatures:
    """Per-frame depth tokens and their metric planar coordinates."""

    tokens: Tensor
    planar_points: Tensor


@dataclass
class ConditionFeatures:
    tokens: Tensor


@dataclass
class TrajectoryTarget:
    """Metric planar B-spline controls and their uniform-arc reference path."""

    control_points: Tensor
    reference_path: Tensor

    def validate(self) -> None:
        if self.control_points.ndim != 3 or self.control_points.shape[-1] != 2:
            raise ValueError("control_points must have shape [B, K, 2]")
        if self.reference_path.ndim != 3 or self.reference_path.shape[-1] != 2:
            raise ValueError("reference_path must have shape [B, P, 2]")
        if self.control_points.shape[0] != self.reference_path.shape[0]:
            raise ValueError("trajectory target batch sizes must match")


@dataclass
class TrajectoryPrediction:
    control_points: Tensor
    path: Tensor
    heading: Tensor
    curvature: Tensor
    candidate_control_points: Tensor
    candidate_paths: Tensor
    candidate_log_probabilities: Tensor
