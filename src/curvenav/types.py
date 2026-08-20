"""Tensor contracts shared across CurveNav modules."""

from dataclasses import dataclass
from torch import Tensor


@dataclass
class PolicyCondition:
    """Robot-centric observations used to condition a local trajectory."""

    depth: Tensor
    task_goal: Tensor
    motion_context: Tensor

    def validate(self) -> None:
        if self.depth.ndim != 5 or self.depth.shape[2] != 1:
            raise ValueError("depth must have shape [B, T, 1, H, W]")
        if self.task_goal.ndim != 2 or self.task_goal.shape[-1] != 2:
            raise ValueError("task_goal must have shape [B, 2]")
        if self.motion_context.ndim != 2 or self.motion_context.shape[-1] != 3:
            raise ValueError("motion_context must have shape [B, 3]")
        if not (
            self.depth.shape[0]
            == self.task_goal.shape[0]
            == self.motion_context.shape[0]
        ):
            raise ValueError("condition batch sizes must match")


@dataclass
class EncodedCondition:
    tokens: Tensor


@dataclass
class TrajectoryTarget:
    """Normalized or metric planar B-spline control points."""

    control_points: Tensor

    def validate(self) -> None:
        if self.control_points.ndim != 3 or self.control_points.shape[-1] != 2:
            raise ValueError("control_points must have shape [B, K, 2]")


@dataclass
class TrajectoryPrediction:
    control_points: Tensor
    dense_path: Tensor
    heading: Tensor
    curvature: Tensor
