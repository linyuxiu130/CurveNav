"""Tensor contracts shared across CurveNav modules."""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class PolicyCondition:
    """Robot-centric observations used to condition a local trajectory."""

    depth: Tensor
    point_goal: Tensor
    camera_intrinsics: Tensor
    camera_to_body: Tensor
    observation_to_current: Tensor
    observation_age_s: Tensor
    observation_valid: Tensor

    def validate(self) -> None:
        if self.depth.ndim != 5 or self.depth.shape[2] != 1:
            raise ValueError(
                "depth must be [B,F,1,H,W], normalized optical Z; zero is invalid"
            )
        b, f, _, h, w = self.depth.shape
        shapes = {
            "point_goal": (b, 2),
            "camera_intrinsics": (b, f, 3, 3),
            "camera_to_body": (b, f, 4, 4),
            "observation_to_current": (b, f, 4, 4),
            "observation_age_s": (b, f),
            "observation_valid": (b, f),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.observation_valid.dtype != torch.bool:
            raise TypeError("observation_valid must be boolean")


@dataclass
class DepthFeatures:
    """Learned image evidence plus the aligned observed configuration field."""

    tokens: Tensor
    token_valid: Tensor
    metric_position: Tensor
    configuration_field: Tensor


@dataclass
class ConfigurationFeatures:
    """One target-independent metric BEV memory."""

    tokens: Tensor
    metric_position: Tensor
    observed_fraction: Tensor


@dataclass
class ConditionFeatures:
    """Target-independent local scene memory and separate metric intents."""

    tokens: Tensor
    token_valid: Tensor
    metric_position: Tensor
    surface_hit: Tensor
    frame_age: Tensor
    motion_token: Tensor
    metric_reference: Tensor
    terminal_goal: Tensor
    configuration_field: Tensor


@dataclass
class TrajectoryTarget:
    """Expert physical B-spline controls used by the production codec."""

    curve_values: Tensor

    def validate(self) -> None:
        if self.curve_values.ndim != 2:
            raise ValueError("curve_values must have shape [B,C]")


@dataclass
class TrajectoryPrediction:
    path: Tensor
    proposal_coordinates: Tensor
