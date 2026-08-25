"""Direction and smoothly compressed range of a mission-level PointGoal."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm


POINT_GOAL_ENCODER_TYPE = "direction_and_log_range"


class PointGoalEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 256,
        hidden_dim: int = 256,
        goal_clip_distance_m: float = 25.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(goal_clip_distance_m) or goal_clip_distance_m <= 0:
            raise ValueError("goal_clip_distance_m must be positive")
        self.goal_clip_distance_m = goal_clip_distance_m
        self.log_range_scale = math.log1p(goal_clip_distance_m)
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
            RMSNorm(model_dim),
        )

    def forward(self, point_goal: Tensor) -> Tensor:
        if point_goal.ndim < 2 or point_goal.shape[-1] != 2:
            raise ValueError("point_goal must end with an xy dimension")
        distance = torch.linalg.vector_norm(point_goal, dim=-1, keepdim=True)
        direction = point_goal / distance.clamp_min(1e-6)
        direction = torch.where(
            distance > 1e-6, direction, torch.zeros_like(direction)
        )
        log_range = (
            torch.log1p(distance.clamp_max(self.goal_clip_distance_m))
            / self.log_range_scale
        )
        features = torch.cat((direction, log_range), dim=-1)
        return self.encoder(features)
