"""Planar PointGoal conditioning."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm


class TaskGoalEncoder(nn.Module):
    def __init__(self, model_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
            RMSNorm(model_dim),
        )

    def forward(self, task_goal: Tensor) -> Tensor:
        if task_goal.ndim != 2 or task_goal.shape[-1] != 2:
            raise ValueError("task_goal must have shape [B, 2]")
        distance = torch.linalg.vector_norm(task_goal, dim=-1, keepdim=True)
        direction = task_goal / distance.clamp_min(1e-6)
        direction = torch.where(distance > 1e-6, direction, torch.zeros_like(direction))
        features = torch.cat((direction, torch.log1p(distance)), dim=-1)
        return self.encoder(features).unsqueeze(1)
