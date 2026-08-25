"""Conditional group-quality logits for local trajectory selection."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


TRAJECTORY_SCORER_TYPE = "conditional_trajectory_group_quality"


class TrajectoryScorer(nn.Module):
    """Return larger logits for higher-quality trajectories in one group."""

    def __init__(
        self,
        num_control_points: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_control_points = num_control_points
        self.control_projection = nn.Linear(2, model_dim)
        self.control_embedding = nn.Parameter(
            torch.empty(1, num_control_points, model_dim)
        )
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout)
            for _ in range(layers)
        )
        self.pool_norm = RMSNorm(model_dim)
        self.quality_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1),
        )
        nn.init.trunc_normal_(self.control_embedding, std=0.02)

    def forward(
        self,
        normalized_controls: Tensor,
        condition: ConditionFeatures,
    ) -> Tensor:
        if normalized_controls.ndim != 4 or normalized_controls.shape[2:] != (
            self.num_control_points,
            2,
        ):
            raise ValueError("normalized_controls must have shape [B, C, K, 2]")
        batch, candidates = normalized_controls.shape[:2]
        trajectory = normalized_controls.reshape(
            batch * candidates, self.num_control_points, 2
        )
        memory = condition.tokens[:, None].expand(
            -1, candidates, -1, -1
        ).reshape(
            batch * candidates,
            condition.tokens.shape[1],
            condition.tokens.shape[2],
        )
        trajectory = self.control_projection(trajectory) + self.control_embedding
        for block in self.blocks:
            trajectory = block(trajectory, memory)
        logits = self.quality_head(self.pool_norm(trajectory).mean(dim=1))
        return logits.reshape(batch, candidates)

    def log_probabilities(
        self,
        normalized_controls: Tensor,
        condition: ConditionFeatures,
    ) -> Tensor:
        """Normalize quality logits within one observation's candidates."""
        return torch.log_softmax(
            self(normalized_controls, condition).float(), dim=1
        )
