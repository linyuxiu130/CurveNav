"""Rectified Flow velocity field for planar control points."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm, TrajectoryFlowBlock
from curvenav.types import EncodedCondition


class FourierTimeEmbedding(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        half = model_dim // 2
        denominator = max(half - 1, 1)
        self.register_buffer(
            "frequencies",
            torch.exp(
                -math.log(10_000.0)
                * torch.arange(half, dtype=torch.float32)
                / denominator
            ),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.SiLU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        frequencies = self.frequencies.to(device=time.device)
        angles = time.float()[:, None] * 1_000.0 * frequencies[None]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.model_dim:
            embedding = torch.nn.functional.pad(
                embedding,
                (0, self.model_dim - embedding.shape[-1]),
            )
        return self.projection(embedding.to(dtype=time.dtype))


class TransformerTrajectoryField(nn.Module):
    """Predict the conditional flow velocity of planar control points."""

    def __init__(
        self,
        num_control_points: int = 12,
        model_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_control_points = num_control_points
        self.input_projection = nn.Linear(2, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_control_points, model_dim))
        self.time_embedding = FourierTimeEmbedding(model_dim)
        self.blocks = nn.ModuleList(
            TrajectoryFlowBlock(model_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, 2)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def prepare_condition(
        self,
        condition: EncodedCondition,
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        """Prepare the layer-specific condition K/V shared by all Flow steps."""
        return tuple(block.prepare_memory(condition.tokens) for block in self.blocks)

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        condition_key_values: tuple[tuple[Tensor, Tensor], ...],
    ) -> Tensor:
        expected = (self.num_control_points, 2)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected:
            raise ValueError(f"state must have shape [B, {expected[0]}, {expected[1]}]")
        if time.ndim != 1 or time.shape[0] != state.shape[0]:
            raise ValueError("time must have shape [B]")
        trajectory_tokens = self.input_projection(state) + self.position_embedding
        time_embedding = self.time_embedding(time)
        for block, key_value in zip(self.blocks, condition_key_values):
            trajectory_tokens = block(trajectory_tokens, key_value, time_embedding)
        return self.output_projection(self.output_norm(trajectory_tokens))
