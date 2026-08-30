"""Transformer blocks for trajectory--condition interaction."""

import torch
from torch import Tensor, nn
from torch.nn import functional

from curvenav.layers import RMSNorm, SwiGLU


ProjectedCondition = tuple[Tensor, Tensor]


class ReusableConditionCrossAttention(nn.Module):
    """Cross-attention whose condition K/V projection is reusable."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.embed_dim = model_dim
        self.num_heads = heads
        self.head_dim = model_dim // heads
        self.dropout = float(dropout)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * model_dim, model_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * model_dim))
        self.out_proj = nn.Linear(model_dim, model_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def project_condition(self, condition: Tensor) -> ProjectedCondition:
        projected = functional.linear(
            condition,
            self.in_proj_weight[self.embed_dim :],
            self.in_proj_bias[self.embed_dim :],
        )
        key, value = projected.unflatten(-1, (2, self.embed_dim)).unbind(-2)
        key = key.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.num_heads, self.head_dim)).transpose(
            1, 2
        )
        return key, value

    def forward(
        self,
        query: Tensor,
        projected_condition: ProjectedCondition,
    ) -> Tensor:
        projected_query = functional.linear(
            query,
            self.in_proj_weight[: self.embed_dim],
            self.in_proj_bias[: self.embed_dim],
        )
        projected_query = projected_query.unflatten(
            -1, (self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key, value = projected_condition
        attended = functional.scaled_dot_product_attention(
            projected_query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).flatten(-2)
        return self.out_proj(attended)


class ConditionalTrajectoryBlock(nn.Module):
    """Ordered self-attention followed by direct condition cross-attention."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = RMSNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.query_norm = RMSNorm(model_dim)
        self.memory_norm = RMSNorm(model_dim)
        self.cross_attention = ReusableConditionCrossAttention(
            model_dim, heads, dropout
        )
        self.feed_forward_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trajectory: Tensor,
        projected_condition: ProjectedCondition,
    ) -> Tensor:
        normalized = self.self_norm(trajectory)
        attended = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        trajectory = trajectory + self.dropout(attended)

        attended = self.cross_attention(
            self.query_norm(trajectory),
            projected_condition,
        )
        trajectory = trajectory + self.dropout(attended)

        normalized = self.feed_forward_norm(trajectory)
        return trajectory + self.dropout(self.feed_forward(normalized))

    def project_condition(self, condition: Tensor) -> ProjectedCondition:
        return self.cross_attention.project_condition(self.memory_norm(condition))
