"""Transformer blocks for trajectory--condition interaction."""

import torch
from torch import Tensor, nn
from torch.nn import functional

from curvenav.layers import RMSNorm, SwiGLU
from curvenav.types import ConditionFeatures


ProjectedCondition = tuple[Tensor, Tensor, Tensor]


class ReusableConditionCrossAttention(nn.Module):
    """Cross-attention whose condition K/V projection is reusable."""

    def __init__(
        self,
        model_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
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
        self.relative_bias = nn.Sequential(
            nn.Linear(7, model_dim // 4),
            nn.SiLU(),
            nn.Linear(model_dim // 4, heads),
        )
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def project_condition(self, condition: ConditionFeatures) -> ProjectedCondition:
        tokens = condition.tokens
        token_valid = condition.token_valid
        if token_valid.shape != tokens.shape[:2] or token_valid.dtype != torch.bool:
            raise ValueError("condition validity must be boolean [B,N]")
        projected = functional.linear(
            tokens,
            self.in_proj_weight[self.embed_dim :],
            self.in_proj_bias[self.embed_dim :],
        )
        key, value = projected.unflatten(-1, (2, self.embed_dim)).unbind(-2)
        key = key.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        return key, value, token_valid[:, None, None]

    def forward(
        self,
        query: Tensor,
        projected_condition: ProjectedCondition,
        pair_geometry: Tensor,
    ) -> Tensor:
        projected_query = functional.linear(
            query,
            self.in_proj_weight[: self.embed_dim],
            self.in_proj_bias[: self.embed_dim],
        )
        projected_query = projected_query.unflatten(
            -1, (self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key, value, token_valid = projected_condition
        # The trainable primal caches K/V under autocast, while the stopped
        # MeanFlow JVP deliberately evaluates the same decoder in float32.
        # Promote the reusable memory at the attention boundary so SDPA sees
        # one numerical domain; under the primal this is an allocation-free
        # no-op because query and memory already share the autocast dtype.
        key = key.to(dtype=projected_query.dtype)
        value = value.to(dtype=projected_query.dtype)
        expected = (*query.shape[:2], key.shape[-2], 7)
        if pair_geometry.shape != expected:
            raise ValueError("path-relative attention requires [B,Q,N,7] geometry")
        attention_mask = self.relative_bias(
            pair_geometry.to(dtype=projected_query.dtype)
        ).permute(0, 3, 1, 2)
        attention_mask = attention_mask.masked_fill(~token_valid, float("-inf"))
        attended = functional.scaled_dot_product_attention(
            projected_query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).flatten(-2)
        return self.out_proj(attended)


class ConditionalTrajectoryBlock(nn.Module):
    """Update trajectory tokens only through self- and metric scene attention."""

    def __init__(
        self,
        model_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_norm = RMSNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.query_norm = RMSNorm(model_dim)
        self.memory_norm = RMSNorm(model_dim)
        self.cross_attention = ReusableConditionCrossAttention(
            model_dim,
            heads,
            dropout,
        )
        self.feed_forward_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trajectory: Tensor,
        projected_condition: ProjectedCondition,
        pair_geometry: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(trajectory)
        attended = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        trajectory = trajectory + self.dropout(attended)

        attended = self.cross_attention(
            self.query_norm(trajectory),
            projected_condition,
            pair_geometry,
        )
        trajectory = trajectory + self.dropout(attended)

        normalized = self.feed_forward_norm(trajectory)
        return trajectory + self.dropout(self.feed_forward(normalized))

    def project_condition(self, condition: ConditionFeatures) -> ProjectedCondition:
        return self.cross_attention.project_condition(
            ConditionFeatures(
                tokens=self.memory_norm(condition.tokens),
                token_valid=condition.token_valid,
                metric_position=condition.metric_position,
                surface_hit=condition.surface_hit,
                frame_age=condition.frame_age,
                motion_token=condition.motion_token,
                goal_reference=condition.goal_reference,
                configuration_field=condition.configuration_field,
            )
        )
