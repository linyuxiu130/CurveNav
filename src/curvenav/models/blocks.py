"""Transformer blocks for metric trajectory--observation interaction."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from curvenav.layers import RMSNorm, SwiGLU
from curvenav.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M


class MetricPathCrossAttention(nn.Module):
    """Attend from decoded path anchors to current visual cells.

    Attention combines learned visual semantics with an additive metric bias
    computed from anchor-to-surface displacement and signed robot clearance.
    The clearance is an input feature, not a hand-written trajectory score or
    a post-processing rule.
    """

    def __init__(
        self,
        model_dim: int,
        heads: int,
        dropout: float,
        planning_horizon_m: float,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = model_dim // heads
        self.dropout = dropout
        self.planning_horizon_m = float(planning_horizon_m)
        self.footprint_clearance_m = ROBOT_FOOTPRINT_RADIUS_M + EXTRA_CLEARANCE_M
        self.query_norm = RMSNorm(model_dim)
        self.memory_norm = RMSNorm(model_dim)
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(5, model_dim // 2),
            nn.SiLU(),
            nn.Linear(model_dim // 2, heads),
        )
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.feed_forward_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        path_tokens: Tensor,
        path_points: Tensor,
        current_tokens: Tensor,
        current_points: Tensor,
        current_obstacle_valid: Tensor,
    ) -> Tensor:
        batch, anchors, model_dim = path_tokens.shape
        cells = current_tokens.shape[1]
        query = self.query_projection(self.query_norm(path_tokens)).view(
            batch, anchors, self.heads, self.head_dim
        ).transpose(1, 2)
        memory = self.memory_norm(current_tokens)
        key = self.key_projection(memory).view(
            batch, cells, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.value_projection(memory).view(
            batch, cells, self.heads, self.head_dim
        ).transpose(1, 2)

        displacement = current_points[:, None] - path_points[:, :, None]
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        scale = self.planning_horizon_m
        relative_geometry = torch.cat(
            (
                displacement / scale,
                distance / scale,
                (distance - self.footprint_clearance_m) / scale,
                current_obstacle_valid[:, None, :, None]
                .expand(-1, anchors, -1, -1)
                .to(distance.dtype),
            ),
            dim=-1,
        )
        attention_bias = self.relative_bias(relative_geometry).permute(0, 3, 1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias.to(query.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, anchors, model_dim)
        path_tokens = path_tokens + self.residual_dropout(
            self.output_projection(attended)
        )
        return path_tokens + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(path_tokens))
        )


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
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trajectory: Tensor,
        condition: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(trajectory)
        attended = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        trajectory = trajectory + self.dropout(attended)

        normalized_condition = self.memory_norm(condition)
        attended = self.cross_attention(
            self.query_norm(trajectory),
            normalized_condition,
            normalized_condition,
            need_weights=False,
        )[0]
        trajectory = trajectory + self.dropout(attended)

        normalized = self.feed_forward_norm(trajectory)
        return trajectory + self.dropout(
            self.feed_forward(normalized)
        )
