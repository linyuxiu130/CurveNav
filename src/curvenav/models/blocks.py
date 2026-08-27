"""Adaptive Transformer blocks for ordered trajectory queries."""

from torch import Tensor, nn

from curvenav.layers import RMSNorm, SwiGLU


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class ConditionalTrajectoryBlock(nn.Module):
    """Bidirectional trajectory attention with adaRMS-Zero conditioning.

    The end-to-end learned route summary modulates every residual branch.
    Geometry remains a token sequence and enters through cross-attention, so
    spatial information is not collapsed into the global modulation vector.
    """

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
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 9 * model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        trajectory: Tensor,
        condition: Tensor,
        condition_padding_mask: Tensor,
        modulation: Tensor,
    ) -> Tensor:
        (
            self_shift,
            self_scale,
            self_gate,
            cross_shift,
            cross_scale,
            cross_gate,
            feed_shift,
            feed_scale,
            feed_gate,
        ) = self.modulation(modulation).chunk(9, dim=-1)

        normalized = _modulate(self.self_norm(trajectory), self_shift, self_scale)
        attended = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        trajectory = trajectory + self_gate[:, None] * self.dropout(attended)

        normalized_condition = self.memory_norm(condition)
        query = _modulate(self.query_norm(trajectory), cross_shift, cross_scale)
        attended = self.cross_attention(
            query,
            normalized_condition,
            normalized_condition,
            key_padding_mask=condition_padding_mask,
            need_weights=False,
        )[0]
        trajectory = trajectory + cross_gate[:, None] * self.dropout(attended)

        normalized = _modulate(
            self.feed_forward_norm(trajectory), feed_shift, feed_scale
        )
        return trajectory + feed_gate[:, None] * self.dropout(
            self.feed_forward(normalized)
        )
