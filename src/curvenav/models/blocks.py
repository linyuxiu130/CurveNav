"""Shared conditional Transformer block for trajectory tokens."""

from torch import Tensor, nn

from curvenav.layers import RMSNorm, SwiGLU


class ConditionalTrajectoryBlock(nn.Module):
    """Pre-norm trajectory self-attention, visual cross-attention and SwiGLU."""

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

    def forward(self, trajectory: Tensor, condition: Tensor) -> Tensor:
        normalized = self.self_norm(trajectory)
        trajectory = trajectory + self.dropout(
            self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        )
        normalized_condition = self.memory_norm(condition)
        trajectory = trajectory + self.dropout(
            self.cross_attention(
                self.query_norm(trajectory),
                normalized_condition,
                normalized_condition,
                need_weights=False,
            )[0]
        )
        return trajectory + self.dropout(
            self.feed_forward(self.feed_forward_norm(trajectory))
        )
