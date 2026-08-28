"""Pre-normalized Transformer blocks used by the single CurveNav model chain."""

import torch
from torch import Tensor, nn


def unit_rms(value: Tensor, epsilon: float = 1e-12) -> Tensor:
    """Remove an unidentifiable query radius using float32 statistics."""
    inverse_rms = (
        value.float().square().mean(dim=-1, keepdim=True).clamp_min(epsilon).rsqrt()
    )
    return value * inverse_rms.to(dtype=value.dtype)


class RMSNorm(nn.Module):
    """RMS normalization with float32 statistics for mixed-precision stability."""

    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, value: Tensor) -> Tensor:
        scale = value.float().square().mean(dim=-1, keepdim=True).add(self.epsilon).rsqrt()
        normalized = value * scale.to(dtype=value.dtype)
        return normalized * self.weight.to(dtype=value.dtype)


class SwiGLU(nn.Module):
    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        parameter_matched = int(8 * model_dim / 3)
        hidden_dim = 8 * ((parameter_matched + 7) // 8)
        self.input_projection = nn.Linear(model_dim, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        gate, content = self.input_projection(value).chunk(2, dim=-1)
        return self.output_projection(self.dropout(torch.nn.functional.silu(gate) * content))


class EncoderBlock(nn.Module):
    """Joint condition-encoding block using optimized PyTorch attention kernels."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        tokens = tokens + self.residual_dropout(attended)
        return tokens + self.residual_dropout(self.feed_forward(self.feed_forward_norm(tokens)))
