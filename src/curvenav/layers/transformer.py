"""Pre-normalized Transformer blocks used by the single CurveNav model chain."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class TrajectoryFlowBlock(nn.Module):
    """AdaLN-Zero-style flow block with self- and cross-attention."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = RMSNorm(model_dim)
        self.cross_norm = RMSNorm(model_dim)
        self.memory_norm = RMSNorm(model_dim)
        self.feed_forward_norm = RMSNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 9 * model_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def prepare_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Project condition K/V once for reuse by every Flow integration step."""
        normalized = self.memory_norm(memory).transpose(0, 1)
        dimension = self.cross_attention.embed_dim
        sequence, batch, _ = normalized.shape
        heads = self.cross_attention.num_heads
        head_dimension = dimension // heads
        projection = F.linear(
            normalized,
            self.cross_attention.in_proj_weight[dimension:],
            self.cross_attention.in_proj_bias[dimension:],
        )
        projection = (
            projection.unflatten(-1, (2, dimension))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        key = projection[0].view(sequence, batch * heads, head_dimension).transpose(0, 1)
        value = projection[1].view(sequence, batch * heads, head_dimension).transpose(0, 1)
        return (
            key.view(batch, heads, sequence, head_dimension),
            value.view(batch, heads, sequence, head_dimension),
        )

    def _cross_attend(
        self,
        query: Tensor,
        memory_key_value: tuple[Tensor, Tensor],
    ) -> Tensor:
        dimension = self.cross_attention.embed_dim
        batch, sequence, _ = query.shape
        heads = self.cross_attention.num_heads
        head_dimension = dimension // heads
        projected_query = F.linear(
            query.transpose(0, 1),
            self.cross_attention.in_proj_weight[:dimension],
            self.cross_attention.in_proj_bias[:dimension],
        )
        projected_query = projected_query.view(
            sequence,
            batch * heads,
            head_dimension,
        ).transpose(0, 1)
        key, value = memory_key_value
        attended = F.scaled_dot_product_attention(
            projected_query.view(batch, heads, sequence, head_dimension),
            key,
            value,
            dropout_p=self.cross_attention.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.permute(2, 0, 1, 3).contiguous().view(
            batch * sequence,
            dimension,
        )
        attended = self.cross_attention.out_proj(attended)
        return attended.view(sequence, batch, dimension).transpose(0, 1)

    def forward(
        self,
        tokens: Tensor,
        memory_key_value: tuple[Tensor, Tensor],
        time_embedding: Tensor,
    ) -> Tensor:
        parameters = self.modulation(time_embedding).chunk(9, dim=-1)
        self_shift, self_scale, self_gate = parameters[0:3]
        cross_shift, cross_scale, cross_gate = parameters[3:6]
        ff_shift, ff_scale, ff_gate = parameters[6:9]

        normalized = self._modulate(self.self_norm(tokens), self_shift, self_scale)
        attended = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        tokens = tokens + self_gate.unsqueeze(1) * self.residual_dropout(attended)

        query = self._modulate(self.cross_norm(tokens), cross_shift, cross_scale)
        attended = self._cross_attend(query, memory_key_value)
        tokens = tokens + cross_gate.unsqueeze(1) * self.residual_dropout(attended)

        normalized = self._modulate(self.feed_forward_norm(tokens), ff_shift, ff_scale)
        update = self.feed_forward(normalized)
        return tokens + ff_gate.unsqueeze(1) * self.residual_dropout(update)
