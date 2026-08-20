"""Candidate-set invariant trajectory critic for CurveNav stage two."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from curvenav.layers.transformer import RMSNorm, SwiGLU


@dataclass
class CriticPrediction:
    score: Tensor
    collision_logit: Tensor
    margin_logit: Tensor
    progress: Tensor
    clearance: Tensor


class CandidateCriticBlock(nn.Module):
    """Pre-norm self/cross-attention block without Flow time modulation."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = RMSNorm(model_dim)
        self.cross_norm = RMSNorm(model_dim)
        self.memory_norm = RMSNorm(model_dim)
        self.feed_forward_norm = RMSNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, memory: Tensor) -> Tensor:
        normalized = self.self_norm(tokens)
        tokens = tokens + self.dropout(
            self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        )
        normalized_memory = self.memory_norm(memory)
        tokens = tokens + self.dropout(
            self.cross_attention(
                self.cross_norm(tokens),
                normalized_memory,
                normalized_memory,
                need_weights=False,
            )[0]
        )
        return tokens + self.dropout(
            self.feed_forward(self.feed_forward_norm(tokens))
        )


class TrajectoryCritic(nn.Module):
    """Rank trajectories from geometry and the frozen policy condition memory."""

    def __init__(
        self,
        *,
        num_control_points: int,
        scale_xy: tuple[float, float],
        model_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_control_points < 4:
            raise ValueError("critic requires at least four control points")
        if model_dim % heads:
            raise ValueError("critic model_dim must be divisible by heads")
        self.num_control_points = num_control_points
        self.register_buffer("scale_xy", torch.tensor(scale_xy, dtype=torch.float32))
        self.control_projection = nn.Linear(2, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, num_control_points + 1, model_dim)
        )
        self.blocks = nn.ModuleList(
            CandidateCriticBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.score_head = nn.Linear(model_dim, 1)
        self.collision_head = nn.Linear(model_dim, 1)
        self.margin_head = nn.Linear(model_dim, 1)
        self.progress_head = nn.Linear(model_dim, 1)
        self.clearance_head = nn.Linear(model_dim, 1)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, controls_m: Tensor, condition_memory: Tensor) -> CriticPrediction:
        if controls_m.ndim != 4 or controls_m.shape[-2:] != (
            self.num_control_points,
            2,
        ):
            raise ValueError("controls_m must have shape [B, N, K, 2]")
        if condition_memory.ndim != 3 or condition_memory.shape[0] != controls_m.shape[0]:
            raise ValueError("condition_memory must have shape [B, S, D]")
        batch, candidates, _, _ = controls_m.shape
        normalized = controls_m / self.scale_xy.to(
            device=controls_m.device, dtype=controls_m.dtype
        )
        tokens = self.control_projection(normalized.flatten(0, 1))
        cls = self.cls_token.expand(batch * candidates, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.position_embedding
        memory = (
            condition_memory[:, None]
            .expand(-1, candidates, -1, -1)
            .reshape(batch * candidates, condition_memory.shape[1], -1)
        )
        for block in self.blocks:
            tokens = block(tokens, memory)
        summary = self.output_norm(tokens[:, 0]).view(batch, candidates, -1)

        def scalar(head: nn.Linear) -> Tensor:
            return head(summary).squeeze(-1)

        return CriticPrediction(
            score=scalar(self.score_head),
            collision_logit=scalar(self.collision_head),
            margin_logit=scalar(self.margin_head),
            progress=scalar(self.progress_head),
            clearance=scalar(self.clearance_head),
        )
