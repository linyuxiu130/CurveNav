"""Joint Transformer encoder for depth-history, PointGoal, and motion tokens."""

import torch
from torch import Tensor, nn

from curvenav.layers import EncoderBlock, RMSNorm
from curvenav.types import EncodedCondition


class ConditionTransformer(nn.Module):
    def __init__(
        self,
        model_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.goal_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.motion_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.observation_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.blocks = nn.ModuleList(
            EncoderBlock(model_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        )
        self.output_norm = RMSNorm(model_dim)
        for parameter in (self.goal_type, self.motion_type, self.observation_type):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(
        self,
        observation_tokens: Tensor,
        goal_token: Tensor,
        motion_token: Tensor,
    ) -> EncodedCondition:
        if any(token.ndim != 3 for token in (observation_tokens, goal_token, motion_token)):
            raise ValueError("condition inputs must be token tensors shaped [B, S, D]")
        if not (
            observation_tokens.shape[0]
            == goal_token.shape[0]
            == motion_token.shape[0]
        ):
            raise ValueError("condition token batch sizes must match")
        tokens = torch.cat(
            [
                goal_token + self.goal_type,
                motion_token + self.motion_type,
                observation_tokens + self.observation_type,
            ],
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens)
        return EncodedCondition(tokens=self.output_norm(tokens))
