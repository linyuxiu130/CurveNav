"""Conditioned executable-curve source for self-consistent Flow Matching."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


CURVE_PROPOSAL_TYPE = "ordered_route_conditioned_bounded_curvature_source"


class ConditionedCurveProposal(nn.Module):
    """Predict the deterministic curve state used at both train and inference."""

    def __init__(
        self,
        curve_tokens: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.curve_tokens = curve_tokens
        self.token_embedding = nn.Parameter(
            torch.empty(1, curve_tokens, model_dim)
        )
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout)
            for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.coordinate_projection = nn.Linear(model_dim, 2)
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        nn.init.zeros_(self.coordinate_projection.bias)

    def forward(
        self,
        condition: ConditionFeatures,
        free_mask: Tensor,
    ) -> Tensor:
        batch = condition.tokens.shape[0]
        if free_mask.shape != (batch, self.curve_tokens, 2):
            raise ValueError("free_mask must have shape [B,T,2]")
        proposal = self.token_embedding.expand(batch, -1, -1)
        for block in self.blocks:
            proposal = block(
                proposal,
                condition.tokens,
                condition.route_token,
            )
        return self.coordinate_projection(self.output_norm(proposal)) * free_mask
