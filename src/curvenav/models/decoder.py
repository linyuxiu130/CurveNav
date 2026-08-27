"""One ordered Transformer decoder for the executable local curve."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


TRAJECTORY_DECODER_TYPE = "ordered_route_conditioned_bounded_curvature_decoder"
CURVE_COORDINATE_SCALE = 8.0


class OrderedCurveDecoder(nn.Module):
    """Map route-conditioned curve queries directly to executable coordinates."""

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
        self.token_embedding = nn.Parameter(torch.empty(1, curve_tokens, model_dim))
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.coordinate_projection = nn.Linear(model_dim, 1)
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        nn.init.zeros_(self.coordinate_projection.bias)

    @staticmethod
    def normalize_curve_coordinates(curve_coordinates: Tensor) -> Tensor:
        """Map compact geometric coordinates to an O(1) regression target."""
        return curve_coordinates * CURVE_COORDINATE_SCALE

    @staticmethod
    def denormalize_curve_coordinates(normalized_coordinates: Tensor) -> Tensor:
        """Recover the exact geometric coordinates consumed by the codec."""
        return normalized_coordinates / CURVE_COORDINATE_SCALE

    def forward(
        self,
        condition: ConditionFeatures,
    ) -> Tensor:
        batch = condition.tokens.shape[0]
        trajectory = self.token_embedding.expand(batch, -1, -1)
        for block in self.blocks:
            trajectory = block(
                trajectory,
                condition.tokens,
                condition.route_token,
            )
        return self.coordinate_projection(self.output_norm(trajectory)).squeeze(-1)
