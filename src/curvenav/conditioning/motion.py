"""Causal robot-motion tokens from metric observation poses."""

import torch
from torch import Tensor, nn


HISTORICAL_STATE_FEATURES = "normalized_xy_sine_cosine"


class HistoricalMotionEncoder(nn.Module):
    """Encode past observation poses in the current robot frame.

    The final observation is the current frame and therefore has an identity
    transform.  Each preceding transform contributes one fixed-slot token.
    Missing history uses a learned null value so batching remains dense without
    pretending that padding is a stationary observation.
    """

    def __init__(
        self,
        *,
        observation_frames: int,
        history_horizon_m: float,
        model_dim: int,
    ) -> None:
        super().__init__()
        if observation_frames < 2:
            raise ValueError("motion encoding requires a past observation")
        if history_horizon_m <= 0:
            raise ValueError("history_horizon_m must be positive")
        self.state_tokens = observation_frames - 1
        self.history_horizon_m = float(history_horizon_m)
        self.projection = nn.Sequential(
            nn.Linear(4, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, self.state_tokens, model_dim)
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.null_token, std=0.02)

    def forward(
        self,
        observation_to_current: Tensor,
        observation_valid: Tensor,
    ) -> Tensor:
        if (
            observation_to_current.ndim != 3
            or observation_to_current.shape[-1] != 4
        ):
            raise ValueError("observation transforms must have shape [B,F,4]")
        if observation_valid.shape != observation_to_current.shape[:2]:
            raise ValueError("observation validity must have shape [B,F]")
        past = observation_to_current[:, :-1].float()
        features = torch.cat(
            (
                past[..., :2] / self.history_horizon_m,
                past[..., 2:4],
            ),
            dim=-1,
        )
        encoded = self.projection(features)
        valid = observation_valid[:, :-1, None]
        encoded = torch.where(
            valid,
            encoded,
            self.null_token.expand(encoded.shape[0], self.state_tokens, -1),
        )
        return encoded + self.slot_embedding
