"""Causal robot-motion tokens from metric observation poses."""

import torch
from torch import Tensor, nn
from curvenav.precision import NEURAL_DTYPE


HISTORICAL_STATE_FEATURES = "metric_xyz_rotation_columns_time"


class HistoricalMotionEncoder(nn.Module):
    """Encode past observation poses in the current robot frame.

    The final observation is the current frame and therefore has an identity
    transform.  Each preceding transform contributes one fixed-slot token.
    Missing history is excluded by the condition memory attention mask.
    """

    def __init__(
        self,
        *,
        observation_frames: int,
        translation_scale_m: float,
        model_dim: int,
    ) -> None:
        super().__init__()
        if observation_frames < 2:
            raise ValueError("motion encoding requires a past observation")
        if translation_scale_m <= 0:
            raise ValueError("translation_scale_m must be positive")
        self.state_tokens = observation_frames - 1
        self.translation_scale_m = float(translation_scale_m)
        self.projection = nn.Sequential(
            nn.Linear(10, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.slot_embedding = nn.Parameter(torch.zeros(1, self.state_tokens, model_dim))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)

    def forward(
        self,
        observation_to_current: Tensor,
        observation_age_s: Tensor,
    ) -> Tensor:
        past = observation_to_current[:, :-1].float()
        features = torch.cat(
            (
                past[..., :3, 3] / self.translation_scale_m,
                past[..., :3, :2].flatten(-2),
                observation_age_s[:, :-1, None],
            ),
            dim=-1,
        )
        with torch.autocast(device_type=features.device.type, dtype=NEURAL_DTYPE):
            encoded = self.projection(features)
        return encoded + self.slot_embedding
