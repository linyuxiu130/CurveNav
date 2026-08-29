"""Global metric tokens for the complete local configuration space."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm


CONFIGURATION_TOKEN_GRID_SIZE = 8
CONFIGURATION_TOKEN_COUNT = CONFIGURATION_TOKEN_GRID_SIZE**2
CONFIGURATION_ENCODER_TYPE = "complete_metric_configuration_space_tokens"


class ConfigurationSpaceEncoder(nn.Module):
    """Encode every local safety cell before one-step trajectory generation.

    The MeanFlow source curve is independent of the observation, so querying
    the field only on that curve hides almost the entire map at inference.
    Three strided convolutions instead turn the complete 64x64 field into an
    8x8 metric memory while retaining local obstacle structure.
    """

    def __init__(self, model_dim: int, planning_horizon_m: float) -> None:
        super().__init__()
        if model_dim % 4:
            raise ValueError("model_dim must be divisible by four")
        if planning_horizon_m <= 0:
            raise ValueError("planning_horizon_m must be positive")
        self.planning_horizon_m = float(planning_horizon_m)
        self.projection = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, model_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.register_buffer(
            "metric_position",
            self._metric_position_encoding(model_dim),
            persistent=True,
        )
        self.output_norm = RMSNorm(model_dim)

    @staticmethod
    def _metric_position_encoding(model_dim: int) -> Tensor:
        quarter = model_dim // 4
        frequency = torch.exp(
            torch.arange(quarter, dtype=torch.float32)
            * (-math.log(10_000.0) / max(quarter - 1, 1))
        )
        axis = torch.linspace(-1.0, 1.0, CONFIGURATION_TOKEN_GRID_SIZE)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        x_phase = math.pi * x[..., None] * frequency
        y_phase = math.pi * y[..., None] * frequency
        return torch.cat(
            (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()),
            dim=-1,
        ).reshape(1, CONFIGURATION_TOKEN_COUNT, model_dim)

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim != 4 or field.shape[1:] != (5, 64, 64):
            raise ValueError("configuration field must have shape [B,5,64,64]")
        normalized = torch.cat(
            (
                field[:, :1].float() / self.planning_horizon_m,
                field[:, 1:].float(),
            ),
            dim=1,
        )
        tokens = self.projection(normalized).flatten(2).transpose(1, 2)
        if tokens.shape[1] != CONFIGURATION_TOKEN_COUNT:
            raise RuntimeError("configuration token grid does not match its contract")
        position = self.metric_position.to(tokens.dtype)
        return self.output_norm(tokens + position)
