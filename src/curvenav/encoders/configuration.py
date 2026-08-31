"""Observed local robot configuration-space encoding."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.types import ConfigurationFeatures


CONFIGURATION_TOKEN_GRID_SIZE = 8
CONFIGURATION_TOKEN_COUNT = CONFIGURATION_TOKEN_GRID_SIZE**2
RAW_CONFIGURATION_FIELD_CHANNELS = 5
PATH_CONFIGURATION_FIELD_CHANNELS = RAW_CONFIGURATION_FIELD_CHANNELS
CONFIGURATION_ENCODER_TYPE = "metric_splat_observed_configuration_space_encoder"


def observed_configuration_features(field: Tensor, channel_dim: int) -> Tensor:
    """Remove unobserved clearance extrapolation from learned inputs.

    The distance transform is defined over a full finite grid so that a path
    query can be differentiable near a visible obstacle.  Its values outside
    raw ray coverage are *not* observations of free space.  Every learned
    consumer therefore receives geometry only through the corresponding
    coverage weight while retaining coverage itself as an explicit feature.
    ``channel_dim`` supports both raster fields and path-query tensors.
    """
    moved = field.movedim(channel_dim, -1)
    if moved.shape[-1] != RAW_CONFIGURATION_FIELD_CHANNELS:
        raise ValueError("configuration features must have five channels")
    observed = moved[..., 3:4].clamp(0.0, 1.0)
    masked = torch.cat(
        (
            moved[..., :3] * observed,
            observed,
            moved[..., 4:5] * observed,
        ),
        dim=-1,
    )
    return masked.movedim(-1, channel_dim)


class ConfigurationSpaceEncoder(nn.Module):
    """Encode only measured C-space evidence into a compact metric memory.

    The raw 64×64 field is a deterministic function of the depth history.  It
    remains unchanged for trajectory queries; this encoder only compresses it
    for attention and injects all calibrated visual frames at their SE(2)-
    aligned metric cells.  There is intentionally no unobserved-map decoder.
    """

    def __init__(
        self,
        model_dim: int,
        planning_horizon_m: float,
    ) -> None:
        super().__init__()
        if model_dim % 4:
            raise ValueError("model_dim must be divisible by four")
        if planning_horizon_m <= 0:
            raise ValueError("planning_horizon_m must be positive")
        self.planning_horizon_m = float(planning_horizon_m)
        self.encoder_32 = nn.Sequential(
            nn.Conv2d(RAW_CONFIGURATION_FIELD_CHANNELS, 32, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.encoder_16 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.encoder_8 = nn.Sequential(
            nn.Conv2d(64, model_dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.visual_projection = nn.Linear(model_dim, model_dim)
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

    def _splat_visual_to_bev(
        self,
        visual_tokens: Tensor,
        visual_points: Tensor,
        visual_valid: Tensor,
    ) -> Tensor:
        """Bilinearly lift any number of calibrated image tokens into BEV."""
        batch, token_count, model_dim = visual_tokens.shape
        if visual_points.shape != (batch, token_count, 3):
            raise ValueError("visual points must have shape [B,N,3]")
        if visual_valid.shape != (batch, token_count) or visual_valid.dtype != torch.bool:
            raise ValueError("visual validity must be boolean with shape [B,N]")
        normalized = visual_points[..., :2].float() / self.planning_horizon_m
        grid_x = (normalized[..., 0] + 1.0) * 0.5 * (
            CONFIGURATION_TOKEN_GRID_SIZE - 1
        )
        grid_y = (normalized[..., 1] + 1.0) * 0.5 * (
            CONFIGURATION_TOKEN_GRID_SIZE - 1
        )
        inside = visual_valid & (normalized.abs() <= 1.0).all(dim=-1)
        x0 = grid_x.floor().clamp(0, CONFIGURATION_TOKEN_GRID_SIZE - 1).long()
        y0 = grid_y.floor().clamp(0, CONFIGURATION_TOKEN_GRID_SIZE - 1).long()
        x1 = (x0 + 1).clamp_max(CONFIGURATION_TOKEN_GRID_SIZE - 1)
        y1 = (y0 + 1).clamp_max(CONFIGURATION_TOKEN_GRID_SIZE - 1)
        fraction_x = grid_x - x0
        fraction_y = grid_y - y0
        projected = self.visual_projection(visual_tokens)
        cells = CONFIGURATION_TOKEN_COUNT
        bev = projected.new_zeros(batch, cells, model_dim)
        mass = projected.new_zeros(batch, cells, 1)
        for x, y, weight in (
            (x0, y0, (1.0 - fraction_x) * (1.0 - fraction_y)),
            (x1, y0, fraction_x * (1.0 - fraction_y)),
            (x0, y1, (1.0 - fraction_x) * fraction_y),
            (x1, y1, fraction_x * fraction_y),
        ):
            weight = (weight * inside).to(projected.dtype)[..., None]
            index = (y * CONFIGURATION_TOKEN_GRID_SIZE + x)[..., None]
            bev.scatter_add_(1, index.expand(-1, -1, model_dim), projected * weight)
            mass.scatter_add_(1, index, weight)
        return (bev / mass.clamp_min(1e-6)).transpose(1, 2).reshape(
            batch,
            model_dim,
            CONFIGURATION_TOKEN_GRID_SIZE,
            CONFIGURATION_TOKEN_GRID_SIZE,
        )

    def forward(
        self,
        field: Tensor,
        visual_tokens: Tensor,
        visual_points: Tensor,
        visual_valid: Tensor,
    ) -> ConfigurationFeatures:
        if field.ndim != 4 or field.shape[1:] != (
            RAW_CONFIGURATION_FIELD_CHANNELS,
            64,
            64,
        ):
            raise ValueError("configuration field must have shape [B,5,64,64]")
        if (
            visual_tokens.ndim != 3
            or visual_tokens.shape[0] != field.shape[0]
            or visual_tokens.shape[-1] != self.metric_position.shape[-1]
        ):
            raise ValueError("visual tokens must have shape [B,N,model_dim]")
        observed_field = observed_configuration_features(field.float(), channel_dim=1)
        normalized = torch.cat(
            (
                observed_field[:, :1] / self.planning_horizon_m,
                observed_field[:, 1:],
            ),
            dim=1,
        )
        feature_8 = self.encoder_8(self.encoder_16(self.encoder_32(normalized)))
        feature_8 = feature_8 + self._splat_visual_to_bev(
            visual_tokens,
            visual_points,
            visual_valid,
        )
        tokens = feature_8.flatten(2).transpose(1, 2)
        if tokens.shape[1] != CONFIGURATION_TOKEN_COUNT:
            raise RuntimeError("configuration token grid does not match its contract")
        return ConfigurationFeatures(
            tokens=self.output_norm(tokens + self.metric_position.to(tokens.dtype)),
            measured_field=field.float(),
        )
