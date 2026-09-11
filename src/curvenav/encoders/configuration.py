"""Target-independent fusion of learned depth evidence and observed C-space."""

import math

import torch
from torch import Tensor, nn
from curvenav.precision import NEURAL_DTYPE
from torch.nn import functional as F

from curvenav.layers import RMSNorm
from curvenav.types import ConfigurationFeatures


RAW_CONFIGURATION_FIELD_CHANNELS = 5
CONFIGURATION_ENCODER_TYPE = "observed_cspace_visual_metric_bev"


def observed_configuration_features(field: Tensor) -> Tensor:
    """Mask EDT extrapolation outside measured support while retaining unknownness."""
    if field.ndim != 4 or field.shape[1] != RAW_CONFIGURATION_FIELD_CHANNELS:
        raise ValueError("configuration field must have shape [B,5,H,W]")
    observed = field[:, 3:4].clamp(0.0, 1.0)
    return torch.cat(
        (
            field[:, :3] * observed,
            observed,
            field[:, 4:5] * observed,
        ),
        dim=1,
    )


class ConfigurationSpaceEncoder(nn.Module):
    """Fuse aligned depth features into one calibrated robot-centric BEV."""

    def __init__(
        self,
        *,
        model_dim: int,
        planning_horizon_m: float,
        grid_size: int,
    ) -> None:
        super().__init__()
        if model_dim % 4:
            raise ValueError("model_dim must be divisible by four")
        if planning_horizon_m <= 0 or grid_size != 16:
            raise ValueError("configuration memory must be a positive-scale 16x16 grid")
        self.model_dim = model_dim
        self.planning_horizon_m = float(planning_horizon_m)
        self.grid_size = grid_size
        self.token_count = grid_size**2
        self.field_encoder = nn.Sequential(
            nn.Conv2d(RAW_CONFIGURATION_FIELD_CHANNELS, 64, 2, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, model_dim, 2, stride=2),
            nn.SiLU(),
            nn.Conv2d(model_dim, model_dim, 3, padding=1),
        )
        self.visual_projection = nn.Linear(model_dim, model_dim)
        position, encoding = self._metric_grid(model_dim, grid_size, planning_horizon_m)
        self.register_buffer("metric_position", position, persistent=True)
        self.register_buffer("metric_encoding", encoding, persistent=True)
        self.output_norm = RMSNorm(model_dim)

    @staticmethod
    def _metric_grid(
        model_dim: int,
        grid_size: int,
        planning_horizon_m: float,
    ) -> tuple[Tensor, Tensor]:
        axis = ((torch.arange(grid_size) * 4 + 1.5) / 63 * 2 - 1) * planning_horizon_m
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        position = torch.stack((x, y, torch.zeros_like(x)), dim=-1).reshape(
            1, grid_size**2, 3
        )
        quarter = model_dim // 4
        frequency = torch.exp(
            torch.arange(quarter, dtype=torch.float32)
            * (-math.log(10_000.0) / max(quarter - 1, 1))
        )
        x_phase = math.pi * x[..., None] / planning_horizon_m * frequency
        y_phase = math.pi * y[..., None] / planning_horizon_m * frequency
        encoding = torch.cat(
            (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
        ).reshape(1, grid_size**2, model_dim)
        return position, encoding

    def _splat_visual(
        self,
        tokens: Tensor,
        points: Tensor,
        valid: Tensor,
    ) -> Tensor:
        batch, count, model_dim = tokens.shape
        if points.shape != (batch, count, 3):
            raise ValueError("visual points must have shape [B,N,3]")
        if valid.shape != (batch, count) or valid.dtype != torch.bool:
            raise ValueError("visual validity must be boolean [B,N]")
        normalized = points[..., :2].float() / self.planning_horizon_m
        coordinate = ((normalized + 1.0) * 0.5 * 63 - 1.5) / 4
        coordinate = coordinate.clamp(0, self.grid_size - 1)
        inside = valid & (normalized.abs() <= 1.0).all(dim=-1)
        x0 = coordinate[..., 0].floor().clamp(0, self.grid_size - 1).long()
        y0 = coordinate[..., 1].floor().clamp(0, self.grid_size - 1).long()
        x1 = (x0 + 1).clamp_max(self.grid_size - 1)
        y1 = (y0 + 1).clamp_max(self.grid_size - 1)
        wx, wy = coordinate[..., 0] - x0, coordinate[..., 1] - y0
        # All selected frames contribute to one cell. Accumulate weighted
        # sums and mass in FP32, outside the BF16 neural projection.
        with torch.autocast(device_type=tokens.device.type, dtype=NEURAL_DTYPE):
            projected = self.visual_projection(tokens).float()
        bev = projected.new_zeros(batch, self.token_count, model_dim)
        mass = projected.new_zeros(batch, self.token_count, 1)
        for x, y, weight in (
            (x0, y0, (1 - wx) * (1 - wy)),
            (x1, y0, wx * (1 - wy)),
            (x0, y1, (1 - wx) * wy),
            (x1, y1, wx * wy),
        ):
            weight = (weight * inside).to(projected.dtype)[..., None]
            index = (y * self.grid_size + x)[..., None]
            bev.scatter_add_(1, index.expand(-1, -1, model_dim), projected * weight)
            mass.scatter_add_(1, index, weight)
        return bev / mass.clamp_min(1e-6)

    def forward(
        self,
        field: Tensor,
        visual_tokens: Tensor,
        visual_points: Tensor,
        visual_valid: Tensor,
    ) -> ConfigurationFeatures:
        if field.shape[1:] != (RAW_CONFIGURATION_FIELD_CHANNELS, 64, 64):
            raise ValueError("configuration field must have shape [B,5,64,64]")
        canonical = observed_configuration_features(field.float())
        normalized = canonical.clone()
        normalized[:, 0] /= self.planning_horizon_m
        with torch.autocast(device_type=field.device.type, dtype=NEURAL_DTYPE):
            raster = self.field_encoder(normalized).flatten(2).transpose(1, 2)
        visual = self._splat_visual(visual_tokens, visual_points, visual_valid)
        tokens = self.output_norm(
            raster + visual + self.metric_encoding.to(dtype=raster.dtype)
        )
        observed_fraction = F.adaptive_avg_pool2d(
            canonical[:, 3:4], (self.grid_size, self.grid_size)
        ).flatten(1)
        return ConfigurationFeatures(
            tokens=tokens,
            metric_position=self.metric_position.expand(field.shape[0], -1, -1),
            observed_fraction=observed_fraction,
        )
