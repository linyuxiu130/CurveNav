"""Continuous configuration-space queries and pathwise collision risk."""

import math

import torch
from torch import Tensor

from curvenav.physical import EXTRA_CLEARANCE_M


SAFETY_CLEARANCE_M = EXTRA_CLEARANCE_M
SAFETY_OBJECTIVE_TYPE = "smooth_maximum_observed_configuration_space_margin_violation"
SAFETY_SMOOTH_MAX_TEMPERATURE = 0.10


def sample_configuration_field(
    configuration_field: Tensor,
    path: Tensor,
    planning_horizon_m: float,
) -> Tensor:
    """Bilinearly query clearance, gradient, visibility, and occupancy."""
    if configuration_field.ndim != 4 or configuration_field.shape[1] != 5:
        raise ValueError("configuration field must have shape [B,5,H,W]")
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError("path must have shape [B,P,2]")
    if path.shape[0] != configuration_field.shape[0] or planning_horizon_m <= 0:
        raise ValueError("path and configuration field batch dimensions must match")
    normalized = path.float() / float(planning_horizon_m)
    height, width = configuration_field.shape[-2:]
    grid_x = ((normalized[..., 0] + 1.0) * 0.5 * (width - 1)).clamp(
        0.0, width - 1.0
    )
    grid_y = ((normalized[..., 1] + 1.0) * 0.5 * (height - 1)).clamp(
        0.0, height - 1.0
    )
    x0 = grid_x.floor().long()
    y0 = grid_y.floor().long()
    x1 = (x0 + 1).clamp_max(width - 1)
    y1 = (y0 + 1).clamp_max(height - 1)
    weight_x = grid_x - x0
    weight_y = grid_y - y0
    flat = configuration_field.float().flatten(2)

    def gather(x: Tensor, y: Tensor) -> Tensor:
        index = (y * width + x)[:, None].expand(-1, flat.shape[1], -1)
        return flat.gather(2, index).transpose(1, 2)

    top = gather(x0, y0) * (1.0 - weight_x[..., None]) + gather(
        x1, y0
    ) * weight_x[..., None]
    bottom = gather(x0, y1) * (1.0 - weight_x[..., None]) + gather(
        x1, y1
    ) * weight_x[..., None]
    sampled = top * (1.0 - weight_y[..., None]) + bottom * weight_y[..., None]
    inside = (normalized.abs() <= 1.0).all(dim=-1)
    observed = sampled[..., 3:4] * inside[..., None].to(sampled.dtype)
    forbidden = sampled[..., 4:5] * inside[..., None].to(sampled.dtype)
    return torch.cat((sampled[..., :3], observed, forbidden), dim=-1)


def configuration_space_risk_loss(
    path: Tensor,
    configuration_field: Tensor,
    planning_horizon_m: float,
) -> Tensor:
    """Penalize a smooth maximum of observed footprint-margin violations."""
    field = sample_configuration_field(
        configuration_field,
        path,
        planning_horizon_m,
    )
    clearance = field[..., 0]
    observed = field[..., 3] > 0.5
    violation = (
        torch.relu(SAFETY_CLEARANCE_M - clearance) / SAFETY_CLEARANCE_M
    ).square()
    scaled = (
        violation.masked_fill(~observed, 0.0)
        / SAFETY_SMOOTH_MAX_TEMPERATURE
    )
    path_risk = SAFETY_SMOOTH_MAX_TEMPERATURE * (
        torch.logsumexp(scaled, dim=-1) - math.log(path.shape[1])
    )
    return path_risk.mean()
