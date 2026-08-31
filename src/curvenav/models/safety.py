"""Continuous observed-C-space queries for generated metric curves."""

import torch
from torch import Tensor

from curvenav.encoders.configuration import PATH_CONFIGURATION_FIELD_CHANNELS
from curvenav.physical import EXTRA_CLEARANCE_M, PATH_CONFIGURATION_QUERY_SPACING_M
from curvenav.trajectory import resample_path_to_horizon


SAFETY_CLEARANCE_M = EXTRA_CLEARANCE_M


def sample_configuration_field(
    configuration_field: Tensor,
    path: Tensor,
    planning_horizon_m: float,
) -> Tensor:
    """Bilinearly query one explicit configuration-space field contract."""
    if (
        configuration_field.ndim != 4
        or configuration_field.shape[1] != PATH_CONFIGURATION_FIELD_CHANNELS
    ):
        raise ValueError(
            "configuration field must have "
            f"{PATH_CONFIGURATION_FIELD_CHANNELS} channels"
        )
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
    inside_value = inside[..., None].to(sampled.dtype)
    return torch.cat(
        (sampled[..., :3], sampled[..., 3:] * inside_value), dim=-1
    )


def observed_clearance_loss(
    predicted_path: Tensor,
    configuration_field: Tensor,
    planning_horizon_m: float,
) -> Tensor:
    """Return one observed signed-clearance residual for every trajectory.

    This is a training-only differentiable coupling to the generator's own
    depth input, not a deployment-time barrier, candidate score, or completed
    map.  Each curve is sampled over its executed arc (never by repeating a
    short endpoint); only cells observed by raw depth contribute.  Thus the
    gradient through bilinear signed clearance directly moves a generated
    curve away from a visible footprint-inflated obstacle.
    """
    if predicted_path.ndim != 3 or predicted_path.shape[-1] != 2:
        raise ValueError("predicted_path must have shape [B,P,2]")
    if planning_horizon_m <= 0:
        raise ValueError("planning_horizon_m must be positive")
    path, active = resample_path_to_horizon(
        predicted_path,
        planning_horizon_m,
        PATH_CONFIGURATION_QUERY_SPACING_M,
    )
    predicted = sample_configuration_field(
        configuration_field,
        path,
        planning_horizon_m,
    )
    observed = active & (predicted[..., 3] > 0.5)
    deficit = torch.relu(EXTRA_CLEARANCE_M - predicted[..., 0])
    residual = (deficit / EXTRA_CLEARANCE_M).square()
    return (residual * observed.to(residual.dtype)).mean(dim=-1)
