"""Differentiable queries of the observed local configuration space."""

from dataclasses import dataclass

import torch
from torch import Tensor


CONFIGURATION_FIELD_CHANNELS = 5


@dataclass(frozen=True)
class ConfigurationFieldQuery:
    """Observed geometry and support at continuous robot-centre positions."""

    observed_features: Tensor
    signed_clearance_m: Tensor
    support_observed: Tensor


def query_configuration_field(
    configuration_field: Tensor,
    path: Tensor,
    planning_horizon_m: float,
) -> ConfigurationFieldQuery:
    """Query a continuous clearance lower bound and interpolated observations.

    Channels are signed clearance, its planar unit gradient, ray coverage, and
    the footprint-inflated obstacle indicator.  Geometry from an unobserved
    corner is multiplied by zero before interpolation; interpolated coverage is
    retained explicitly so learned consumers can distinguish weak support from
    measured free space.
    """
    if configuration_field.ndim != 4 or configuration_field.shape[1] != 5:
        raise ValueError("configuration field must have five channels")
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError("path must have shape [B,P,2]")
    if path.shape[0] != configuration_field.shape[0] or planning_horizon_m <= 0:
        raise ValueError("path and configuration field batch dimensions must match")
    normalized = path.float() / float(planning_horizon_m)
    height, width = configuration_field.shape[-2:]
    grid_x = ((normalized[..., 0] + 1.0) * 0.5 * (width - 1)).clamp(0, width - 1)
    grid_y = ((normalized[..., 1] + 1.0) * 0.5 * (height - 1)).clamp(0, height - 1)
    x0, y0 = grid_x.floor().long(), grid_y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    wx, wy = grid_x - x0, grid_y - y0
    flat = configuration_field.float().flatten(2)

    def gather(x: Tensor, y: Tensor) -> Tensor:
        index = (y * width + x)[:, None].expand(-1, flat.shape[1], -1)
        return flat.gather(2, index).transpose(1, 2)

    corners = torch.stack(
        (gather(x0, y0), gather(x1, y0), gather(x0, y1), gather(x1, y1)),
        dim=-2,
    )
    weights = torch.stack(
        ((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy),
        dim=-1,
    )
    # Distance to a fixed obstacle set is 1-Lipschitz. Each node lower bound
    # therefore gives d(p) >= d_lower(node) - ||p-node||. Their maximum remains
    # a lower bound; bilinear distance interpolation does not have this property.
    node_x = torch.stack((x0, x1, x0, x1), dim=-1).float()
    node_y = torch.stack((y0, y0, y1, y1), dim=-1).float()
    nodes = torch.stack(
        (
            node_x * (2 * planning_horizon_m / (width - 1)) - planning_horizon_m,
            node_y * (2 * planning_horizon_m / (height - 1)) - planning_horizon_m,
        ),
        dim=-1,
    )
    corner_lower_bound = (
        corners[..., 0]
        - torch.linalg.vector_norm(path.float()[..., None, :] - nodes, dim=-1)
    )
    lower_bound = corner_lower_bound.amax(dim=-1)
    inside = (normalized.abs() <= 1).all(dim=-1)
    observed = corners[..., 3].clamp(0, 1)
    support_observed = inside & torch.where(
        weights > 0,
        observed > 0.5,
        torch.ones_like(observed, dtype=torch.bool),
    ).all(dim=-1)
    observed_geometry = (
        corners[..., :3] * observed[..., None] * weights[..., None]
    ).sum(dim=-2)
    coverage = (observed * weights).sum(dim=-1, keepdim=True)
    # Learned geometry uses only measured support, just like the BEV encoder.
    # The all-node bound above remains available for geometric evaluation.
    observed_lower_bound = corner_lower_bound.masked_fill(
        observed == 0, -torch.inf
    ).amax(dim=-1)
    observed_lower_bound = torch.where(
        (observed > 0).any(dim=-1), observed_lower_bound, 0.0
    )
    observed_geometry = torch.cat(
        (observed_lower_bound[..., None] * coverage, observed_geometry[..., 1:]),
        dim=-1,
    )
    forbidden = (corners[..., 4:5] * observed[..., None] * weights[..., None]).sum(
        dim=-2
    )
    inside_value = inside[..., None].to(lower_bound.dtype)
    return ConfigurationFieldQuery(
        observed_features=torch.cat((observed_geometry, coverage, forbidden), dim=-1)
        * inside_value,
        signed_clearance_m=lower_bound,
        support_observed=support_observed,
    )
