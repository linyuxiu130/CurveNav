"""Differentiable queries of the observed local configuration space."""

from dataclasses import dataclass

import torch
from torch import Tensor


CONFIGURATION_FIELD_CHANNELS = 5




@dataclass(frozen=True)
class ConfigurationFieldQuery:
    """Observed geometry and support at continuous robot-centre positions."""

    observed_features: Tensor
    estimated_clearance_m: Tensor
    clearance_lower_bound_m: Tensor
    clearance_upper_bound_m: Tensor
    support_observed: Tensor


def _configuration_corners(
    configuration_field: Tensor,
    path: Tensor,
    planning_horizon_m: float,
):
    """Gather candidate pixels directly from each shared scene raster."""
    if configuration_field.ndim != 4 or configuration_field.shape[1] not in (5, 6, 7):
        raise ValueError("configuration field requires five geometry channels and optional free evidence")
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError("path must have shape [B,P,2]")
    if configuration_field.shape[0]<1 or path.shape[0] % configuration_field.shape[0] or planning_horizon_m <= 0:
        raise ValueError("paths must contain the same number of candidates for every configuration field")
    normalized = path.float() / float(planning_horizon_m)
    height, width = configuration_field.shape[-2:]
    grid_x = ((normalized[..., 0] + 1.0) * 0.5 * (width - 1)).clamp(0, width - 1)
    grid_y = ((normalized[..., 1] + 1.0) * 0.5 * (height - 1)).clamp(0, height - 1)
    x0, y0 = grid_x.floor().long(), grid_y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    wx, wy = grid_x - x0, grid_y - y0
    flat = configuration_field.float().flatten(2)

    def gather(x: Tensor, y: Tensor) -> Tensor:
        index = (y * width + x).reshape(flat.shape[0],-1)[:, None].expand(-1, flat.shape[1], -1)
        return flat.gather(2, index).transpose(1, 2).reshape(*path.shape[:2],flat.shape[1])

    corners = torch.stack(
        (gather(x0, y0), gather(x1, y0), gather(x0, y1), gather(x1, y1)),
        dim=-2,
    )
    weights = torch.stack(
        ((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy),
        dim=-1,
    )
    return corners,weights,(normalized.abs()<=1).all(dim=-1),(x0,x1,y0,y1)


def _observed_features(corners,weights,inside,represented_geometry):
    observed=corners[...,3].clamp(0,1)
    geometry=(corners[...,:3]*(1 if represented_geometry else observed[...,None])*weights[...,None]).sum(-2)
    coverage=(observed*weights).sum(-1,keepdim=True)
    overlap=(corners[...,4:5]*observed[...,None]*weights[...,None]).sum(-2)
    free=(corners[...,5:]*weights[...,None]).sum(-2)
    return torch.cat((geometry,coverage,overlap,free),-1)*inside[...,None].to(corners.dtype)


def sample_configuration_features(configuration_field,path,planning_horizon_m,*,represented_geometry=False):
    """Network inputs only; distance bounds and support diagnostics are computed on request."""
    corners,weights,inside,_=_configuration_corners(configuration_field,path,planning_horizon_m)
    return _observed_features(corners,weights,inside,represented_geometry)


def query_configuration_field(
    configuration_field: Tensor,
    path: Tensor,
    planning_horizon_m: float,
    *, represented_geometry: bool = False,
) -> ConfigurationFieldQuery:
    """Interpolate observed features and bound represented sites' continuous distance.

    Every scene serves an equal number of consecutive candidate paths. Unobserved
    corners contribute no geometry unless represented_geometry is set; support
    still requires observed evidence, so unseen space is never certified free.
    """
    corners,weights,inside,(x0,x1,y0,y1)=_configuration_corners(configuration_field,path,planning_horizon_m)
    height,width=configuration_field.shape[-2:]
    # Raster-site distance is 1-Lipschitz: d(node) +/- ||p-node|| bounds d(p).
    # Bilinear interpolation is only an estimate, not either bound.
    node_x = torch.stack((x0, x1, x0, x1), dim=-1).float()
    node_y = torch.stack((y0, y0, y1, y1), dim=-1).float()
    nodes = torch.stack(
        (
            node_x * (2 * planning_horizon_m / (width - 1)) - planning_horizon_m,
            node_y * (2 * planning_horizon_m / (height - 1)) - planning_horizon_m,
        ),
        dim=-1,
    )
    distance_to_node = torch.linalg.vector_norm(
        path.float()[..., None, :] - nodes, dim=-1
    )
    corner_lower_bound = corners[..., 0] - distance_to_node
    corner_upper_bound = corners[..., 0] + distance_to_node
    lower_bound = corner_lower_bound.amax(dim=-1)
    upper_bound = corner_upper_bound.amin(dim=-1)
    observed = corners[..., 3].clamp(0, 1)
    support_observed = inside & torch.where(
        weights > 0,
        observed > 0.5,
        torch.ones_like(observed, dtype=torch.bool),
    ).all(dim=-1)
    return ConfigurationFieldQuery(
        observed_features=_observed_features(corners,weights,inside,represented_geometry),
        estimated_clearance_m=(corners[..., 0] * weights).sum(dim=-1),
        clearance_lower_bound_m=lower_bound,
        clearance_upper_bound_m=upper_bound,
        support_observed=support_observed,
    )
