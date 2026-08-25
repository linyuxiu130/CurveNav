"""Metric-space diagnostics for decoded planar trajectories."""

import torch
from torch import Tensor

from .resampling import path_arc_length


def path_scale_summary(path: Tensor) -> dict[str, Tensor]:
    """Return scale diagnostics used to compare datasets and predictions."""
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError("path must have shape [B, N, 2]")
    return {
        "arc_length_m": path_arc_length(path),
        "forward_extent_m": path[..., 0].amax(dim=1) - path[..., 0].amin(dim=1),
        "lateral_extent_m": path[..., 1].amax(dim=1) - path[..., 1].amin(dim=1),
        "endpoint_distance_m": torch.linalg.vector_norm(path[:, -1], dim=-1),
    }
