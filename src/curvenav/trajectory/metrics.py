"""Metric-space diagnostics for decoded planar trajectories."""

import torch
from torch import Tensor

from .resampling import path_arc_length


def path_scale_summary(dense_path: Tensor) -> dict[str, Tensor]:
    """Return scale diagnostics used to compare datasets and predictions."""
    if dense_path.ndim != 3 or dense_path.shape[-1] != 2:
        raise ValueError("dense_path must have shape [B, N, 2]")
    return {
        "arc_length_m": path_arc_length(dense_path),
        "forward_extent_m": dense_path[..., 0].amax(dim=1) - dense_path[..., 0].amin(dim=1),
        "lateral_extent_m": dense_path[..., 1].amax(dim=1) - dense_path[..., 1].amin(dim=1),
        "endpoint_distance_m": torch.linalg.vector_norm(dense_path[:, -1], dim=-1),
    }
