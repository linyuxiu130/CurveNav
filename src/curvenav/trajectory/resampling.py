"""Metric arc-length operations used to align upstream trajectory scales."""

import torch
from torch import Tensor


def path_arc_length(path: Tensor) -> Tensor:
    """Return metric arc length for each ``[N, 2]`` path in a batch."""
    _validate_path(path)
    segments = path[:, 1:] - path[:, :-1]
    return torch.linalg.vector_norm(segments, dim=-1).sum(dim=1)


def resample_path_by_arc_length(
    path: Tensor,
    num_samples: int,
) -> Tensor:
    """Resample complete metric paths at uniform arc progress.

    Inputs are dense absolute positions in robot coordinates.  Paths may have
    different lengths but must have the same input point count; every output
    preserves its own full endpoint and arc-length distribution.
    """
    _validate_path(path)
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")

    segment_length = torch.linalg.vector_norm(path[:, 1:] - path[:, :-1], dim=-1)
    cumulative = torch.cat(
        [torch.zeros_like(segment_length[:, :1]), segment_length.cumsum(dim=1)], dim=1
    )
    total = cumulative[:, -1]
    progress = torch.linspace(0.0, 1.0, num_samples, device=path.device, dtype=path.dtype)
    targets = total[:, None] * progress[None]

    upper = torch.searchsorted(cumulative.contiguous(), targets.contiguous(), right=True)
    upper = upper.clamp(1, path.shape[1] - 1)
    lower = upper - 1
    gather_index = lambda index: index.unsqueeze(-1).expand(-1, -1, 2)
    lower_point = path.gather(1, gather_index(lower))
    upper_point = path.gather(1, gather_index(upper))
    lower_distance = cumulative.gather(1, lower)
    upper_distance = cumulative.gather(1, upper)
    fraction = (targets - lower_distance) / (upper_distance - lower_distance).clamp_min(1e-8)
    result = lower_point + fraction.unsqueeze(-1) * (upper_point - lower_point)
    result[:, 0] = 0
    return result


def _validate_path(path: Tensor) -> None:
    if path.ndim != 3 or path.shape[-1] != 2 or path.shape[1] < 2:
        raise ValueError("path must have shape [B, N, 2] with N >= 2")
