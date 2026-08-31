"""Metric arc-length operations used to align upstream trajectory scales."""

import math

import torch
from torch import Tensor


def path_arc_length(path: Tensor) -> Tensor:
    """Return metric arc length for each ``[N, 2]`` path in a batch."""
    _validate_path(path)
    segments = path[:, 1:] - path[:, :-1]
    return torch.linalg.vector_norm(segments, dim=-1).sum(dim=1)


def resample_path_at_distance(path: Tensor, distance_m: Tensor) -> Tensor:
    """Sample every metric polyline at explicit non-negative arc distances.

    Samples past a path endpoint hold that endpoint.  This makes one physical
    query grid usable for paths of different lengths without silently dropping
    a short trajectory from a safety check.
    """
    _validate_path(path)
    if distance_m.ndim != 2 or distance_m.shape[0] != path.shape[0]:
        raise ValueError("distance_m must have shape [B,Q]")
    if (distance_m < 0).any() or not torch.isfinite(distance_m).all():
        raise ValueError("distance_m must contain finite non-negative values")
    segment_length = torch.linalg.vector_norm(path[:, 1:] - path[:, :-1], dim=-1)
    cumulative = torch.cat(
        (torch.zeros_like(segment_length[:, :1]), segment_length.cumsum(dim=1)),
        dim=1,
    )
    query = torch.minimum(distance_m, cumulative[:, -1:]).contiguous()
    lower = torch.searchsorted(cumulative.contiguous(), query, right=True) - 1
    lower = lower.clamp(0, path.shape[1] - 2)
    upper = lower + 1
    lower_distance = cumulative.gather(1, lower)
    upper_distance = cumulative.gather(1, upper)
    fraction = (query - lower_distance) / (
        upper_distance - lower_distance
    ).clamp_min(1e-8)
    index = lower[..., None].expand(-1, -1, 2)
    start = path.gather(1, index)
    end = path.gather(1, upper[..., None].expand_as(index))
    return start + fraction[..., None] * (end - start)


def resample_path_to_horizon(
    path: Tensor,
    horizon_m: float,
    spacing_m: float,
) -> tuple[Tensor, Tensor]:
    """Sample an executed path on one fixed physical query grid.

    The returned tensor has the same fixed ``Q=ceil(H / spacing)+1`` shape
    for every item, while ``active`` marks exactly the points that lie on the
    executed arc up to ``min(length, H)``.  Values after a short endpoint are
    held only to preserve the static tensor shape and must always be masked by
    ``active``.  This avoids silently treating repeated endpoints as extra
    physical observations in training or evaluation.
    """
    _validate_path(path)
    if horizon_m <= 0 or spacing_m <= 0:
        raise ValueError("horizon_m and spacing_m must be positive")
    samples = math.ceil(horizon_m / spacing_m) + 1
    distances = torch.arange(
        samples,
        device=path.device,
        dtype=path.dtype,
    )[None] * spacing_m
    distances = distances.expand(path.shape[0], -1)
    length = path_arc_length(path).clamp_max(horizon_m)
    active = distances <= length[:, None]
    points = resample_path_at_distance(
        path,
        torch.minimum(distances, length[:, None]),
    )
    return points, active


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

    total = path_arc_length(path)
    progress = torch.linspace(0.0, 1.0, num_samples, device=path.device, dtype=path.dtype)
    targets = total[:, None] * progress[None]
    return resample_path_at_distance(path, targets)


def _validate_path(path: Tensor) -> None:
    if path.ndim != 3 or path.shape[-1] != 2 or path.shape[1] < 2:
        raise ValueError("path must have shape [B, N, 2] with N >= 2")
