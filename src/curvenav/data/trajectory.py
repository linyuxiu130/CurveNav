"""Shared collation of variable-length metric expert paths."""

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from curvenav.trajectory import PlanarBSplineCodec, resample_path_by_arc_length


def collate_metric_paths(
    metric_paths: list[Tensor],
    codec: PlanarBSplineCodec,
) -> tuple[Tensor, Tensor]:
    if not metric_paths or any(path.ndim != 2 or path.shape[-1] != 2 for path in metric_paths):
        raise ValueError("metric_paths must be a non-empty list of [N,2] tensors")
    lengths = torch.tensor([path.shape[0] for path in metric_paths])
    if torch.any(lengths < 2):
        raise ValueError("every metric path must contain at least two points")
    padded = pad_sequence(metric_paths, batch_first=True)
    padding = torch.arange(padded.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    endpoints = torch.stack([path[-1] for path in metric_paths])
    padded = torch.where(padding.unsqueeze(-1), endpoints.unsqueeze(1), padded)
    canonical = resample_path_by_arc_length(padded, num_samples=codec.num_path_points)
    return canonical, codec.encode(canonical)
