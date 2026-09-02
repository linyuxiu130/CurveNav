"""Project variable-length expert paths into CurveNav's production codec."""

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from curvenav.trajectory import (
    IncrementalBSplineTrajectory,
    resample_path_by_arc_length,
)


MAXIMUM_EXPERT_PROJECTION_ADE_RATIO = 0.2


def collate_metric_paths(
    metric_paths: list[Tensor],
    codec: IncrementalBSplineTrajectory,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return metric controls, decoded targets, and source projection error."""
    if not metric_paths or any(
        path.ndim != 2 or path.shape[-1] != 2 for path in metric_paths
    ):
        raise ValueError("metric_paths must be a non-empty list of [N,2] tensors")
    lengths = torch.tensor([path.shape[0] for path in metric_paths])
    if torch.any(lengths < 2):
        raise ValueError("every metric path must contain at least two points")
    if any(torch.count_nonzero(path[0]).item() for path in metric_paths):
        raise ValueError("every metric path must start at the robot origin")
    padded = pad_sequence(metric_paths, batch_first=True)
    padding = torch.arange(padded.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    endpoints = torch.stack([path[-1] for path in metric_paths])
    padded = torch.where(padding.unsqueeze(-1), endpoints.unsqueeze(1), padded)
    source_reference = resample_path_by_arc_length(
        padded,
        num_samples=codec.num_path_points,
    )
    values, reference_path = codec.project_expert(source_reference)
    projection_error = torch.linalg.vector_norm(
        reference_path - source_reference,
        dim=-1,
    ).mean(1)
    return values, reference_path, projection_error
