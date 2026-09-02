"""Differentiable observed-clearance objective for the deployed curve."""

import torch
from torch import Tensor

from curvenav.configuration_space import query_configuration_field
from curvenav.physical import EXTRA_CLEARANCE_M, PATH_CONFIGURATION_QUERY_SPACING_M
from curvenav.trajectory import path_arc_length, resample_path_to_horizon


def observed_clearance_loss(
    path: Tensor,
    configuration_field: Tensor,
    planning_horizon_m: float,
) -> Tensor:
    """Return dimensionless visible C-space risk for each generated path.

    The fixed physical query grid and fixed denominator prevent coverage or
    path length from changing the scale of the objective.  Strict four-corner
    support ensures that EDT values extrapolated through unknown space never
    contribute a geometric gradient.
    """
    # Preserve the physical curve in the forward pass while removing the
    # uniform radial-scale direction from this objective's gradient.  MeanFlow
    # alone therefore supervises path length; clearance changes turning shape
    # instead of buying lower risk by uniformly shrinking the trajectory.
    arc_length = path_arc_length(path).clamp_min(1e-6)
    shape_path = path * (arc_length.detach() / arc_length)[:, None, None]
    dense_path, active = resample_path_to_horizon(
        shape_path,
        planning_horizon_m,
        PATH_CONFIGURATION_QUERY_SPACING_M,
    )
    query = query_configuration_field(
        configuration_field,
        dense_path,
        planning_horizon_m,
    )
    supported = active & query.support_observed
    normalized_deficit = torch.relu(
        (EXTRA_CLEARANCE_M - query.signed_clearance_m) / EXTRA_CLEARANCE_M
    )
    return (
        normalized_deficit.square() * supported.to(normalized_deficit.dtype)
    ).mean(dim=-1)
