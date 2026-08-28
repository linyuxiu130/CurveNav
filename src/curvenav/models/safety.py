"""Differentiable configuration-space supervision for generated trajectories."""

import torch
from torch import Tensor

from curvenav.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M


SAFETY_CLEARANCE_M = ROBOT_FOOTPRINT_RADIUS_M + EXTRA_CLEARANCE_M
SAFETY_OBJECTIVE_TYPE = (
    "predicted_clean_curve_current_depth_configuration_space_clearance_hinge"
)


def configuration_space_clearance_loss(
    path: Tensor,
    obstacle_points: Tensor,
    obstacle_valid: Tensor,
) -> Tensor:
    """Return mean squared normalized robot-footprint clearance violation."""
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError("path must have shape [B,P,2]")
    if obstacle_points.ndim != 3 or obstacle_points.shape[-1] != 2:
        raise ValueError("obstacle_points must have shape [B,O,2]")
    if obstacle_valid.shape != obstacle_points.shape[:2]:
        raise ValueError("obstacle_valid must have shape [B,O]")
    if (
        obstacle_valid.dtype != torch.bool
        or path.shape[0] != obstacle_points.shape[0]
    ):
        raise ValueError("obstacle mask and batch dimensions must match")

    distance = torch.cdist(path.float(), obstacle_points.float())
    distance = distance.masked_fill(~obstacle_valid[:, None], torch.inf)
    nearest = distance.amin(dim=-1)
    violation = torch.relu(SAFETY_CLEARANCE_M - nearest) / SAFETY_CLEARANCE_M
    return violation.square().mean()
