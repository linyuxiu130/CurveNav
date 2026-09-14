"""Supervised route utility: geometry is the teacher, never an inference input."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from curvenav.data.goal_distance import GoalDistanceQuery
from curvenav.data.privileged import SourceConfigurationSpaceQuery, SourcePathQuery
from curvenav.models.policy import CurveNavTrainingOutput
from curvenav.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M
from curvenav.trajectory import path_arc_length


CRITIC_TARGET = "geodesic_progress_minus_half_length_contact_and_clearance_margin"


@torch.no_grad()
def candidate_query(
    query: SourceConfigurationSpaceQuery,
    paths: Tensor,
    batch: dict[str, Tensor],
    planning_horizon_m: float,
) -> tuple[SourcePathQuery, Tensor]:
    count = paths.shape[1]
    flat = paths.flatten(0, 1).float()
    horizon = max(planning_horizon_m, path_arc_length(flat).max().item())
    result = query.query(
        flat,
        batch["source_grid_index"].repeat_interleave(count),
        batch["source_origin_xy"].repeat_interleave(count, dim=0),
        batch["source_yaw_rad"].repeat_interleave(count),
        horizon,
    )
    # OOB semantics must not depend on another candidate's length.
    clearance = result.clearance_m.masked_fill(~result.in_world_bounds, -planning_horizon_m)
    return result, clearance


@dataclass
class CandidateUtility:
    score: Tensor
    clearance_m: Tensor
    progress_m: Tensor


class RouteUtilityTeacher:
    def __init__(self, source_query: SourceConfigurationSpaceQuery, planning_horizon_m: float):
        self.source_query = source_query
        self.horizon = planning_horizon_m
        self.goal_distance = GoalDistanceQuery(source_query)

    @torch.no_grad()
    def __call__(self, paths: Tensor, batch: dict[str, Tensor]) -> CandidateUtility:
        b, count = paths.shape[:2]
        result, clearance = candidate_query(self.source_query, paths, batch, self.horizon)
        collision = (clearance < 0) & result.active
        prefix = result.active & (collision.cumsum(-1) == 0)
        last = (prefix.sum(-1) - 1).clamp_min(0)
        rows = torch.arange(b * count, device=paths.device)
        endpoint_cells = result.grid_cells[rows, last].unflatten(0, (b, count))
        start_and_goal = torch.stack((torch.zeros_like(batch["point_goal"]), batch["point_goal"]), 1)
        fixed_cells = self.source_query.point_cells(
            start_and_goal, batch["source_grid_index"], batch["source_origin_xy"], batch["source_yaw_rad"],
        )
        cells = torch.cat((fixed_cells[:, :1], endpoint_cells, fixed_cells[:, 1:]), 1).cpu().numpy()
        progress = np.empty((b, count), dtype=np.float32)
        for row, index in enumerate(batch["source_grid_index"].cpu().tolist()):
            distances = self.goal_distance.query(index, cells[row, -1], cells[row, :-1])
            if not np.isfinite(distances[0]):
                raise ValueError("source start is disconnected from mission goal")
            if not np.isfinite(distances[1:]).all():
                raise ValueError("collision-free prefix crossed disconnected source components")
            progress[row] = distances[0] - distances[1:]
        progress = torch.as_tensor(progress, device=paths.device)
        minimum = clearance.masked_fill(~result.active, torch.inf).amin(-1).unflatten(0, (b, count))
        contact = collision.any(-1).unflatten(0, (b, count)).float()
        length = path_arc_length(paths.flatten(0, 1).float()).unflatten(0, (b, count))
        score = (
            (progress - 0.5 * length) / self.horizon
            - contact
            - torch.asinh(
                F.relu(EXTRA_CLEARANCE_M - minimum) / ROBOT_FOOTPRINT_RADIUS_M
            )
        )
        return CandidateUtility(score, minimum, progress)


@dataclass
class CurveNavLoss:
    loss: Tensor
    flow_loss: Tensor
    critic_loss: Tensor

    def logging_values(self) -> tuple[Tensor, ...]:
        return self.loss, self.flow_loss, self.critic_loss


class CurveNavCriterion:
    def __init__(self, source_query: SourceConfigurationSpaceQuery, planning_horizon_m: float):
        self.teacher = RouteUtilityTeacher(source_query, planning_horizon_m)

    def __call__(self, output: CurveNavTrainingOutput, batch: dict[str, Tensor]) -> CurveNavLoss:
        target = self.teacher(output.candidate_paths, batch).score
        scores = output.candidate_scores
        regression = F.smooth_l1_loss(scores, target)
        return CurveNavLoss(
            output.flow_loss + regression, output.flow_loss, regression
        )
