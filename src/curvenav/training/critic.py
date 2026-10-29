"""Supervised route utility: geometry is the teacher, never an inference input."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from curvenav.data.goal_distance import GoalDistanceQuery
from curvenav.data.privileged import SourceConfigurationSpaceQuery, SourcePathQuery
from curvenav.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M
from curvenav.trajectory import path_arc_length


CRITIC_TARGET = "omni_task_admissible_progress_novelty_failure_clearance"


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
    failed: Tensor


class RouteUtilityTeacher:
    def __init__(self, source_query: SourceConfigurationSpaceQuery, planning_horizon_m: float, robot_radius_m: float = ROBOT_FOOTPRINT_RADIUS_M):
        self.robot_radius_m = robot_radius_m
        self.source_query = source_query
        self.horizon = planning_horizon_m
        self.goal_distance = GoalDistanceQuery(source_query)

    @torch.no_grad()
    def __call__(self, paths: Tensor, batch: dict[str, Tensor]) -> CandidateUtility:
        b, count = paths.shape[:2]
        result, clearance = candidate_query(self.source_query, paths, batch, self.horizon)
        collision = (clearance < 0) & result.active
        failure=collision
        if 'task_states' in batch:
            from curvenav.data.fixed40_targets import task_failure_mask
            local=result.local_path.unflatten(0,(b,count)).double().cpu().numpy()
            yaw=batch['source_yaw_rad'].double().cpu().numpy();c,s=np.cos(yaw),np.sin(yaw)
            rotation=np.stack((np.stack((c,-s),-1),np.stack((s,c),-1)),-2)
            world=np.einsum('bcpj,bij->bcpi',local,rotation)+batch['source_origin_xy'].double().cpu().numpy()[:,None,None,:]
            invalid=torch.as_tensor(task_failure_mask(world,batch['task_states']),device=paths.device).flatten(0,1)
            failure=collision|(invalid&result.active)
        prefix = result.active & (failure.cumsum(-1) == 0)
        last = (prefix.sum(-1) - 1).clamp_min(0)
        rows = torch.arange(b * count, device=paths.device)
        endpoint_cells = result.grid_cells[rows, last].unflatten(0, (b, count))
        present = batch["goal_present"].bool().reshape(b)
        goal = torch.where(present[:, None], batch["point_goal"], torch.zeros_like(batch["point_goal"]))
        start_and_goal = torch.stack((torch.zeros_like(goal), goal), 1)
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
            # Without a mission, reward traversable displacement from the start.
            progress[row] = distances[0] - distances[1:] if present[row] else distances[1:]
        progress = torch.as_tensor(progress, device=paths.device)
        minimum = clearance.masked_fill(~result.active, torch.inf).amin(-1).unflatten(0, (b, count))
        contact = failure.any(-1).unflatten(0, (b, count)).float()
        length = path_arc_length(paths.flatten(0, 1).float()).unflatten(0, (b, count))
        if (~present).any():
            # Progress and novelty end at the same last physically/task-valid point.
            sampled=result.local_path.unflatten(0,(b,count))
            segments=sampled[:,:,1:]-sampled[:,:,:-1]
            valid_segments=(prefix[:,1:]&prefix[:,:-1]).unflatten(0,(b,count))
            safe_length=segments.norm(dim=-1)*valid_segments
            midpoint=(sampled[:,:,1:]+sampled[:,:,:-1])/(2*self.horizon)
            coverage=F.grid_sample(batch['observed_field'][:,3:4].float(),midpoint.flatten(1,2).unsqueeze(2),align_corners=True).reshape(b,count,-1)
            coverage_available=(midpoint.abs()<=1).all(-1)
            novelty=((1-coverage)*coverage_available*safe_length).sum(-1)/safe_length.sum(-1).clamp_min(1e-12)
            progress=progress*(1+(~present)[:,None]*novelty)
        score = (
            (progress - 0.5 * length) / self.horizon
            - contact
            - torch.asinh(
                F.relu(EXTRA_CLEARANCE_M - minimum) / self.robot_radius_m
            )
        )
        return CandidateUtility(score, minimum, progress,failure.any(-1).unflatten(0,(b,count)))
