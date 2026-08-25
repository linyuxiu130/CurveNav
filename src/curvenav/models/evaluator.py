"""SanD-style analytic geometry costs for generated local trajectories."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


TRAJECTORY_EVALUATOR_TYPE = "depth_surface_clearance_length_goal"


@dataclass(frozen=True)
class TrajectoryCosts:
    total: Tensor
    clearance: Tensor
    length: Tensor
    goal: Tensor
    minimum_clearance: Tensor


class GeometricTrajectoryEvaluator(nn.Module):
    """Rank candidates from current metric depth without learned parameters."""

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        focal_x_px: float,
        focal_y_px: float,
        max_depth_m: float,
        camera_forward_offset_m: float,
        camera_height_m: float,
        camera_downward_pitch_degrees: float,
        minimum_obstacle_height_m: float,
        robot_height_m: float,
        robot_radius_m: float,
        safety_margin_m: float,
        discount_factor: float,
        clearance_weight: float,
        length_weight: float,
        goal_weight: float,
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.focal_x_px = float(focal_x_px)
        self.focal_y_px = float(focal_y_px)
        self.max_depth_m = float(max_depth_m)
        self.camera_forward_offset_m = float(camera_forward_offset_m)
        self.camera_height_m = float(camera_height_m)
        pitch = math.radians(camera_downward_pitch_degrees)
        self.pitch_sine = math.sin(pitch)
        self.pitch_cosine = math.cos(pitch)
        self.minimum_obstacle_height_m = float(minimum_obstacle_height_m)
        self.robot_height_m = float(robot_height_m)
        self.robot_radius_m = float(robot_radius_m)
        self.safe_center_distance_m = float(robot_radius_m + safety_margin_m)
        self.discount_factor = float(discount_factor)
        self.clearance_weight = float(clearance_weight)
        self.length_weight = float(length_weight)
        self.goal_weight = float(goal_weight)

        rows, columns = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("pixel_rows", rows, persistent=False)
        self.register_buffer("pixel_columns", columns, persistent=False)

    def _obstacle_points(self, current_depth: Tensor) -> tuple[Tensor, Tensor]:
        depth_m = current_depth.float() * self.max_depth_m
        optical_x = (
            (self.pixel_columns - self.image_width / 2.0)
            / self.focal_x_px
            * depth_m
        )
        optical_y = (
            (self.pixel_rows - self.image_height / 2.0)
            / self.focal_y_px
            * depth_m
        )
        forward = (
            self.camera_forward_offset_m
            + self.pitch_cosine * depth_m
            - self.pitch_sine * optical_y
        )
        obstacle_height = (
            self.camera_height_m
            - self.pitch_sine * depth_m
            - self.pitch_cosine * optical_y
        )
        valid = (
            torch.isfinite(depth_m)
            & (depth_m > 0.0)
            & (depth_m < self.max_depth_m)
            & (obstacle_height >= self.minimum_obstacle_height_m)
            & (obstacle_height <= self.robot_height_m)
        )
        points = torch.stack((forward, -optical_x), dim=-1)
        return points, valid

    def _path_clearance(
        self,
        candidate_paths: Tensor,
        obstacle_points: Tensor,
        obstacle_valid: Tensor,
    ) -> Tensor:
        batch, candidates, path_points = candidate_paths.shape[:3]
        output = torch.full(
            (batch, candidates * path_points),
            self.max_depth_m,
            device=candidate_paths.device,
            dtype=torch.float32,
        )
        flat_paths = candidate_paths.float().flatten(1, 2)
        for batch_index in range(batch):
            visible = obstacle_points[batch_index][obstacle_valid[batch_index]]
            if len(visible) == 0:
                continue
            minimum = output[batch_index]
            for chunk in visible.split(2048):
                distance = torch.cdist(flat_paths[batch_index], chunk.float())
                minimum = torch.minimum(minimum, distance.amin(dim=1))
            output[batch_index] = minimum.clamp_max(self.max_depth_m)

        forward = flat_paths[..., 0] - self.camera_forward_offset_m
        lateral = flat_paths[..., 1]
        optical_forward = self.pitch_cosine * forward
        projected_column = (
            self.image_width / 2.0
            - self.focal_x_px * lateral / optical_forward.clamp_min(1e-6)
        )
        supported = (
            (optical_forward >= 0.0)
            & (optical_forward <= self.max_depth_m)
            & (projected_column >= 0.0)
            & (projected_column <= self.image_width - 1)
        )
        known_origin = torch.linalg.vector_norm(flat_paths, dim=-1) <= max(
            self.robot_radius_m,
            self.camera_forward_offset_m,
        )
        supported = supported | known_origin
        supported = supported.reshape(batch, candidates, path_points)
        output = output.reshape(batch, candidates, path_points)
        return torch.where(supported, output, torch.zeros_like(output))

    def forward(
        self,
        candidate_paths: Tensor,
        current_depth: Tensor,
        point_goal: Tensor,
    ) -> TrajectoryCosts:
        if candidate_paths.ndim != 4 or candidate_paths.shape[-1] != 2:
            raise ValueError("candidate_paths must have shape [B,C,P,2]")
        if current_depth.shape != (
            candidate_paths.shape[0],
            self.image_height,
            self.image_width,
        ):
            raise ValueError("current_depth must have shape [B,H,W]")
        if point_goal.shape != (candidate_paths.shape[0], 2):
            raise ValueError("point_goal must have shape [B,2]")

        obstacle_points, obstacle_valid = self._obstacle_points(current_depth)
        distances = self._path_clearance(
            candidate_paths,
            obstacle_points,
            obstacle_valid,
        )
        discount = self.discount_factor ** torch.arange(
            candidate_paths.shape[2],
            device=candidate_paths.device,
            dtype=torch.float32,
        )
        violation = (self.safe_center_distance_m - distances).clamp_min(0.0)
        clearance_cost = (violation * discount).sum(dim=-1) / discount.sum()
        length_cost = torch.linalg.vector_norm(
            candidate_paths[:, :, 1:] - candidate_paths[:, :, :-1], dim=-1
        ).sum(dim=-1)
        goal_cost = torch.linalg.vector_norm(
            candidate_paths[:, :, -1] - point_goal[:, None], dim=-1
        )
        total = (
            self.clearance_weight * clearance_cost
            + self.length_weight * length_cost
            + self.goal_weight * goal_cost
        )
        return TrajectoryCosts(
            total=total,
            clearance=clearance_cost,
            length=length_cost,
            goal=goal_cost,
            minimum_clearance=distances.amin(dim=-1),
        )
