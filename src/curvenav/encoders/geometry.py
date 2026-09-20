"""Optical-Z backprojection and bounded SE(3)-aligned depth geometry."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from curvenav.physical import (
    BODY_OBSTACLE_MIN_Z_M,
    EXTRA_CLEARANCE_M,
    ROBOT_COLLISION_TOP_Z_M,
    ROBOT_FOOTPRINT_RADIUS_M,
)


CONFIGURATION_GRID_SIZE = 64
# Finite sampled ray coverage is observation evidence, not a swept-body free-space certificate.
VISIBILITY_RAY_SAMPLES = 64


@dataclass(frozen=True)
class MetricDepthProjection:
    """One consistent per-cell surface and body-obstacle projection."""

    points: Tensor
    depth: Tensor
    obstacle_points: Tensor
    obstacle_valid: Tensor
    configuration_field: Tensor
    token_valid: Tensor
    pixel_indices: Tensor


class MetricDepthProjector(nn.Module):
    """Backproject depth to body XYZ and align horizontal coordinates in time."""

    def __init__(
        self,
        token_height: int,
        token_width: int,
        max_depth_m: float,
        planning_horizon_m: float,
    ) -> None:
        super().__init__()
        self.token_height, self.token_width = token_height, token_width
        self.max_depth_m = max_depth_m
        self.planning_horizon_m = planning_horizon_m
        self.configuration_grid_size = CONFIGURATION_GRID_SIZE
        self.configuration_resolution_m = (
            2 * planning_horizon_m / (CONFIGURATION_GRID_SIZE - 1)
        )
        axis = torch.linspace(
            -planning_horizon_m, planning_horizon_m, CONFIGURATION_GRID_SIZE
        )
        self.register_buffer("configuration_axis_m", axis)
        self.obstacle_padding = math.ceil(
            (ROBOT_FOOTPRINT_RADIUS_M + EXTRA_CLEARANCE_M)
            / self.configuration_resolution_m
        ) + 1
        index = torch.arange(
            CONFIGURATION_GRID_SIZE + 2 * self.obstacle_padding, dtype=torch.float32
        )
        self.register_buffer(
            "axis_squared_distance",
            (index[:, None] - index[None]).square(),
            persistent=False,
        )

    def _rasterize(self, points: Tensor, valid: Tensor, padding: int = 0) -> Tensor:
        """Rasterize aligned planar samples into the fixed robot-centric grid."""
        if points.shape[:-1] != valid.shape or points.shape[-1] != 2:
            raise ValueError("raster points and validity must match")
        batch = points.shape[0]
        size = self.configuration_grid_size + 2 * padding
        coordinate = (
            (
                (points + self.planning_horizon_m)
                / (2.0 * self.planning_horizon_m)
                * (self.configuration_grid_size - 1)
            )
            .round()
            .long() + padding
        )
        extent = self.planning_horizon_m + padding * self.configuration_resolution_m
        inside = valid & (points.abs() <= extent).all(dim=-1)
        x = coordinate[..., 0].clamp(0, size - 1)
        y = coordinate[..., 1].clamp(0, size - 1)
        batch_index = torch.arange(batch, device=points.device).view(
            batch, *([1] * (points.ndim - 2))
        )
        flat_index = batch_index * (size * size) + y * size + x
        raster = torch.zeros(
            batch * size * size,
            device=points.device,
            dtype=torch.float32,
        )
        raster.scatter_add_(
            0,
            flat_index.reshape(-1),
            inside.reshape(-1).float(),
        )
        return raster.reshape(batch, 1, size, size) > 0

    def _euclidean_distance_transform(self, mask: Tensor) -> Tensor:
        """Exact separable squared-Euclidean transform on the fixed grid."""
        size = mask.shape[-1]
        if mask.ndim != 4 or mask.shape[1:3] != (1, size):
            raise ValueError("distance mask does not match the configuration grid")
        occupied_cost = torch.where(
            mask[:, 0],
            torch.zeros((), device=mask.device),
            torch.full((), torch.inf, device=mask.device),
        )
        squared = self.axis_squared_distance[:size, :size]
        horizontal = (occupied_cost[:, :, None, :] + squared[None, None]).amin(dim=-1)
        distance_squared = (horizontal[:, None, :, :] + squared[None, :, :, None]).amin(
            dim=2
        )
        maximum = 2.0 * self.planning_horizon_m
        distance = distance_squared.sqrt() * self.configuration_resolution_m
        return torch.where(torch.isfinite(distance), distance, maximum)

    def _configuration_field(
        self,
        surface_points: Tensor,
        surface_valid: Tensor,
        camera_to_current: Tensor,
        observation_valid: Tensor,
        obstacle_memory: Tensor,
    ) -> Tensor:
        # The sensor-clock memory already includes the current observation and
        # every intervening clear. Historical image tokens are timed evidence,
        # not another writer of the current occupied state.
        occupancy = obstacle_memory[:, None]
        # Inflate before cropping: obstacles just outside the BEV still collide
        # with robot footprints whose centres are inside its boundary.
        pad = self.obstacle_padding
        # Metric distance to the represented raster sites, not a physical
        # clearance certificate. Keep discretization error out of learned values.
        signed_clearance = (
            self._euclidean_distance_transform(occupancy)[:, pad:-pad, pad:-pad]
            - ROBOT_FOOTPRINT_RADIUS_M
        )
        raster_overlap = signed_clearance[:, None] <= 0.0

        # A past free-space ray does not establish visibility now: a moving
        # obstacle may have entered it. Only current rays mark observed space.
        origin = camera_to_current[:, -1:, :2, 3][..., None, :]
        endpoint = surface_points[:, -1:, :, :2]
        alpha = torch.linspace(
            0.0,
            1.0,
            VISIBILITY_RAY_SAMPLES,
            device=endpoint.device,
        )
        ray = origin[..., None, :] + alpha[None, None, None, :, None] * (
            endpoint[..., None, :] - origin[..., None, :]
        )
        ray_valid = (surface_valid[:, -1:] & observation_valid[:, -1:, None])[..., None].expand_as(
            ray[..., 0]
        )
        observed = self._rasterize(ray, ray_valid)
        # Include local evidence around represented obstacles even where no
        # sampled ray crosses the robot centre. Coverage is not a free-space
        # certificate, and raster overlap is not ground-truth collision.
        observed |= signed_clearance[:, None] <= EXTRA_CLEARANCE_M

        clearance = signed_clearance[:, None]
        padded = F.pad(clearance, (1, 1, 1, 1), mode="replicate")
        gradient_x = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / (
            2.0 * self.configuration_resolution_m
        )
        gradient_y = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / (
            2.0 * self.configuration_resolution_m
        )
        gradient_norm = torch.sqrt(gradient_x.square() + gradient_y.square()).clamp_min(
            1e-6
        )
        return torch.cat(
            (
                clearance,
                gradient_x / gradient_norm,
                gradient_y / gradient_norm,
                observed.float(),
                raster_overlap.float(),
            ),
            dim=1,
        )

    def forward(self, condition) -> MetricDepthProjection:
        depth = condition.depth.float().squeeze(2)
        batch, frames, height, width = depth.shape
        k = condition.camera_intrinsics.float()
        camera_to_current = (
            condition.observation_to_current.float() @ condition.camera_to_body.float()
        )
        y, x = torch.meshgrid(
            torch.arange(height, device=depth.device) + 0.5,
            torch.arange(width, device=depth.device) + 0.5,
            indexing="ij",
        )
        rays = torch.stack(
            (
                (x - k[..., 0, 2, None, None]) / k[..., 0, 0, None, None],
                (y - k[..., 1, 2, None, None]) / k[..., 1, 1, None, None],
                torch.ones_like(depth),
            ),
            dim=-1,
        )
        optical = rays * (depth * self.max_depth_m)[..., None]
        points = torch.einsum(
            "bfij,bfhwj->bfhwi", camera_to_current[..., :3, :3], optical
        )
        points = points + camera_to_current[..., None, None, :3, 3]
        valid = (depth > 0) & condition.observation_valid[..., None, None]
        hit = valid & (depth < 1)
        # Preserve historical measurements at their acquisition times. Ego-motion
        # alignment does not move a dynamic object to its present position; the
        # encoder's age features distinguish these from current observations.
        # This is a planar body-collision field, not a terrain-connectivity map.
        # A flat surface in the body band is still an obstacle (e.g. a platform).
        body_pixel = (
            hit
            & (points[..., 2] >= BODY_OBSTACLE_MIN_Z_M)
            & (points[..., 2] <= ROBOT_COLLISION_TOP_Z_M)
        )

        def select(mask):
            value, indices = F.adaptive_max_pool2d(
                torch.where(mask, -depth, -2.0).flatten(0, 1).unsqueeze(1),
                (self.token_height, self.token_width),
                return_indices=True,
            )
            indices = indices.reshape(batch, frames, -1)
            selected_points = points.flatten(2, 3).gather(
                2, indices[..., None].expand(-1, -1, -1, 3)
            )
            selected_depth = depth.flatten(2).gather(2, indices) * self.max_depth_m
            return (
                selected_points,
                selected_depth,
                value.reshape(batch, frames, -1) > -1.5,
                indices,
            )

        surface, surface_depth, surface_valid, surface_indices = select(valid)
        obstacle, obstacle_depth, obstacle_valid, obstacle_indices = select(body_pixel)
        field = self._configuration_field(
            surface,
            surface_valid,
            camera_to_current,
            condition.observation_valid,
            condition.obstacle_memory,
        )
        return MetricDepthProjection(
            points=torch.where(obstacle_valid[..., None], obstacle, surface),
            depth=torch.where(obstacle_valid, obstacle_depth, surface_depth),
            obstacle_points=obstacle[..., :2],
            obstacle_valid=obstacle_valid,
            configuration_field=field,
            token_valid=surface_valid,
            pixel_indices=torch.where(obstacle_valid, obstacle_indices, surface_indices),
        )
