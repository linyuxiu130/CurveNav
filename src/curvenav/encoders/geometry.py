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
        index = torch.arange(CONFIGURATION_GRID_SIZE, dtype=torch.float32)
        self.register_buffer(
            "axis_squared_distance",
            (index[:, None] - index[None]).square(),
            persistent=False,
        )

    def _rasterize(self, points: Tensor, valid: Tensor) -> Tensor:
        """Rasterize aligned planar samples into the fixed robot-centric grid."""
        if points.shape[:-1] != valid.shape or points.shape[-1] != 2:
            raise ValueError("raster points and validity must match")
        batch = points.shape[0]
        size = self.configuration_grid_size
        coordinate = (
            (
                (points + self.planning_horizon_m)
                / (2.0 * self.planning_horizon_m)
                * (size - 1)
            )
            .round()
            .long()
        )
        inside = valid & (points.abs() <= self.planning_horizon_m).all(dim=-1)
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
        if mask.ndim != 4 or mask.shape[1:] != (
            1,
            self.configuration_grid_size,
            self.configuration_grid_size,
        ):
            raise ValueError("distance mask does not match the configuration grid")
        occupied_cost = torch.where(
            mask[:, 0],
            torch.zeros((), device=mask.device),
            torch.full((), torch.inf, device=mask.device),
        )
        squared = self.axis_squared_distance.to(mask.device)
        horizontal = (occupied_cost[:, :, None, :] + squared[None, None]).amin(dim=-1)
        distance_squared = (horizontal[:, None, :, :] + squared[None, :, :, None]).amin(
            dim=2
        )
        maximum = 2.0 * self.planning_horizon_m
        distance = distance_squared.sqrt() * self.configuration_resolution_m
        return torch.where(torch.isfinite(distance), distance, maximum)

    def _configuration_field(
        self,
        aligned_body_points: Tensor,
        body_pixel: Tensor,
        surface_points: Tensor,
        surface_valid: Tensor,
        camera_to_current: Tensor,
        observation_valid: Tensor,
    ) -> Tensor:
        occupancy = self._rasterize(
            aligned_body_points[..., :2],
            body_pixel & observation_valid[..., None, None],
        )
        signed_clearance = (
            self._euclidean_distance_transform(occupancy)
            - ROBOT_FOOTPRINT_RADIUS_M
            # Nearest-node rasterization moves a measured point by at most
            # sqrt(2)*resolution/2. Triangle inequality makes this a lower
            # bound on clearance to measured points at each grid node.
            - self.configuration_resolution_m / math.sqrt(2)
        )
        forbidden = signed_clearance[:, None] <= 0.0

        origin = camera_to_current[..., :2, 3][..., None, :]
        endpoint = surface_points[..., :2]
        alpha = torch.linspace(
            0.0,
            1.0,
            VISIBILITY_RAY_SAMPLES,
            device=endpoint.device,
        )
        ray = origin[..., None, :] + alpha[None, None, None, :, None] * (
            endpoint[..., None, :] - origin[..., None, :]
        )
        ray_valid = (surface_valid & observation_valid[..., None])[..., None].expand_as(
            ray[..., 0]
        )
        observed = self._rasterize(ray, ray_valid)
        # A measured obstacle makes every robot-centre configuration inside
        # its footprint plus safety margin known-unsafe, even if that centre
        # cell is not itself crossed by the camera ray.  Gating only on the
        # obstacle pixel incorrectly hides most of the inflated C-obstacle.
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
                forbidden.float(),
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
        # Discard old measured points contradicted by a newer free-space ray.
        # Occluded or out-of-view history remains in the sixteen-observation window.
        for newer in range(1, frames):
            current_rotation = camera_to_current[:, newer, :3, :3]
            current_origin = camera_to_current[:, newer, :3, 3]
            in_current = torch.einsum(
                "bij,bfhwi->bfhwj",
                current_rotation,
                points - current_origin[:, None, None, None],
            )
            z = in_current[..., 2]
            current_k = k[:, newer]
            u = (
                current_k[:, None, None, None, 0, 0]
                * in_current[..., 0]
                / z.clamp_min(1e-6)
                + current_k[:, None, None, None, 0, 2]
                - 0.5
            )
            v = (
                current_k[:, None, None, None, 1, 1]
                * in_current[..., 1]
                / z.clamp_min(1e-6)
                + current_k[:, None, None, None, 1, 2]
                - 0.5
            )
            inside = (z > 0) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
            # Use the nearest depth in the four surrounding pixels. Bilinear depth
            # or one rounded pixel can invent free space across a depth discontinuity.
            u0 = u.floor().clamp(0, width - 1).long()
            v0 = v.floor().clamp(0, height - 1).long()
            u1, v1 = (u0 + 1).clamp_max(width - 1), (v0 + 1).clamp_max(height - 1)
            latest = depth[:, newer].flatten(1)
            samples = [
                latest.gather(1, (yy * width + xx).flatten(1)).reshape_as(depth)
                for xx, yy in ((u0, v0), (u1, v0), (u0, v1), (u1, v1))
            ]
            measured = torch.stack(samples).amin(0) * self.max_depth_m
            # Propagate the storage quantization bound through the optical-Z
            # transform: z_current = a * Z_history + b. One FP16 epsilon over
            # [0,max_depth] also covers the reserved endpoint encodings. This
            # is a numerical tolerance, not an assumed sensor-noise model.
            z_scale = torch.einsum(
                "bi,bfij,bfhwj->bfhw",
                current_rotation[:, :, 2],
                camera_to_current[..., :3, :3],
                rays,
            ).abs()
            tolerance = self.max_depth_m * torch.finfo(torch.float16).eps * (1 + z_scale)
            contradicted = inside & (measured > 0) & (z < measured - tolerance)
            historical = (
                torch.arange(frames, device=depth.device)[None, :, None, None] < newer
            )
            contradicted = contradicted & condition.observation_valid[:, newer, None, None, None]
            valid = valid & ~(historical & hit & contradicted)
        hit = hit & valid
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
            points,
            body_pixel,
            surface,
            surface_valid,
            camera_to_current,
            condition.observation_valid,
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
