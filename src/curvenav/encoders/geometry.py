"""Metric 3D geometry for the canonical calibrated depth camera."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from curvenav.physical import (
    BODY_OBSTACLE_MIN_Z_M,
    EXTRA_CLEARANCE_M,
    MAXIMUM_TRAVERSABLE_SLOPE_DEGREES,
    ROBOT_COLLISION_TOP_Z_M,
    ROBOT_FOOTPRINT_RADIUS_M,
)


CONFIGURATION_GRID_SIZE = 64
# The canonical camera's longest planar ray is below 6.3 m.  Sixty-four
# samples keep adjacent ray points closer than one 7.2/63 m grid cell, so the
# rounded visibility raster cannot skip a cell along a valid sensor ray.
VISIBILITY_RAY_SAMPLES = 64


@dataclass(frozen=True)
class MetricDepthProjection:
    """One consistent per-cell surface and body-obstacle projection."""

    points: Tensor
    depth: Tensor
    obstacle_points: Tensor
    obstacle_valid: Tensor
    configuration_field: Tensor


class MetricDepthProjector(nn.Module):
    """Backproject depth to body XYZ and align horizontal coordinates in time."""

    def __init__(
        self,
        token_height: int,
        token_width: int,
        max_depth_m: float,
        focal_x_px: float,
        focal_y_px: float,
        camera_forward_offset_m: float,
        camera_height_m: float,
        camera_downward_pitch_degrees: float,
        planning_horizon_m: float,
    ) -> None:
        super().__init__()
        if token_height < 1 or token_width < 1:
            raise ValueError("planar token dimensions must be positive")
        if max_depth_m <= 0 or focal_x_px <= 0 or focal_y_px <= 0:
            raise ValueError("depth scale and focal length must be positive")
        self.token_height = token_height
        self.token_width = token_width
        self.max_depth_m = float(max_depth_m)
        self.focal_x_px = float(focal_x_px)
        self.focal_y_px = float(focal_y_px)
        self.camera_forward_offset_m = float(camera_forward_offset_m)
        self.camera_height_m = float(camera_height_m)
        if planning_horizon_m <= 0:
            raise ValueError("planning_horizon_m must be positive")
        self.planning_horizon_m = float(planning_horizon_m)
        self.configuration_grid_size = CONFIGURATION_GRID_SIZE
        self.configuration_resolution_m = (
            2.0 * self.planning_horizon_m / (CONFIGURATION_GRID_SIZE - 1)
        )
        self.minimum_traversable_normal_z = math.cos(
            math.radians(MAXIMUM_TRAVERSABLE_SLOPE_DEGREES)
        )
        pitch = math.radians(camera_downward_pitch_degrees)
        self.pitch_sine = math.sin(pitch)
        self.pitch_cosine = math.cos(pitch)
        axis = torch.linspace(
            -self.planning_horizon_m,
            self.planning_horizon_m,
            CONFIGURATION_GRID_SIZE,
        )
        self.register_buffer("configuration_axis_m", axis, persistent=True)
        axis_index = torch.arange(CONFIGURATION_GRID_SIZE, dtype=torch.float32)
        self.register_buffer(
            "axis_squared_distance",
            (axis_index[:, None] - axis_index[None]).square(),
            persistent=False,
        )

    def _traversable_surface(self, body_points: Tensor, valid: Tensor) -> Tensor:
        """Classify locally supported surfaces by their gravity-relative slope.

        Each pixel participates in four image-grid triangles.  A surface is
        traversable when at least one valid adjacent triangle has a normal no
        steeper than the robot's physical slope limit.  Taking the best local
        triangle avoids turning a traversable ramp into a vertical obstacle at
        depth discontinuities, while vertical furniture faces remain invalid.
        """
        if body_points.shape[:-1] != valid.shape or body_points.shape[-1] != 3:
            raise ValueError("body points and validity mask shapes do not match")
        padded_points = F.pad(
            body_points.permute(0, 3, 1, 2),
            (1, 1, 1, 1),
            mode="replicate",
        ).permute(0, 2, 3, 1)
        center = padded_points[:, 1:-1, 1:-1]
        directions = (
            padded_points[:, 1:-1, 2:] - center,
            padded_points[:, 2:, 1:-1] - center,
            padded_points[:, 1:-1, :-2] - center,
            padded_points[:, :-2, 1:-1] - center,
        )
        padded_valid = F.pad(valid, (1, 1, 1, 1), value=False)
        neighbor_valid = (
            padded_valid[:, 1:-1, 2:],
            padded_valid[:, 2:, 1:-1],
            padded_valid[:, 1:-1, :-2],
            padded_valid[:, :-2, 1:-1],
        )
        traversable = torch.zeros_like(valid)
        for index in range(4):
            following = (index + 1) % 4
            normal = torch.linalg.cross(
                directions[index],
                directions[following],
                dim=-1,
            )
            normal_z = normal[..., 2].abs() / torch.linalg.vector_norm(
                normal,
                dim=-1,
            ).clamp_min(1e-8)
            triangle_valid = (
                valid & neighbor_valid[index] & neighbor_valid[following]
            )
            traversable |= triangle_valid & (
                normal_z >= self.minimum_traversable_normal_z
            )
        return traversable

    def _body_points(
        self,
        selected_depth_m: Tensor,
        selected_index: Tensor,
        image_height: int,
        image_width: int,
    ) -> Tensor:
        """Backproject selected pinhole-depth pixels into the robot body frame."""
        column = selected_index.remainder(image_width).float()
        row = torch.div(
            selected_index,
            image_width,
            rounding_mode="floor",
        ).float()
        ray_x = (column - image_width / 2.0) / self.focal_x_px
        ray_y = (row - image_height / 2.0) / self.focal_y_px
        optical_y = selected_depth_m * ray_y
        return torch.stack(
            (
                self.camera_forward_offset_m
                + self.pitch_cosine * selected_depth_m
                - self.pitch_sine * optical_y,
                -selected_depth_m * ray_x,
                self.camera_height_m
                - self.pitch_cosine * optical_y
                - self.pitch_sine * selected_depth_m,
            ),
            dim=-1,
        )

    def _backproject(
        self,
        selected_depth_m: Tensor,
        selected_index: Tensor,
        image_height: int,
        image_width: int,
        observation_to_current: Tensor,
    ) -> Tensor:
        body_points = self._body_points(
            selected_depth_m,
            selected_index,
            image_height,
            image_width,
        )
        forward, lateral, vertical = body_points.unbind(dim=-1)

        translation = observation_to_current[..., :2].float()
        sine = observation_to_current[..., 2].float()
        cosine = observation_to_current[..., 3].float()
        x_current = cosine[..., None, None] * forward - sine[
            ..., None, None
        ] * lateral
        y_current = sine[..., None, None] * forward + cosine[
            ..., None, None
        ] * lateral
        return torch.stack(
            (
                x_current + translation[..., None, None, 0],
                y_current + translation[..., None, None, 1],
                vertical,
            ),
            dim=-1,
        ).flatten(2, 3)

    def _rasterize(self, points: Tensor, valid: Tensor) -> Tensor:
        """Rasterize aligned planar samples into the fixed robot-centric grid."""
        if points.shape[:-1] != valid.shape or points.shape[-1] != 2:
            raise ValueError("raster points and validity must match")
        batch = points.shape[0]
        size = self.configuration_grid_size
        coordinate = (
            (points + self.planning_horizon_m)
            / (2.0 * self.planning_horizon_m)
            * (size - 1)
        ).round().long()
        inside = (
            valid
            & (coordinate[..., 0] >= 0)
            & (coordinate[..., 0] < size)
            & (coordinate[..., 1] >= 0)
            & (coordinate[..., 1] < size)
        )
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
        horizontal = (
            occupied_cost[:, :, None, :] + squared[None, None]
        ).amin(dim=-1)
        distance_squared = (
            horizontal[:, None, :, :] + squared[None, :, :, None]
        ).amin(dim=2)
        maximum = 2.0 * self.planning_horizon_m
        distance = distance_squared.sqrt() * self.configuration_resolution_m
        return torch.where(torch.isfinite(distance), distance, maximum)

    def _configuration_field(
        self,
        aligned_body_points: Tensor,
        body_pixel: Tensor,
        surface_points: Tensor,
        surface_valid: Tensor,
        observation_to_current: Tensor,
        observation_valid: Tensor,
    ) -> Tensor:
        occupancy = self._rasterize(
            aligned_body_points[..., :2],
            body_pixel & observation_valid[..., None, None],
        )
        signed_clearance = (
            self._euclidean_distance_transform(occupancy)
            - ROBOT_FOOTPRINT_RADIUS_M
        )
        forbidden = signed_clearance[:, None] <= 0.0

        translation = observation_to_current[..., :2].float()
        sine = observation_to_current[..., 2].float()
        cosine = observation_to_current[..., 3].float()
        camera_x = (
            translation[..., 0]
            + cosine * self.camera_forward_offset_m
        )
        camera_y = (
            translation[..., 1]
            + sine * self.camera_forward_offset_m
        )
        origin = torch.stack((camera_x, camera_y), dim=-1)[..., None, :]
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
        ray_valid = (
            surface_valid & observation_valid[..., None]
        )[..., None].expand_as(ray[..., 0])
        observed = self._rasterize(ray, ray_valid)
        # A measured obstacle makes every robot-centre configuration inside
        # its footprint plus safety margin known-unsafe, even if that centre
        # cell is not itself crossed by the camera ray.  Gating only on the
        # obstacle pixel incorrectly hides most of the inflated C-obstacle.
        observed |= signed_clearance[:, None] <= EXTRA_CLEARANCE_M

        clearance = signed_clearance[:, None]
        padded = F.pad(clearance, (1, 1, 1, 1), mode="replicate")
        gradient_x = (
            padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
        ) / (2.0 * self.configuration_resolution_m)
        gradient_y = (
            padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
        ) / (2.0 * self.configuration_resolution_m)
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

    def forward(
        self,
        depth: Tensor,
        observation_to_current: Tensor,
        observation_valid: Tensor,
    ) -> MetricDepthProjection:
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B, T, 1, H, W]")
        if observation_to_current.shape != (*depth.shape[:2], 4):
            raise ValueError("observation_to_current must have shape [B, T, 4]")
        if observation_valid.shape != depth.shape[:2] or observation_valid.dtype != torch.bool:
            raise ValueError("observation_valid must be boolean with shape [B, T]")
        batch, frames, _, height, width = depth.shape
        normalized_depth = depth.squeeze(2).flatten(0, 1)

        # Preserve the nearest visible surface for scene geometry.
        pooled_negative_depth, nearest_index = F.adaptive_max_pool2d(
            -normalized_depth.unsqueeze(1),
            (self.token_height, self.token_width),
            return_indices=True,
        )
        surface_depth = -pooled_negative_depth.squeeze(1).reshape(
            batch,
            frames,
            self.token_height,
            self.token_width,
        )
        surface_index = nearest_index.squeeze(1).reshape(
            batch,
            frames,
            self.token_height,
            self.token_width,
        )

        # Select the nearest body-height obstacle independently.  Selecting a
        # generic nearest surface first can choose the floor and permanently
        # discard a wall occupying the same adaptive cell.
        dense_depth_m = normalized_depth.float() * self.max_depth_m
        dense_index = torch.arange(
            height * width,
            device=depth.device,
        ).reshape(1, height, width)
        dense_body_points = self._body_points(
            dense_depth_m,
            dense_index,
            height,
            width,
        )
        dense_valid = normalized_depth.float() < 1.0
        traversable_surface = self._traversable_surface(
            dense_body_points,
            dense_valid,
        )
        vertical = dense_body_points[..., 2]
        body_pixel_flat = (
            dense_valid
            & (vertical >= BODY_OBSTACLE_MIN_Z_M)
            & (vertical <= ROBOT_COLLISION_TOP_Z_M)
            & ~traversable_surface
        )
        dense_body_points = dense_body_points.reshape(
            batch,
            frames,
            height,
            width,
            3,
        )
        dense_forward, dense_lateral, dense_vertical = dense_body_points.unbind(dim=-1)
        translation = observation_to_current[..., :2].float()
        sine = observation_to_current[..., 2].float()
        cosine = observation_to_current[..., 3].float()
        aligned_dense_body_points = torch.stack(
            (
                cosine[..., None, None] * dense_forward
                - sine[..., None, None] * dense_lateral
                + translation[..., None, None, 0],
                sine[..., None, None] * dense_forward
                + cosine[..., None, None] * dense_lateral
                + translation[..., None, None, 1],
                dense_vertical,
            ),
            dim=-1,
        )
        body_pixel = body_pixel_flat.reshape(batch, frames, height, width)
        masked_negative_depth = torch.where(
            body_pixel_flat,
            -normalized_depth.float(),
            -2.0,
        )
        obstacle_negative_depth, obstacle_index = F.adaptive_max_pool2d(
            masked_negative_depth.unsqueeze(1),
            (self.token_height, self.token_width),
            return_indices=True,
        )
        obstacle_negative_depth = obstacle_negative_depth.squeeze(1).reshape(
            batch,
            frames,
            self.token_height,
            self.token_width,
        )
        obstacle_valid = obstacle_negative_depth > -1.5
        obstacle_depth = torch.where(
            obstacle_valid,
            -obstacle_negative_depth,
            surface_depth.float(),
        )
        obstacle_index = obstacle_index.squeeze(1).reshape_as(surface_index)

        surface_depth_m = surface_depth.float() * self.max_depth_m
        obstacle_depth_m = obstacle_depth.float() * self.max_depth_m
        surface_points = self._backproject(
            surface_depth_m,
            surface_index,
            height,
            width,
            observation_to_current,
        )
        obstacle_points = self._backproject(
            obstacle_depth_m,
            obstacle_index,
            height,
            width,
            observation_to_current,
        )
        flat_obstacle_valid = obstacle_valid.flatten(2)
        surface_valid = surface_depth < 1.0
        configuration_field = self._configuration_field(
            aligned_dense_body_points,
            body_pixel,
            surface_points.reshape(batch, frames, -1, 3),
            surface_valid.flatten(2),
            observation_to_current,
            observation_valid,
        )
        points = torch.where(
            flat_obstacle_valid[..., None],
            obstacle_points,
            surface_points,
        )
        selected_depth = torch.where(
            flat_obstacle_valid,
            obstacle_depth_m.flatten(2),
            surface_depth_m.flatten(2),
        )
        return MetricDepthProjection(
            points=points,
            depth=selected_depth,
            obstacle_points=obstacle_points[..., :2],
            obstacle_valid=flat_obstacle_valid,
            configuration_field=configuration_field,
        )
