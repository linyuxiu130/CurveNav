"""Metric 3D geometry for the canonical calibrated depth camera."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from curvenav.physical import MINIMUM_OBSTACLE_HEIGHT_M, ROBOT_HEIGHT_M


@dataclass(frozen=True)
class MetricDepthProjection:
    """One consistent per-cell surface and body-obstacle projection."""

    points: Tensor
    depth: Tensor
    obstacle_points: Tensor
    obstacle_valid: Tensor


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
        pitch = math.radians(camera_downward_pitch_degrees)
        self.pitch_sine = math.sin(pitch)
        self.pitch_cosine = math.cos(pitch)

    def _backproject(
        self,
        selected_depth_m: Tensor,
        selected_index: Tensor,
        image_height: int,
        image_width: int,
        observation_to_current: Tensor,
    ) -> Tensor:
        column = selected_index.remainder(image_width).float()
        row = torch.div(
            selected_index,
            image_width,
            rounding_mode="floor",
        ).float()
        ray_x = (column - image_width / 2.0) / self.focal_x_px
        ray_y = (row - image_height / 2.0) / self.focal_y_px
        optical_y = selected_depth_m * ray_y
        forward = (
            self.camera_forward_offset_m
            + self.pitch_cosine * selected_depth_m
            - self.pitch_sine * optical_y
        )
        lateral = -selected_depth_m * ray_x
        vertical = (
            self.camera_height_m
            - self.pitch_cosine * optical_y
            - self.pitch_sine * selected_depth_m
        )

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

    def forward(
        self,
        depth: Tensor,
        observation_to_current: Tensor,
    ) -> MetricDepthProjection:
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B, T, 1, H, W]")
        if observation_to_current.shape != (*depth.shape[:2], 4):
            raise ValueError("observation_to_current must have shape [B, T, 4]")
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
        row = torch.arange(height, device=depth.device, dtype=torch.float32)
        ray_y = (row - height / 2.0) / self.focal_y_px
        vertical = (
            self.camera_height_m
            - self.pitch_cosine * dense_depth_m * ray_y[None, :, None]
            - self.pitch_sine * dense_depth_m
        )
        body_pixel = (
            (normalized_depth.float() < 1.0)
            & (vertical >= MINIMUM_OBSTACLE_HEIGHT_M)
            & (vertical <= ROBOT_HEIGHT_M)
        )
        masked_negative_depth = torch.where(
            body_pixel,
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
        )
