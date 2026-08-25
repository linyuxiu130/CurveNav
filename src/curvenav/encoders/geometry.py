"""Metric planar geometry for the canonical calibrated depth camera."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PlanarDepthProjector(nn.Module):
    """Backproject pooled optical-axis depth and align it to the current frame."""

    def __init__(
        self,
        token_height: int,
        token_width: int,
        max_depth_m: float,
        focal_x_px: float,
        focal_y_px: float,
        camera_forward_offset_m: float,
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
        pitch = math.radians(camera_downward_pitch_degrees)
        self.pitch_sine = math.sin(pitch)
        self.pitch_cosine = math.cos(pitch)

    def forward(
        self,
        depth: Tensor,
        observation_to_current: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B, T, 1, H, W]")
        if observation_to_current.shape != (*depth.shape[:2], 4):
            raise ValueError("observation_to_current must have shape [B, T, 4]")
        batch, frames, _, height, width = depth.shape
        depth_m = depth.float().squeeze(2) * self.max_depth_m
        pooled_depth = F.adaptive_avg_pool2d(
            depth_m.flatten(0, 1).unsqueeze(1),
            (self.token_height, self.token_width),
        ).squeeze(1).reshape(
            batch, frames, self.token_height, self.token_width
        )

        column = torch.linspace(
            0.5 * width / self.token_width - 0.5,
            width - 0.5 * width / self.token_width,
            self.token_width,
            device=depth.device,
            dtype=depth.dtype,
        )
        row = torch.linspace(
            0.5 * height / self.token_height - 0.5,
            height - 0.5 * height / self.token_height,
            self.token_height,
            device=depth.device,
            dtype=depth.dtype,
        )
        ray_x = (column - width / 2.0) / self.focal_x_px
        ray_y = (row - height / 2.0) / self.focal_y_px
        optical_y = pooled_depth * ray_y[None, None, :, None]
        forward = (
            self.camera_forward_offset_m
            + self.pitch_cosine * pooled_depth
            - self.pitch_sine * optical_y
        )
        lateral = -pooled_depth * ray_x[None, None, None, :]
        points = torch.stack((forward, lateral), dim=-1)

        translation = observation_to_current[..., :2].float()
        sine = observation_to_current[..., 2].float()
        cosine = observation_to_current[..., 3].float()
        x_current = cosine[..., None, None] * points[..., 0] - sine[..., None, None] * points[..., 1]
        y_current = sine[..., None, None] * points[..., 0] + cosine[..., None, None] * points[..., 1]
        points = torch.stack((x_current, y_current), dim=-1)
        points = points + translation[..., None, None, :]
        points = torch.nan_to_num(
            points,
            nan=0.0,
            posinf=self.max_depth_m,
            neginf=-self.max_depth_m,
        )
        return points.flatten(2, 3), pooled_depth.flatten(2)
