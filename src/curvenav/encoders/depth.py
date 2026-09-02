"""Shared multi-frame metric depth-token encoder."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.types import DepthFeatures

from .geometry import MetricDepthProjector


DEPTH_ENCODER_TYPE = "shared_multiframe_resnet18_aligned_cspace_evidence"


def _channel_group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class ResNet18BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm_1 = _channel_group_norm(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.convolution_2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.norm_2 = _channel_group_norm(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _channel_group_norm(out_channels),
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        residual = self.skip(value)
        value = self.activation(self.norm_1(self.convolution_1(value)))
        value = self.norm_2(self.convolution_2(value))
        return self.activation(value + residual)


def _resnet18_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        ResNet18BasicBlock(in_channels, out_channels, stride),
        ResNet18BasicBlock(out_channels, out_channels),
    )


class DepthObservationEncoder(nn.Module):
    """Encode every calibrated depth frame with one shared visual backbone.

    Each image token retains its learned appearance feature and receives the
    corresponding measured ray endpoint transformed into the current robot
    frame. The same projection constructs observed robot configuration space;
    no PointGoal or map completion enters perception.
    """

    def __init__(
        self,
        *,
        observation_frames: int,
        model_dim: int,
        frame_tokens_height: int,
        frame_tokens_width: int,
        dropout: float,
        max_depth_m: float,
        focal_x_px: float,
        focal_y_px: float,
        camera_forward_offset_m: float,
        camera_height_m: float,
        camera_downward_pitch_degrees: float,
        planning_horizon_m: float,
    ) -> None:
        super().__init__()
        if observation_frames < 1:
            raise ValueError("observation_frames must be positive")
        if model_dim % 4:
            raise ValueError("model_dim must be divisible by four")
        if min(frame_tokens_height, frame_tokens_width) < 1:
            raise ValueError("depth token grid dimensions must be positive")
        self.observation_frames = observation_frames
        self.model_dim = model_dim
        self.tokens_height = frame_tokens_height
        self.tokens_width = frame_tokens_width
        self.tokens_per_frame = frame_tokens_height * frame_tokens_width
        self.max_depth_m = float(max_depth_m)

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            _channel_group_norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            _resnet18_stage(64, 64, 1),
            _resnet18_stage(64, 128, 2),
            _resnet18_stage(128, 256, 2),
        )
        self.spatial_projection = nn.Conv2d(256, model_dim, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (frame_tokens_height, frame_tokens_width)
        )
        self.metric_projector = MetricDepthProjector(
            frame_tokens_height,
            frame_tokens_width,
            max_depth_m,
            focal_x_px,
            focal_y_px,
            camera_forward_offset_m,
            camera_height_m,
            camera_downward_pitch_degrees,
            planning_horizon_m,
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.register_buffer(
            "image_position",
            self._position_encoding(model_dim, frame_tokens_height, frame_tokens_width),
            persistent=True,
        )
        self.frame_embedding = nn.Parameter(
            torch.empty(1, observation_frames, 1, model_dim)
        )
        self.null_token = nn.Parameter(torch.empty(1, 1, 1, model_dim))
        nn.init.trunc_normal_(self.frame_embedding, std=0.02)
        nn.init.trunc_normal_(self.null_token, std=0.02)
        self.output_norm = RMSNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _position_encoding(model_dim: int, height: int, width: int) -> Tensor:
        quarter = model_dim // 4
        frequency = torch.exp(
            torch.arange(quarter, dtype=torch.float32)
            * (-math.log(10_000.0) / max(quarter - 1, 1))
        )
        rows = torch.arange(height, dtype=torch.float32)[:, None] * frequency[None]
        columns = torch.arange(width, dtype=torch.float32)[:, None] * frequency[None]
        row = torch.cat((rows.sin(), rows.cos()), dim=-1)
        column = torch.cat((columns.sin(), columns.cos()), dim=-1)
        return torch.cat(
            (
                row[:, None].expand(-1, width, -1),
                column[None].expand(height, -1, -1),
            ),
            dim=-1,
        ).reshape(1, 1, height * width, model_dim)

    def forward(
        self,
        depth: Tensor,
        observation_to_current: Tensor,
        observation_valid: Tensor,
    ) -> DepthFeatures:
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B,F,1,H,W]")
        if depth.shape[1] != self.observation_frames:
            raise ValueError("depth history does not match the encoder contract")
        if observation_to_current.shape != (*depth.shape[:2], 4):
            raise ValueError("observation transforms must have shape [B,F,4]")
        if observation_valid.shape != depth.shape[:2]:
            raise ValueError("observation validity must have shape [B,F]")
        batch, frames = depth.shape[:2]
        visual = self.backbone(depth.flatten(0, 1))
        visual = self.adaptive_pool(self.spatial_projection(visual))
        visual = (
            visual.flatten(2)
            .transpose(1, 2)
            .reshape(batch, frames, self.tokens_per_frame, self.model_dim)
        )
        projection = self.metric_projector(
            depth,
            observation_to_current,
            observation_valid,
        )
        points = projection.points
        selected_depth = projection.depth / self.max_depth_m
        surface_hit = projection.depth < self.max_depth_m
        body_obstacle = projection.obstacle_valid & observation_valid[..., None]
        token_valid = observation_valid[..., None].expand_as(surface_hit)
        geometry = torch.cat(
            (
                points / self.max_depth_m,
                selected_depth[..., None],
                surface_hit[..., None].to(points.dtype),
                body_obstacle[..., None].to(points.dtype),
            ),
            dim=-1,
        )
        tokens = visual + self.geometry_projection(geometry.to(visual.dtype))
        tokens = tokens + self.image_position.to(tokens.dtype)
        tokens = tokens + self.frame_embedding.to(tokens.dtype)
        tokens = torch.where(
            token_valid[..., None],
            tokens,
            self.null_token.to(tokens.dtype) + self.frame_embedding.to(tokens.dtype),
        )
        tokens = self.dropout(self.output_norm(tokens)).flatten(1, 2)
        return DepthFeatures(
            tokens=tokens,
            token_valid=token_valid.flatten(1),
            metric_position=points.flatten(1, 2),
            configuration_field=projection.configuration_field,
        )
