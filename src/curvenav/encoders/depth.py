"""SanD-style depth tokens with metric planar backprojection."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.types import DepthFeatures
from .geometry import PlanarDepthProjector


DEPTH_ENCODER_TYPE = (
    "sand_resnet18_groupnorm_stage3_spatial_tokens_plus_planar_backprojection_8x12"
)


def _channel_group_norm(channels: int) -> nn.GroupNorm:
    """Normalize each sample without mutable batch or running statistics."""
    return nn.GroupNorm(32, channels)


class ResNet18BasicBlock(nn.Module):
    """The two-convolution residual block used by ResNet-18."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm_1 = _channel_group_norm(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.convolution_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm_2 = _channel_group_norm(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _channel_group_norm(out_channels),
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        residual = self.skip(value)
        value = self.activation(self.norm_1(self.convolution_1(value)))
        value = self.norm_2(self.convolution_2(value))
        return self.activation(value + residual)


def _resnet18_stage(
    in_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Sequential:
    return nn.Sequential(
        ResNet18BasicBlock(in_channels, out_channels, stride),
        ResNet18BasicBlock(out_channels, out_channels),
    )


class DepthObservationEncoder(nn.Module):
    """Encode ordered depth frames with one shared SanD-style ResNet-18."""

    def __init__(
        self,
        model_dim: int = 256,
        frame_tokens_height: int = 8,
        frame_tokens_width: int = 12,
        dropout: float = 0.0,
        max_depth_m: float = 5.0,
        focal_x_px: float = 166.80851063829786,
        focal_y_px: float = 166.80851063829786,
        camera_forward_offset_m: float = 0.28618,
        camera_downward_pitch_degrees: float = 10.0,
    ) -> None:
        super().__init__()
        if model_dim % 4:
            raise ValueError(
                "model_dim must be divisible by four for 2D position encoding"
            )
        self.model_dim = model_dim
        self.tokens_height = frame_tokens_height
        self.tokens_width = frame_tokens_width
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            _channel_group_norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _resnet18_stage(64, 64, stride=1),
            _resnet18_stage(64, 128, stride=2),
            _resnet18_stage(128, 256, stride=2),
        )
        self.spatial_projection = nn.Conv2d(256, model_dim, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (frame_tokens_height, frame_tokens_width)
        )
        self.planar_projector = PlanarDepthProjector(
            frame_tokens_height,
            frame_tokens_width,
            max_depth_m,
            focal_x_px,
            focal_y_px,
            camera_forward_offset_m,
            camera_downward_pitch_degrees,
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.register_buffer(
            "position_2d",
            self._position_encoding(
                model_dim,
                frame_tokens_height,
                frame_tokens_width,
            ),
            persistent=True,
        )
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
        row_encoding = torch.cat((rows.sin(), rows.cos()), dim=-1)
        column_encoding = torch.cat((columns.sin(), columns.cos()), dim=-1)
        position = torch.cat(
            (
                row_encoding[:, None].expand(-1, width, -1),
                column_encoding[None].expand(height, -1, -1),
            ),
            dim=-1,
        )
        return position.reshape(1, height * width, model_dim)

    def forward(
        self,
        depth: Tensor,
        observation_to_current: Tensor,
        observation_valid: Tensor,
    ) -> DepthFeatures:
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B, T, 1, H, W]")
        if observation_to_current.shape != (*depth.shape[:2], 4):
            raise ValueError("observation_to_current must have shape [B, T, 4]")
        if (
            observation_valid.shape != depth.shape[:2]
            or observation_valid.dtype != torch.bool
        ):
            raise ValueError("observation_valid must be boolean with shape [B, T]")
        batch, frames = depth.shape[:2]
        features = self.backbone(depth.flatten(0, 1))
        features = self.adaptive_pool(self.spatial_projection(features))
        features = features.flatten(2).transpose(1, 2)
        position = self.position_2d.to(device=features.device, dtype=features.dtype)
        planar_points, pooled_depth = self.planar_projector(
            depth,
            observation_to_current,
        )
        geometry = torch.cat(
            (
                planar_points / self.planar_projector.max_depth_m,
                pooled_depth[..., None] / self.planar_projector.max_depth_m,
            ),
            dim=-1,
        )
        geometry_tokens = self.geometry_projection(geometry.to(features.dtype))
        features = self.output_norm(features + position + geometry_tokens.flatten(0, 1))
        tokens = self.dropout(features).reshape(batch, frames, -1, self.model_dim)
        tokens = torch.where(
            observation_valid[:, :, None, None],
            tokens,
            torch.zeros_like(tokens),
        )
        return DepthFeatures(tokens=tokens)
