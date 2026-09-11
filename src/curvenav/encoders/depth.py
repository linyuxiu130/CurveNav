"""Shared depth backbone with timed metric scene evidence."""

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from curvenav.precision import NEURAL_DTYPE

from curvenav.layers import RMSNorm
from curvenav.types import DepthFeatures

from .geometry import MetricDepthProjector


DEPTH_ENCODER_TYPE = "depth_pixel_lift_se3_memory_v2"


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
    """Encode every registered depth frame with one shared visual backbone.

    Each image token retains its learned appearance feature and receives the
    corresponding measured ray endpoint transformed into the current robot
    frame. The same projection constructs observed robot configuration space;
    no PointGoal or map completion enters perception.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        frame_tokens_height: int,
        frame_tokens_width: int,
        dropout: float,
        max_depth_m: float,
        planning_horizon_m: float,
    ) -> None:
        super().__init__()
        self.max_depth_m = float(max_depth_m)

        self.backbone = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False),
            _channel_group_norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            _resnet18_stage(64, 64, 1),
            _resnet18_stage(64, 128, 2),
            _resnet18_stage(128, 256, 2),
        )
        self.spatial_projection = nn.Conv2d(256, model_dim, 1)
        self.metric_projector = MetricDepthProjector(
            frame_tokens_height,
            frame_tokens_width,
            max_depth_m,
            planning_horizon_m,
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.output_norm = RMSNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def sample_pixel_features(features: Tensor, indices: Tensor, image_width: int) -> Tensor:
        """Lift features at the exact selected depth pixel, on the stride-16 lattice.

        Symmetric odd-kernel padding gives feature centre i at input index 16*i.
        align_corners=False maps that index explicitly; border extension is the
        defined interpolation outside the final feature centre, not a new ray.
        """
        x = (indices % image_width).float() / 16
        y = (indices // image_width).float() / 16
        grid = torch.stack((2 * (x + .5) / features.shape[-1] - 1,
                            2 * (y + .5) / features.shape[-2] - 1), dim=-1)
        # FP32 sampling preserves pixel coordinates under neural BF16 autocast.
        sampled = F.grid_sample(features.float(), grid.flatten(0, 1).unsqueeze(1),
                                mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.squeeze(2).transpose(1, 2).reshape(*indices.shape, features.shape[1]).to(features.dtype)

    def forward(self, condition) -> DepthFeatures:
        depth = condition.depth
        visual_input = torch.cat((depth.float(), (depth > 0).float()), dim=2)
        with torch.autocast(device_type=depth.device.type, dtype=NEURAL_DTYPE):
            visual = self.spatial_projection(self.backbone(visual_input.flatten(0, 1)))
        projection = self.metric_projector(condition)
        visual = self.sample_pixel_features(visual, projection.pixel_indices, depth.shape[-1])
        points = projection.points
        selected_depth = projection.depth / self.max_depth_m
        surface_hit = projection.depth < self.max_depth_m
        body_obstacle = (
            projection.obstacle_valid & condition.observation_valid[..., None]
        )
        token_valid = projection.token_valid & (projection.depth < self.max_depth_m)
        geometry = torch.cat(
            (
                points / self.max_depth_m,
                selected_depth[..., None],
                surface_hit[..., None].to(points.dtype),
                body_obstacle[..., None].to(points.dtype),
            ),
            dim=-1,
        )
        with torch.autocast(device_type=depth.device.type, dtype=NEURAL_DTYPE):
            tokens = visual + self.geometry_projection(geometry.to(visual.dtype))
            tokens = (
                tokens
                + self.time_projection(
                    condition.observation_age_s[..., None].to(tokens.dtype)
                )[:, :, None]
            )
        tokens = tokens * token_valid[..., None]
        tokens = self.dropout(self.output_norm(tokens)).flatten(1, 2)
        return DepthFeatures(
            tokens=tokens,
            token_valid=token_valid.flatten(1),
            metric_position=points.flatten(1, 2),
            configuration_field=projection.configuration_field,
        )
