"""Residual depth backbone with fixed four-frame temporal channel fusion."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm


DEPTH_ENCODER_REVISION = "stride4_temporal_stack_dilated_residual_v4"


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = next(
        candidate
        for candidate in range(min(8, channels), 0, -1)
        if channels % candidate == 0
    )
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class ResidualDepthBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm_1 = _group_norm(out_channels)
        self.convolution_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm_2 = _group_norm(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        residual = self.skip(value)
        value = torch.nn.functional.silu(self.norm_1(self.convolution_1(value)))
        value = self.norm_2(self.convolution_2(value))
        return torch.nn.functional.silu(value + residual)


class DilatedResidualDepthBlock(nn.Module):
    """Refine the lowest-resolution map once while preserving its maximum reach."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False,
        )
        self.norm = _group_norm(channels)

    def forward(self, value: Tensor) -> Tensor:
        return torch.nn.functional.silu(self.norm(self.convolution(value)) + value)


class DepthPatchStem(nn.Sequential):
    """Overlapping stride-4 stem that removes redundant high-resolution compute."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=4,
            padding=2,
            bias=False,
        )
        super().__init__(
            convolution,
            _group_norm(out_channels),
            nn.SiLU(),
        )


class DepthSequenceEncoder(nn.Module):
    """Encode exactly four ordered depth frames into spatial condition tokens."""

    def __init__(
        self,
        model_dim: int = 256,
        frame_tokens_per_side: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.tokens_per_side = frame_tokens_per_side
        stage_2_dim = max(model_dim // 2, 64)
        self.backbone = nn.Sequential(
            DepthPatchStem(4, 64),
            ResidualDepthBlock(64, stage_2_dim, stride=2),
            ResidualDepthBlock(stage_2_dim, model_dim, stride=2),
            DilatedResidualDepthBlock(model_dim),
            nn.AdaptiveAvgPool2d((frame_tokens_per_side, frame_tokens_per_side)),
        )
        spatial_tokens = frame_tokens_per_side**2
        self.spatial_embedding = nn.Parameter(torch.zeros(1, spatial_tokens, model_dim))
        self.output_norm = RMSNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)

    def forward(self, depth: Tensor) -> Tensor:
        if depth.ndim != 5 or tuple(depth.shape[1:3]) != (4, 1):
            raise ValueError("depth must have shape [B, 4, 1, H, W]")
        features = self.backbone(depth.squeeze(2))
        features = features.flatten(2).transpose(1, 2)
        return self.dropout(self.output_norm(features + self.spatial_embedding))
