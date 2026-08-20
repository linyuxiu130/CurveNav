"""Scale-only normalization that preserves the robot-frame origin."""

import torch
from torch import Tensor, nn


class PlanarScaleNormalizer(nn.Module):
    def __init__(self, scale_xy: tuple[float, float]) -> None:
        super().__init__()
        scale = torch.tensor(scale_xy, dtype=torch.float32)
        if scale.shape != (2,) or torch.any(scale <= 0):
            raise ValueError("scale_xy must contain two positive values")
        self.register_buffer("scale_xy", scale)

    def normalize(self, value: Tensor) -> Tensor:
        return value / self.scale_xy.to(device=value.device, dtype=value.dtype)

    def denormalize(self, value: Tensor) -> Tensor:
        return value * self.scale_xy.to(device=value.device, dtype=value.dtype)
