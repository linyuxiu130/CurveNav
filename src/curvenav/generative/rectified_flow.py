"""Task-goal-conditioned Rectified Flow with a fixed robot-frame origin."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from curvenav.types import EncodedCondition


@dataclass
class RectifiedFlowLoss:
    loss: Tensor
    prediction: Tensor
    velocity_target: Tensor
    state: Tensor
    time: Tensor


class RectifiedFlow(nn.Module):
    """Linear conditional flow over normalized planar spline controls."""

    def __init__(
        self,
        field: nn.Module,
        num_control_points: int,
        inference_steps: int,
        source_cholesky: Tensor,
        source_std_xy: tuple[float, float],
    ) -> None:
        super().__init__()
        if inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        expected = (num_control_points - 1, num_control_points - 1)
        if tuple(source_cholesky.shape) != expected:
            raise ValueError(f"source_cholesky must have shape {expected}")
        source_std = torch.tensor(source_std_xy, dtype=torch.float32)
        valid_source_std = torch.isfinite(source_std) & (source_std > 0)
        if source_std.shape != (2,) or not torch.all(valid_source_std):
            raise ValueError("source_std_xy must contain two positive values")
        self.field = field
        self.num_control_points = num_control_points
        self.inference_steps = inference_steps
        self.register_buffer("source_cholesky", source_cholesky.clone(), persistent=True)
        self.register_buffer("source_std_xy", source_std, persistent=True)

    def draw_source(self, source_mean: Tensor) -> Tensor:
        noise = torch.randn_like(source_mean[:, 1:])
        factor = self.source_cholesky.to(device=noise.device, dtype=noise.dtype)
        source_std = self.source_std_xy.to(device=noise.device, dtype=noise.dtype)
        residual = torch.einsum("ij,bjd->bid", factor, noise * source_std)
        source = self.enforce_origin(source_mean)
        source[:, 1:] = source[:, 1:] + residual
        return source

    def interpolate(self, clean: Tensor, source: Tensor, time: Tensor) -> Tensor:
        time = time.view(-1, 1, 1).to(device=clean.device, dtype=clean.dtype)
        return self.enforce_origin((1.0 - time) * source + time * clean)

    def velocity_target(self, clean: Tensor, source: Tensor) -> Tensor:
        return self.zero_origin(clean - source)

    def training_loss(
        self,
        clean: Tensor,
        source_mean: Tensor,
        condition: EncodedCondition,
    ) -> RectifiedFlowLoss:
        self._validate_controls(clean)
        self._validate_controls(source_mean)
        clean = self.enforce_origin(clean)
        source = self.draw_source(source_mean)
        time = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype)
        state = self.interpolate(clean, source, time)
        target = self.velocity_target(clean, source)
        condition_key_values = self.field.prepare_condition(condition)
        prediction = self.zero_origin(self.field(state, time, condition_key_values))
        loss = (prediction[:, 1:] - target[:, 1:]).float().square().mean()
        return RectifiedFlowLoss(loss, prediction, target, state, time)

    @torch.no_grad()
    def sample(
        self,
        condition: EncodedCondition,
        source_mean: Tensor,
        num_samples: int = 1,
    ) -> Tensor:
        self._validate_controls(source_mean)
        condition = self.repeat_condition(condition, num_samples)
        source_mean = source_mean.repeat_interleave(num_samples, dim=0)
        state = self.draw_source(source_mean)
        condition_key_values = self.field.prepare_condition(condition)
        dt = 1.0 / self.inference_steps
        for step in range(self.inference_steps):
            time = state.new_full((state.shape[0],), step / self.inference_steps)
            velocity = self.zero_origin(self.field(state, time, condition_key_values))
            state = self.enforce_origin(state + dt * velocity)
        return state

    @staticmethod
    def enforce_origin(value: Tensor) -> Tensor:
        result = value.clone()
        result[:, 0] = 0
        return result

    @staticmethod
    def zero_origin(value: Tensor) -> Tensor:
        result = value.clone()
        result[:, 0] = 0
        return result

    @staticmethod
    def repeat_condition(condition: EncodedCondition, repeats: int) -> EncodedCondition:
        if repeats < 1:
            raise ValueError("num_samples must be positive")
        return EncodedCondition(tokens=condition.tokens.repeat_interleave(repeats, dim=0))

    def _validate_controls(self, controls: Tensor) -> None:
        expected = (self.num_control_points, 2)
        if controls.ndim != 3 or tuple(controls.shape[1:]) != expected:
            raise ValueError(f"controls must have shape [B, {expected[0]}, 2]")
