"""Conditional rectified flow over planar B-spline control points."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


TRAJECTORY_FLOW_TYPE = "conditional_bspline_rectified_flow_heun"


class FourierTimeEmbedding(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even for the flow time embedding")
        self.model_dim = model_dim
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        half = self.model_dim // 2
        frequency = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        phase = time[:, None] * frequency[None] * (2.0 * math.pi)
        return self.projection(torch.cat((phase.sin(), phase.cos()), dim=-1))


class SplineControlFlow(nn.Module):
    """Predict the straight-path conditional flow velocity from noise to data."""

    def __init__(
        self,
        num_control_points: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        inference_candidates: int,
        inference_seed: int,
    ) -> None:
        super().__init__()
        self.num_control_points = num_control_points
        self.inference_candidates = inference_candidates
        self.control_projection = nn.Linear(2, model_dim)
        self.control_embedding = nn.Parameter(
            torch.empty(1, num_control_points, model_dim)
        )
        self.time_embedding = FourierTimeEmbedding(model_dim)
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.velocity_projection = nn.Linear(model_dim, 2)
        nn.init.trunc_normal_(self.control_embedding, std=0.02)
        nn.init.normal_(self.velocity_projection.weight, std=0.02)
        nn.init.zeros_(self.velocity_projection.bias)

        generator = torch.Generator(device="cpu").manual_seed(inference_seed)
        noise = torch.randn(
            inference_candidates,
            num_control_points,
            2,
            generator=generator,
        )
        noise[:, 0] = 0
        self.register_buffer("inference_noise", noise, persistent=True)

    def forward(
        self,
        noisy_controls: Tensor,
        time: Tensor,
        condition_tokens: Tensor,
    ) -> Tensor:
        if noisy_controls.ndim != 3 or noisy_controls.shape[1:] != (
            self.num_control_points,
            2,
        ):
            raise ValueError("noisy_controls must have shape [B, K, 2]")
        if time.shape != (noisy_controls.shape[0],):
            raise ValueError("time must have shape [B]")
        trajectory = (
            self.control_projection(noisy_controls)
            + self.control_embedding
            + self.time_embedding(time).unsqueeze(1)
        )
        for block in self.blocks:
            trajectory = block(trajectory, condition_tokens)
        velocity = self.velocity_projection(self.output_norm(trajectory))
        velocity = velocity.clone()
        velocity[:, 0] = 0
        return velocity

    def training_pair(self, clean_controls: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        noise = torch.randn_like(clean_controls)
        noise[:, 0] = 0
        time = torch.rand(
            clean_controls.shape[0],
            device=clean_controls.device,
            dtype=clean_controls.dtype,
        )
        noisy = torch.lerp(noise, clean_controls, time[:, None, None])
        return noisy, time, clean_controls - noise

    @staticmethod
    def reconstruct_clean(
        noisy_controls: Tensor,
        time: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        """Recover the clean endpoint of the linear flow path from ``v``.

        For ``x_t=(1-t)z+t x_1`` and ``v*=x_1-z``, the endpoint is exactly
        ``x_1=x_t+(1-t)v*``.  Keeping this relation explicit prevents the path
        auxiliary loss from silently becoming a diffusion-style x0 estimator.
        """
        return noisy_controls + (1.0 - time[:, None, None]) * velocity

    def sample(
        self,
        condition: ConditionFeatures,
        integration_steps: int,
    ) -> Tensor:
        batch = condition.tokens.shape[0]
        candidates = self.inference_candidates
        controls = self.inference_noise.to(
            device=condition.tokens.device,
            dtype=condition.tokens.dtype,
        ).unsqueeze(0).expand(batch, -1, -1, -1).reshape(
            batch * candidates, self.num_control_points, 2
        ).clone()
        memory = condition.tokens[:, None].expand(
            -1, candidates, -1, -1
        ).reshape(batch * candidates, condition.tokens.shape[1], condition.tokens.shape[2])
        step_size = 1.0 / integration_steps
        for index in range(integration_steps):
            time = torch.full(
                (batch * candidates,),
                index * step_size,
                device=controls.device,
                dtype=controls.dtype,
            )
            next_time = torch.full_like(time, (index + 1) * step_size)
            velocity = self(controls, time, memory)
            predictor = controls + step_size * velocity
            corrected = self(predictor, next_time, memory)
            controls = controls + 0.5 * step_size * (velocity + corrected)
            controls[:, 0] = 0
        return controls.reshape(batch, candidates, self.num_control_points, 2)
