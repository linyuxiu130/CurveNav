"""Deterministic conditional flow over future executable-curve coordinates."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


TRAJECTORY_FLOW_TYPE = (
    "zero_source_future_bounded_curvature_rectified_flow_adarmszero_heun"
)


class FourierTimeEmbedding(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even for the flow time embedding")
        self.model_dim = model_dim
        self.register_buffer(
            "frequency",
            torch.logspace(0.0, 3.0, model_dim // 2, dtype=torch.float32),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        frequency = self.frequency.to(device=time.device)
        phase = time.float()[:, None] * frequency[None] * (2.0 * math.pi)
        return self.projection(torch.cat((phase.sin(), phase.cos()), dim=-1))


class CurvatureTrajectoryFlow(nn.Module):
    """Generate one future curve from a zero-source conditional flow.

    For expert curve coordinates ``x_1`` the conditional probability path is
    ``x_t = t x_1`` and its exact velocity target is ``v_t = x_1``.  Executed
    history is already present in the condition tokens, so it is never noised,
    reconstructed, or used to rank stochastic future samples.
    """

    def __init__(
        self,
        future_tokens: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if future_tokens < 1:
            raise ValueError("future_tokens must be positive")
        self.future_tokens = future_tokens
        self.state_projection = nn.Linear(2, model_dim)
        self.token_embedding = nn.Parameter(
            torch.empty(1, future_tokens, model_dim)
        )
        self.time_embedding = FourierTimeEmbedding(model_dim)
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.velocity_projection = nn.Linear(model_dim, 2)
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        nn.init.zeros_(self.velocity_projection.weight)
        nn.init.zeros_(self.velocity_projection.bias)

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        condition_tokens: Tensor,
        route_token: Tensor,
    ) -> Tensor:
        if state.ndim != 3 or state.shape[1:] != (
            self.future_tokens,
            2,
        ):
            raise ValueError("state does not match the future token contract")
        if time.shape != (state.shape[0],):
            raise ValueError("time must have shape [B]")
        if route_token.shape != (state.shape[0], self.token_embedding.shape[-1]):
            raise ValueError("route_token must have shape [B,D]")
        trajectory = self.state_projection(state) + self.token_embedding
        modulation = self.time_embedding(time) + route_token
        for block in self.blocks:
            trajectory = block(trajectory, condition_tokens, modulation)
        return self.velocity_projection(self.output_norm(trajectory))

    def training_path(
        self,
        clean_state: Tensor,
        free_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if clean_state.shape != free_mask.shape:
            raise ValueError("clean_state and free_mask must have identical shapes")
        time = torch.rand(
            clean_state.shape[0],
            device=clean_state.device,
            dtype=clean_state.dtype,
        )
        clean_state = clean_state * free_mask
        state = time[:, None, None] * clean_state
        return state, time, clean_state

    @staticmethod
    def reconstruct_clean(
        state: Tensor,
        time: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        return state + (1.0 - time[:, None, None]) * velocity

    def integrate(
        self,
        condition: ConditionFeatures,
        integration_steps: int,
        free_mask: Tensor,
    ) -> Tensor:
        """Integrate the unique zero-source future state with Heun's method."""
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        batch = condition.tokens.shape[0]
        if free_mask.shape != (batch, self.future_tokens, 2):
            raise ValueError("free_mask must have shape [B,T,2]")
        state = torch.zeros(
            batch,
            self.future_tokens,
            2,
            device=condition.tokens.device,
            dtype=condition.tokens.dtype,
        )
        mask = free_mask.to(dtype=state.dtype)
        step_size = 1.0 / integration_steps
        for index in range(integration_steps):
            time = torch.full(
                (batch,),
                index * step_size,
                device=state.device,
                dtype=state.dtype,
            )
            next_time = torch.full_like(time, (index + 1) * step_size)
            velocity = self(
                state,
                time,
                condition.tokens,
                condition.route_token,
            ) * mask
            predictor = (state + step_size * velocity) * mask
            corrected = self(
                predictor,
                next_time,
                condition.tokens,
                condition.route_token,
            ) * mask
            state = (state + 0.5 * step_size * (velocity + corrected)) * mask
        return state
