"""Conditional rectified flow over past motion and future spline coordinates."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


TRAJECTORY_FLOW_TYPE = "past_future_bounded_curvature_rectified_flow_adarmszero_heun"


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
    """Jointly reconstruct executed history and generate feasible curve state."""

    def __init__(
        self,
        future_tokens: int,
        history_tokens: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if history_tokens < 1:
            raise ValueError("history_tokens must be positive")
        self.future_tokens = future_tokens
        self.history_tokens = history_tokens
        self.total_tokens = history_tokens + future_tokens
        self.state_projection = nn.Linear(2, model_dim)
        self.token_embedding = nn.Parameter(
            torch.empty(1, self.total_tokens, model_dim)
        )
        self.role_embedding = nn.Parameter(torch.empty(1, 2, model_dim))
        self.time_embedding = FourierTimeEmbedding(model_dim)
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.velocity_projection = nn.Linear(model_dim, 2)
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        nn.init.trunc_normal_(self.role_embedding, std=0.02)
        nn.init.zeros_(self.velocity_projection.weight)
        nn.init.zeros_(self.velocity_projection.bias)

    def forward(
        self,
        noisy_state: Tensor,
        time: Tensor,
        condition_tokens: Tensor,
        route_token: Tensor,
    ) -> Tensor:
        if noisy_state.ndim != 3 or noisy_state.shape[1:] != (
            self.total_tokens,
            2,
        ):
            raise ValueError(
                "noisy_state does not match the past-future token contract"
            )
        if time.shape != (noisy_state.shape[0],):
            raise ValueError("time must have shape [B]")
        if route_token.shape != (noisy_state.shape[0], self.token_embedding.shape[-1]):
            raise ValueError("route_token must have shape [B,D]")
        roles = torch.cat(
            (
                self.role_embedding[:, :1].expand(-1, self.history_tokens, -1),
                self.role_embedding[:, 1:].expand(-1, self.future_tokens, -1),
            ),
            dim=1,
        )
        trajectory = self.state_projection(noisy_state) + self.token_embedding + roles
        modulation = self.time_embedding(time) + route_token
        for block in self.blocks:
            trajectory = block(trajectory, condition_tokens, modulation)
        return self.velocity_projection(self.output_norm(trajectory))

    def training_pair(
        self,
        clean_state: Tensor,
        free_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if clean_state.shape != free_mask.shape:
            raise ValueError("clean_state and free_mask must have identical shapes")
        noise = torch.randn_like(clean_state) * free_mask
        time = torch.rand(
            clean_state.shape[0],
            device=clean_state.device,
            dtype=clean_state.dtype,
        )
        noisy = torch.lerp(noise, clean_state, time[:, None, None]) * free_mask
        return noisy, time, (clean_state - noise) * free_mask

    @staticmethod
    def reconstruct_clean(
        noisy_state: Tensor,
        time: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        return noisy_state + (1.0 - time[:, None, None]) * velocity

    def integrate_candidates(
        self,
        condition: ConditionFeatures,
        integration_steps: int,
        base_samples: Tensor,
        free_mask: Tensor,
    ) -> Tensor:
        """Integrate a fixed candidate group in one batched Heun solve."""
        if base_samples.ndim != 3 or base_samples.shape[1:] != (
            self.total_tokens,
            2,
        ):
            raise ValueError("base_samples must have shape [C,T,2]")
        batch = condition.tokens.shape[0]
        candidates = base_samples.shape[0]
        if free_mask.shape != (batch, self.total_tokens, 2):
            raise ValueError("free_mask must have shape [B,T,2]")

        state = (
            base_samples.to(
                device=condition.tokens.device,
                dtype=condition.tokens.dtype,
            )[None]
            .expand(batch, -1, -1, -1)
            .clone()
        )
        mask = free_mask[:, None].to(dtype=state.dtype).expand_as(state)
        state = state * mask
        state = state.reshape(batch * candidates, self.total_tokens, 2)
        mask = mask.reshape_as(state)
        memory = (
            condition.tokens[:, None]
            .expand(-1, candidates, -1, -1)
            .reshape(batch * candidates, condition.tokens.shape[1], -1)
        )
        route = (
            condition.route_token[:, None]
            .expand(-1, candidates, -1)
            .reshape(batch * candidates, -1)
        )
        step_size = 1.0 / integration_steps
        for index in range(integration_steps):
            time = torch.full(
                (batch * candidates,),
                index * step_size,
                device=state.device,
                dtype=state.dtype,
            )
            next_time = torch.full_like(time, (index + 1) * step_size)
            velocity = self(state, time, memory, route) * mask
            predictor = (state + step_size * velocity) * mask
            corrected = self(predictor, next_time, memory, route) * mask
            state = (state + 0.5 * step_size * (velocity + corrected)) * mask
        return state.reshape(batch, candidates, self.total_tokens, 2)
