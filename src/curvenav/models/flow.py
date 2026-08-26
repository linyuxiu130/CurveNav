"""Conditional flow matching over future executable-curve coordinates."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


TRAJECTORY_FLOW_TYPE = (
    "conditioned_curve_source_self_consistent_bounded_curvature_rectified_flow_"
    "adarmszero_heun"
)
FLOW_CURVE_COORDINATE_SCALE = 8.0
FLOW_TRAINING_SOURCE_TYPE = "deterministic_conditioned_curve_proposal"
FLOW_INFERENCE_SOURCE_TYPE = "same_conditioned_curve_proposal"
FLOW_SELF_CONSISTENCY_WEIGHT = 0.1


@dataclass(frozen=True)
class FlowPrediction:
    """Local velocity and data endpoint predicted from one shared flow state."""

    velocity: Tensor
    endpoint: Tensor


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
    """Refine one conditioned executable curve through a self-consistent flow.

    For conditioned proposal ``b(c)`` and expert curve coordinates ``x_1``, the
    probability path is ``x_t = (1-t)b(c) + t x_1`` and its exact velocity is
    ``x_1 - b(c)``.  Training and deterministic inference therefore start from
    the identical curve state instead of using different points of a Gaussian.
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
        self.endpoint_projection = nn.Linear(model_dim, 2)
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        nn.init.zeros_(self.velocity_projection.bias)
        nn.init.zeros_(self.endpoint_projection.bias)

    @staticmethod
    def normalize_curve_coordinates(curve_coordinates: Tensor) -> Tensor:
        """Map compact geometric coordinates to the O(1) Flow state."""
        return curve_coordinates * FLOW_CURVE_COORDINATE_SCALE

    @staticmethod
    def denormalize_curve_coordinates(flow_state: Tensor) -> Tensor:
        """Recover the exact geometric coordinates consumed by the codec."""
        return flow_state / FLOW_CURVE_COORDINATE_SCALE

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        condition_tokens: Tensor,
        route_token: Tensor,
    ) -> FlowPrediction:
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
        trajectory = self.output_norm(trajectory)
        return FlowPrediction(
            velocity=self.velocity_projection(trajectory),
            endpoint=self.endpoint_projection(trajectory),
        )

    def training_path(
        self,
        clean_state: Tensor,
        source_state: Tensor,
        free_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if clean_state.shape != source_state.shape or clean_state.shape != free_mask.shape:
            raise ValueError(
                "clean_state, source_state and free_mask must have identical shapes"
            )
        time = torch.rand(
            clean_state.shape[0],
            device=clean_state.device,
            dtype=clean_state.dtype,
        )
        clean_state = clean_state * free_mask
        source_state = source_state * free_mask
        state = (
            (1.0 - time[:, None, None]) * source_state
            + time[:, None, None] * clean_state
        )
        return state, time, clean_state - source_state

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
        source_state: Tensor,
        integration_steps: int,
        free_mask: Tensor,
    ) -> Tensor:
        """Transport the conditioned proposal with Heun's method."""
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        batch = condition.tokens.shape[0]
        if free_mask.shape != (batch, self.future_tokens, 2):
            raise ValueError("free_mask must have shape [B,T,2]")
        mask = free_mask.to(
            device=condition.tokens.device,
            dtype=condition.tokens.dtype,
        )
        if source_state.shape != mask.shape:
            raise ValueError("source_state must have shape [B,T,2]")
        state = source_state.to(
            device=condition.tokens.device,
            dtype=condition.tokens.dtype,
        ) * mask
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
            ).velocity * mask
            predictor = (state + step_size * velocity) * mask
            corrected = self(
                predictor,
                next_time,
                condition.tokens,
                condition.route_token,
            ).velocity * mask
            state = (state + 0.5 * step_size * (velocity + corrected)) * mask
        return state
