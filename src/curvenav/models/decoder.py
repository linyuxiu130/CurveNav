"""Conditional Flow Matching Transformer over executable curve values."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock, MetricPathCrossAttention
from curvenav.types import ConditionFeatures


TRAJECTORY_DECODER_TYPE = "metric_path_conditioned_curve_flow_transformer"


class FlowTimeEmbedding(nn.Module):
    """Smooth embedding of continuous flow time on the closed unit interval."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(1, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        return self.projection(timestep.float()[:, None])


class StructuredCurveReadout(nn.Module):
    """Read one heterogeneous velocity scalar from each ordered curve token."""

    def __init__(self, curve_tokens: int, model_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, curve_tokens, model_dim))
        self.bias = nn.Parameter(torch.zeros(1, curve_tokens))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        return (tokens * self.weight).sum(dim=-1) + self.bias


class ConditionalCurveFlowDecoder(nn.Module):
    """Predict the conditional vector field from curve state and continuous time."""

    def __init__(
        self,
        curve_tokens: int,
        path_tokens: int,
        num_path_points: int,
        planning_horizon_m: float,
        curvature_scale_inv_m: float,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.curve_tokens = curve_tokens
        self.path_tokens = path_tokens
        self.planning_horizon_m = float(planning_horizon_m)
        self.curvature_scale_inv_m = float(curvature_scale_inv_m)
        self.state_embedding = nn.Linear(1, model_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, curve_tokens, model_dim))
        self.path_geometry_embedding = nn.Sequential(
            nn.Linear(8, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.path_position_embedding = nn.Parameter(
            torch.empty(1, path_tokens, model_dim)
        )
        self.time_embedding = FlowTimeEmbedding(model_dim)
        self.metric_path_attention = MetricPathCrossAttention(
            model_dim,
            heads,
            dropout,
            planning_horizon_m,
        )
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.velocity_readout = StructuredCurveReadout(curve_tokens, model_dim)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.path_position_embedding, std=0.02)
        self.register_buffer(
            "path_indices",
            torch.linspace(0, num_path_points - 1, path_tokens).round().long(),
            persistent=True,
        )
        self.register_buffer(
            "path_progress",
            torch.linspace(0.0, 1.0, path_tokens).reshape(1, path_tokens, 1),
            persistent=True,
        )

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        condition: ConditionFeatures,
        path: Tensor,
        heading: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.curve_tokens:
            raise ValueError("flow state does not match decoder tokens")
        if time.shape != state.shape[:1]:
            raise ValueError("flow time must have shape [B]")
        flow_time = self.time_embedding(time)[:, None]
        controls = (
            self.state_embedding(state[..., None])
            + self.position_embedding
            + flow_time
        ).float()
        indices = self.path_indices
        anchor_points = path[:, indices]
        anchor_heading = heading[:, indices]
        anchor_curvature = curvature[:, indices]
        goal_delta = condition.point_goal[:, None] - anchor_points
        path_geometry = torch.cat(
            (
                anchor_points / self.planning_horizon_m,
                anchor_heading.sin()[..., None],
                anchor_heading.cos()[..., None],
                (anchor_curvature / self.curvature_scale_inv_m)[..., None],
                self.path_progress.to(path.dtype).expand(path.shape[0], -1, -1),
                goal_delta / self.planning_horizon_m,
            ),
            dim=-1,
        )
        path_tokens = (
            self.path_geometry_embedding(path_geometry)
            + self.path_position_embedding
            + flow_time
        ).float()
        path_tokens = self.metric_path_attention(
            path_tokens,
            anchor_points,
            condition.current_tokens,
            condition.current_points,
            condition.current_obstacle_valid,
        )
        trajectory = torch.cat((controls, path_tokens), dim=1)
        for block in self.blocks:
            trajectory = block(trajectory, condition.tokens)
        controls = self.output_norm(trajectory[:, : self.curve_tokens])
        return self.velocity_readout(controls)
