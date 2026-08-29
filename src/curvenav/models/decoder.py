"""Conditional MeanFlow Transformer over executable curve values."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.types import ConditionFeatures


TRAJECTORY_DECODER_TYPE = (
    "complete_configuration_memory_conditioned_curve_mean_flow_transformer"
)


class MeanFlowIntervalEmbedding(nn.Module):
    """Embed interval endpoint and width for the average velocity field."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )

    def forward(self, start_time: Tensor, end_time: Tensor) -> Tensor:
        interval = torch.stack((end_time, end_time - start_time), dim=-1)
        return self.projection(interval.float())


class StructuredCurveReadout(nn.Module):
    """Read one heterogeneous velocity scalar from each ordered curve token."""

    def __init__(self, curve_tokens: int, model_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, curve_tokens, model_dim))
        self.bias = nn.Parameter(torch.zeros(1, curve_tokens))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        return (tokens * self.weight).sum(dim=-1) + self.bias


class ConditionalCurveMeanFlowDecoder(nn.Module):
    """Predict conditional average velocity over a continuous flow interval."""

    def __init__(
        self,
        curve_tokens: int,
        path_tokens: int,
        num_path_points: int,
        planning_horizon_m: float,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.curve_tokens = curve_tokens
        self.path_tokens = path_tokens
        self.planning_horizon_m = float(planning_horizon_m)
        self.state_embedding = nn.Linear(1, model_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, curve_tokens, model_dim))
        self.path_geometry_embedding = nn.Sequential(
            nn.Linear(7, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.path_position_embedding = nn.Parameter(
            torch.empty(1, path_tokens, model_dim)
        )
        self.time_embedding = MeanFlowIntervalEmbedding(model_dim)
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.velocity_readout = StructuredCurveReadout(curve_tokens, model_dim)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.path_position_embedding, std=0.02)
        path_indices = torch.linspace(
            0, num_path_points - 1, path_tokens
        ).round().long()
        self.register_buffer("path_indices", path_indices, persistent=True)
        self.register_buffer(
            "path_progress",
            (path_indices.float() / (num_path_points - 1)).reshape(
                1, path_tokens, 1
            ),
            persistent=True,
        )

    def forward(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        condition: ConditionFeatures,
        path: Tensor,
        heading: Tensor,
    ) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.curve_tokens:
            raise ValueError("flow state does not match decoder tokens")
        if start_time.shape != state.shape[:1] or end_time.shape != state.shape[:1]:
            raise ValueError("mean-flow interval times must have shape [B]")
        flow_time = self.time_embedding(start_time, end_time)[:, None]
        controls = (
            self.state_embedding(state[..., None])
            + self.position_embedding
            + flow_time
        ).float()
        indices = self.path_indices
        anchor_points = path[:, indices]
        anchor_heading = heading[:, indices]
        goal_delta = condition.point_goal[:, None] - anchor_points
        path_geometry = torch.cat(
            (
                anchor_points / self.planning_horizon_m,
                anchor_heading.sin()[..., None],
                anchor_heading.cos()[..., None],
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
        trajectory = torch.cat((controls, path_tokens), dim=1)
        for block in self.blocks:
            trajectory = block(trajectory, condition.tokens)
        controls = self.output_norm(trajectory[:, : self.curve_tokens])
        return self.velocity_readout(controls)
