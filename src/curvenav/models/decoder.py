"""Conditional MeanFlow Transformer over executable curve values."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock
from curvenav.models.safety import sample_configuration_field
from curvenav.types import ConditionFeatures


TRAJECTORY_DECODER_TYPE = (
    "path_relative_configuration_refined_curve_mean_flow_transformer"
)
DECODER_REFINEMENT_STAGES = 3


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
    """Predict average velocity while refining its own data-end trajectory."""

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
        if layers % DECODER_REFINEMENT_STAGES:
            raise ValueError("decoder layers must divide into three refinement stages")
        self.curve_tokens = curve_tokens
        self.path_tokens = path_tokens
        self.planning_horizon_m = float(planning_horizon_m)
        self.layers_per_stage = layers // DECODER_REFINEMENT_STAGES
        self.state_embedding = nn.Linear(1, model_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, curve_tokens, model_dim))
        self.path_geometry_embedding = nn.Sequential(
            nn.Linear(12, model_dim),
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

    def _read_velocity(self, controls: Tensor) -> Tensor:
        return self.velocity_readout(self.output_norm(controls))

    def _path_tokens(
        self,
        state: Tensor,
        end_time: Tensor,
        velocity: Tensor,
        condition: ConditionFeatures,
        flow_time: Tensor,
        curve_codec: nn.Module,
    ) -> Tensor:
        # For the production data-anchored field, z_t - t*u estimates the
        # actual data endpoint.  Geometry is therefore queried on the curve
        # being generated, never on the independent Gaussian flow state.
        estimated_clean = state - end_time[:, None] * velocity
        path, heading = curve_codec.decode_path(estimated_clean)
        indices = self.path_indices
        anchor_points = path[:, indices]
        anchor_heading = heading[:, indices]
        goal_delta = condition.point_goal[:, None] - anchor_points
        field = sample_configuration_field(
            condition.configuration_field,
            anchor_points,
            self.planning_horizon_m,
        )
        normalized_field = torch.cat(
            (
                field[..., :1] / self.planning_horizon_m,
                field[..., 1:],
            ),
            dim=-1,
        )
        geometry = torch.cat(
            (
                anchor_points / self.planning_horizon_m,
                anchor_heading.sin()[..., None],
                anchor_heading.cos()[..., None],
                self.path_progress.to(path.dtype).expand(path.shape[0], -1, -1),
                goal_delta / self.planning_horizon_m,
                normalized_field,
            ),
            dim=-1,
        )
        return (
            self.path_geometry_embedding(geometry)
            + self.path_position_embedding
            + flow_time
        ).float()

    def forward(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        condition: ConditionFeatures,
        curve_codec: nn.Module,
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
        trajectory = controls
        for stage in range(DECODER_REFINEMENT_STAGES):
            if stage:
                proposal = self._read_velocity(
                    trajectory[:, : self.curve_tokens]
                )
                path_tokens = self._path_tokens(
                    state,
                    end_time,
                    proposal,
                    condition,
                    flow_time,
                    curve_codec,
                )
                trajectory = torch.cat(
                    (trajectory[:, : self.curve_tokens], path_tokens), dim=1
                )
            begin = stage * self.layers_per_stage
            end = begin + self.layers_per_stage
            for block in self.blocks[begin:end]:
                trajectory = block(trajectory, condition.tokens)
        return self._read_velocity(trajectory[:, : self.curve_tokens])
