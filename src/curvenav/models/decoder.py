"""Conditional MeanFlow Transformer over executable curve values."""

import torch
from torch import Tensor, nn

from curvenav.encoders.configuration import (
    PATH_CONFIGURATION_FIELD_CHANNELS,
    observed_configuration_features,
)
from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock, ProjectedCondition
from curvenav.models.safety import sample_configuration_field
from curvenav.types import ConditionFeatures


TRAJECTORY_DECODER_TYPE = (
    "pointgoal_and_configuration_field_conditioned_curve_mean_flow_transformer"
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
            nn.Linear(5 + PATH_CONFIGURATION_FIELD_CHANNELS, model_dim),
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
        self.average_velocity_readout = StructuredCurveReadout(curve_tokens, model_dim)
        self.instantaneous_velocity_readout = StructuredCurveReadout(
            curve_tokens, model_dim
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.path_position_embedding, std=0.02)
        path_indices = (
            torch.linspace(0, num_path_points - 1, path_tokens).round().long()
        )
        self.register_buffer("path_indices", path_indices, persistent=True)
        self.register_buffer(
            "path_progress",
            (path_indices.float() / (num_path_points - 1)).reshape(1, path_tokens, 1),
            persistent=True,
        )
        field_scale = torch.ones(PATH_CONFIGURATION_FIELD_CHANNELS)
        field_scale[0] = 1.0 / self.planning_horizon_m
        self.register_buffer(
            "path_configuration_field_scale",
            field_scale.reshape(1, 1, -1),
            persistent=True,
        )

    def _read_velocities(self, controls: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.output_norm(controls)
        return (
            self.average_velocity_readout(normalized),
            self.instantaneous_velocity_readout(normalized),
        )

    def project_condition_memory(
        self,
        condition_tokens: Tensor,
    ) -> tuple[ProjectedCondition, ...]:
        return tuple(block.project_condition(condition_tokens) for block in self.blocks)

    def _path_tokens(
        self,
        state: Tensor,
        end_time: Tensor,
        instantaneous_velocity: Tensor,
        path_configuration_field: Tensor,
        flow_time: Tensor,
        curve_codec: nn.Module,
    ) -> Tensor:
        # On the linear interpolant, x = z_t - t*(e-x).  Replacing the
        # conditional velocity by its learned marginal estimate therefore
        # gives the data endpoint independently of the MeanFlow start r.
        estimated_clean = state - end_time[:, None] * instantaneous_velocity
        path, heading = curve_codec.decode_path(estimated_clean)
        indices = self.path_indices
        anchor_points = path[:, indices]
        anchor_heading = heading[:, indices]
        field = observed_configuration_features(
            sample_configuration_field(
                path_configuration_field,
                anchor_points,
                self.planning_horizon_m,
            ),
            channel_dim=-1,
        )
        normalized_field = field * self.path_configuration_field_scale
        geometry = torch.cat(
            (
                anchor_points / self.planning_horizon_m,
                anchor_heading.sin()[..., None],
                anchor_heading.cos()[..., None],
                self.path_progress.to(path.dtype).expand(path.shape[0], -1, -1),
                normalized_field,
            ),
            dim=-1,
        )
        embedded = self.path_geometry_embedding(geometry)
        return (
            embedded
            + self.path_position_embedding.to(dtype=embedded.dtype)
            + flow_time.to(dtype=embedded.dtype)
        )

    def forward(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        condition: ConditionFeatures,
        projected_condition: tuple[ProjectedCondition, ...],
        curve_codec: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        if state.ndim != 2 or state.shape[1] != self.curve_tokens:
            raise ValueError("flow state does not match decoder tokens")
        if start_time.shape != state.shape[:1] or end_time.shape != state.shape[:1]:
            raise ValueError("mean-flow interval times must have shape [B]")
        if len(projected_condition) != len(self.blocks):
            raise ValueError("projected condition must cover every decoder block")
        flow_time = self.time_embedding(start_time, end_time)[:, None]
        controls = self.state_embedding(state[..., None])
        controls = (
            controls
            + self.position_embedding.to(dtype=controls.dtype)
            + flow_time.to(dtype=controls.dtype)
        )
        trajectory = controls
        average_velocities = []
        instantaneous_velocities = []
        path_configuration_field = condition.path_configuration_field
        for stage in range(DECODER_REFINEMENT_STAGES):
            if stage:
                path_tokens = self._path_tokens(
                    state,
                    end_time,
                    instantaneous_velocities[-1],
                    path_configuration_field,
                    flow_time,
                    curve_codec,
                )
                trajectory = torch.cat(
                    (trajectory[:, : self.curve_tokens], path_tokens), dim=1
                )
            begin = stage * self.layers_per_stage
            end = begin + self.layers_per_stage
            for index in range(begin, end):
                trajectory = self.blocks[index](
                    trajectory,
                    projected_condition[index],
                )
            average, instantaneous = self._read_velocities(
                trajectory[:, : self.curve_tokens]
            )
            average_velocities.append(average)
            instantaneous_velocities.append(instantaneous)
        return (
            torch.stack(average_velocities, dim=1),
            torch.stack(instantaneous_velocities, dim=1),
        )
