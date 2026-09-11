"""Conditional Flow Transformer over physical curve coordinates."""

import torch
from torch import Tensor, nn
from curvenav.precision import NEURAL_DTYPE

from curvenav.configuration_space import query_configuration_field
from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock, ProjectedCondition
from curvenav.types import ConditionFeatures


TRAJECTORY_DECODER_TYPE = "state_geometry_increment_curve_flow_transformer"


class FlowTimeEmbedding(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(1, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        return self.projection(time[:, None].float())


class PlanarControlReadout(nn.Module):
    """Apply one shared two-dimensional head to every increment token."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(model_dim, 2)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens).flatten(1)


class ConditionalCurveFlowDecoder(nn.Module):
    """Generate curve velocity from one shared scene-conditioned token stream."""

    def __init__(
        self,
        *,
        control_tokens: int,
        coordinate_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        planning_horizon_m: float,
        path_to_increment_weight: Tensor,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("decoder layers must be positive")
        if coordinate_dim != 2 * control_tokens:
            raise ValueError("each trajectory token must be one planar increment")
        self.control_tokens = control_tokens
        self.coordinate_dim = coordinate_dim
        self.planning_horizon_m = float(planning_horizon_m)
        if path_to_increment_weight.shape != (control_tokens, 64):
            raise ValueError("path-to-increment weights must have shape [C,64]")
        self.register_buffer(
            "path_to_increment_weight",
            path_to_increment_weight.float()
            / path_to_increment_weight.float().sum(dim=-1, keepdim=True),
            persistent=True,
        )
        self.state_embedding = nn.Linear(2, model_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, control_tokens, model_dim)
        )
        self.time_embedding = FlowTimeEmbedding(model_dim)
        self.path_geometry_embedding = nn.Sequential(
            nn.Linear(7, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.goal_geometry_embedding = nn.Sequential(
            nn.Linear(5, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(
                model_dim,
                heads,
                dropout,
            )
            for _ in range(layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.velocity_readout = PlanarControlReadout(model_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _trajectory_geometry(
        self,
        candidate_path: Tensor,
        condition: ConditionFeatures,
    ) -> tuple[Tensor, Tensor]:
        query = query_configuration_field(
            condition.configuration_field,
            candidate_path,
            self.planning_horizon_m,
        )
        observed = query.observed_features
        path_features = torch.cat(
            (
                candidate_path.float() / self.planning_horizon_m,
                observed[..., :1] / self.planning_horizon_m,
                observed[..., 1:],
            ),
            dim=-1,
        )
        increment_path_features = torch.einsum(
            "cp,bpf->bcf",
            self.path_to_increment_weight,
            path_features,
        )
        return increment_path_features, torch.einsum(
            "cp,bpf->bcf",
            self.path_to_increment_weight,
            self._goal_geometry(candidate_path, condition),
        )

    def _goal_geometry(
        self,
        candidate_path: Tensor,
        condition: ConditionFeatures,
    ) -> Tensor:
        """Express intent as remaining displacement to one local endpoint."""
        terminal_goal = condition.terminal_goal[:, None, :].expand_as(
            candidate_path
        )
        goal_delta = terminal_goal - candidate_path.float()
        return torch.cat(
            (
                candidate_path.float() / self.planning_horizon_m,
                goal_delta / self.planning_horizon_m,
                torch.linalg.vector_norm(goal_delta, dim=-1, keepdim=True)
                / self.planning_horizon_m,
            ),
            dim=-1,
        )

    def _path_relative_geometry(
        self,
        query_position: Tensor,
        condition: ConditionFeatures,
    ) -> Tensor:
        relative = (
            condition.metric_position[:, None, :, :2]
            - query_position[:, :, None, :].float()
        ) / self.planning_horizon_m
        vertical = (
            condition.metric_position[:, None, :, 2:3] / self.planning_horizon_m
        ).expand(-1, self.control_tokens, -1, -1)
        # Pool the nonlinear distance before reducing path samples. Distance
        # to a pooled position would discard the increment's extended support.
        distance = torch.einsum(
            "cp,bpnf->bcnf",
            self.path_to_increment_weight,
            torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
        )
        relative = torch.einsum(
            "cp,bpnf->bcnf", self.path_to_increment_weight, relative
        )
        return torch.cat(
            (
                relative,
                vertical,
                distance,
                condition.surface_hit[:, None, :, None]
                .expand(-1, self.control_tokens, -1, -1)
                .to(relative.dtype),
                condition.frame_age[:, None, :, None]
                .expand(-1, self.control_tokens, -1, -1)
                .to(relative.dtype),
                condition.motion_token[:, None, :, None]
                .expand(-1, self.control_tokens, -1, -1)
                .to(relative.dtype),
            ),
            dim=-1,
        )

    def project_condition_memory(
        self,
        condition: ConditionFeatures,
    ) -> tuple[ProjectedCondition, ...]:
        """Cache only scene K/V; trajectory geometry changes with flow state."""
        with torch.autocast(
            device_type=condition.tokens.device.type, dtype=NEURAL_DTYPE
        ):
            return tuple(
                block.project_condition(condition) for block in self.blocks
            )

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        candidate_path: Tensor,
        condition: ConditionFeatures,
        memory: tuple[ProjectedCondition, ...],
    ) -> Tensor:
        path_geometry, goal_geometry = self._trajectory_geometry(
            candidate_path, condition
        )
        pair_geometry = self._path_relative_geometry(candidate_path, condition)
        with torch.autocast(device_type=state.device.type, dtype=NEURAL_DTYPE):
            tokens = self.state_embedding(state.reshape(-1, self.control_tokens, 2))
            tokens = (
                tokens
                + self.position_embedding.to(tokens.dtype)
                + self.time_embedding(time)[:, None]
                + self.path_geometry_embedding(path_geometry)
                + self.goal_geometry_embedding(goal_geometry)
            )
            for block, projected in zip(self.blocks, memory, strict=True):
                tokens = block(tokens, projected, pair_geometry)
            return self.velocity_readout(self.output_norm(tokens))
