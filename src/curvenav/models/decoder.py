"""Single-call proposal-grounded improved MeanFlow Transformer."""

import torch
from torch import Tensor, nn

from curvenav.configuration_space import query_configuration_field
from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock, ProjectedCondition
from curvenav.types import ConditionFeatures


TRAJECTORY_DECODER_TYPE = (
    "single_call_flow_state_then_clean_proposal_cspace_improved_mean_flow"
)


class MeanFlowIntervalEmbedding(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )

    def forward(self, start_time: Tensor, end_time: Tensor) -> Tensor:
        return self.projection(
            torch.stack((end_time, end_time - start_time), dim=-1).float()
        )


class PlanarControlReadout(nn.Module):
    """Apply one shared two-dimensional head to every control-point token."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(model_dim, 2)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens).flatten(1)


class ConditionalCurveMeanFlowDecoder(nn.Module):
    """Predict a clean proposal, query its geometry, then predict average flow."""

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
        path_to_control_weight: Tensor,
    ) -> None:
        super().__init__()
        if layers < 2 or layers % 2:
            raise ValueError("decoder layers must split proposal and average phases")
        if coordinate_dim != 2 * control_tokens:
            raise ValueError("each trajectory token must be one planar control")
        self.control_tokens = control_tokens
        self.coordinate_dim = coordinate_dim
        self.layers_per_phase = layers // 2
        self.planning_horizon_m = float(planning_horizon_m)
        if path_to_control_weight.shape != (control_tokens, 64):
            raise ValueError("path-to-control weights must have shape [C,64]")
        self.register_buffer(
            "path_to_control_weight",
            path_to_control_weight.float()
            / path_to_control_weight.float().sum(dim=-1, keepdim=True).clamp_min(1e-6),
            persistent=True,
        )
        self.state_embedding = nn.Linear(2, model_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, control_tokens, model_dim)
        )
        self.time_embedding = MeanFlowIntervalEmbedding(model_dim)
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
        self.instantaneous_velocity_readout = PlanarControlReadout(model_dim)
        self.average_velocity_readout = PlanarControlReadout(model_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _trajectory_geometry(
        self,
        candidate_path: Tensor,
        candidate_controls: Tensor,
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
        control_path_features = torch.einsum(
            "cp,bpf->bcf",
            self.path_to_control_weight.to(path_features),
            path_features,
        )
        goal_delta = condition.goal_reference - candidate_controls.float()
        goal_features = torch.cat(
            (
                candidate_controls.float() / self.planning_horizon_m,
                goal_delta / self.planning_horizon_m,
                torch.linalg.vector_norm(goal_delta, dim=-1, keepdim=True)
                / self.planning_horizon_m,
            ),
            dim=-1,
        )
        return control_path_features, goal_features

    def project_condition_memory(
        self, condition: ConditionFeatures
    ) -> tuple[ProjectedCondition, ...]:
        return tuple(block.project_condition(condition) for block in self.blocks)

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
        ).expand(-1, query_position.shape[1], -1, -1)
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        return torch.cat(
            (
                relative,
                vertical,
                distance,
                condition.surface_hit[:, None, :, None]
                .expand(-1, query_position.shape[1], -1, -1)
                .to(relative.dtype),
                condition.frame_age[:, None, :, None]
                .expand(-1, query_position.shape[1], -1, -1)
                .to(relative.dtype),
                condition.motion_token[:, None, :, None]
                .expand(-1, query_position.shape[1], -1, -1)
                .to(relative.dtype),
            ),
            dim=-1,
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
        if state.ndim != 2 or state.shape[1] != self.coordinate_dim:
            raise ValueError("flow state does not match planar control coordinates")
        if start_time.shape != state.shape[:1] or end_time.shape != state.shape[:1]:
            raise ValueError("MeanFlow interval times must have shape [B]")
        if len(projected_condition) != len(self.blocks):
            raise ValueError("projected condition must cover every decoder block")
        instantaneous_time = self.time_embedding(end_time, end_time)[:, None]
        interval_time = self.time_embedding(start_time, end_time)[:, None]
        base_tokens = self.state_embedding(
            state.reshape(state.shape[0], self.control_tokens, 2)
        )
        if condition.goal_reference.shape != (
            state.shape[0],
            self.control_tokens,
            2,
        ):
            raise ValueError("goal reference must have shape [B,C,2]")
        # Standardized incremental controls are an affine Euclidean chart, so
        # every linear-interpolant Flow state decodes to one physical curve.
        # Query geometry at that state before estimating its clean endpoint;
        # PointGoal supplies relative intent but never relocates the query.
        flow_controls = curve_codec.control_positions_from_coordinates(state)
        flow_path, _ = curve_codec.decode(state)
        flow_path_geometry, flow_goal_geometry = self._trajectory_geometry(
            flow_path,
            flow_controls,
            condition,
        )
        flow_pair_geometry = self._path_relative_geometry(flow_controls, condition)
        proposal_tokens = (
            base_tokens
            + self.position_embedding.to(dtype=base_tokens.dtype)
            + instantaneous_time.to(dtype=base_tokens.dtype)
            + self.path_geometry_embedding(
                flow_path_geometry.to(base_tokens.dtype)
            )
            + self.goal_geometry_embedding(
                flow_goal_geometry.to(base_tokens.dtype)
            )
        )
        for index in range(self.layers_per_phase):
            proposal_tokens = self.blocks[index](
                proposal_tokens,
                projected_condition[index],
                flow_pair_geometry,
            )
        instantaneous = self.instantaneous_velocity_readout(
            self.output_norm(proposal_tokens)
        )

        estimated_clean = state - end_time[:, None] * instantaneous.float()
        estimated_path, _ = curve_codec.decode(estimated_clean)
        estimated_controls = curve_codec.control_positions_from_coordinates(
            estimated_clean
        )
        path_geometry, goal_geometry = self._trajectory_geometry(
            estimated_path,
            estimated_controls,
            condition,
        )
        pair_geometry = self._path_relative_geometry(estimated_controls, condition)
        average_tokens = (
            proposal_tokens
            + (interval_time - instantaneous_time).to(proposal_tokens.dtype)
            + self.path_geometry_embedding(path_geometry.to(proposal_tokens.dtype))
            + self.goal_geometry_embedding(goal_geometry.to(proposal_tokens.dtype))
        )
        for index in range(self.layers_per_phase, len(self.blocks)):
            average_tokens = self.blocks[index](
                average_tokens,
                projected_condition[index],
                pair_geometry,
            )
        average = self.average_velocity_readout(self.output_norm(average_tokens))
        return average[:, None], instantaneous[:, None]
