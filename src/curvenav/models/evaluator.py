"""A single goal-conditioned critic over the complete decoded metric curve."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from curvenav.configuration_space import query_configuration_field
from curvenav.layers import RMSNorm
from curvenav.models.blocks import ConditionalTrajectoryBlock, ProjectedCondition
from curvenav.precision import NEURAL_DTYPE
from curvenav.types import ConditionFeatures


EVALUATOR_TYPE = "goal_conditioned_pointwise_geometry_single_score"
EVALUATOR_LAYERS = 2


class TrajectoryEvaluator(nn.Module):
    def __init__(self, model_dim: int, heads: int, planning_horizon_m: float):
        super().__init__()
        self.horizon = planning_horizon_m
        # Position, observed geometry, tangent, arc progress, and mission goal.
        self.input_embedding = nn.Sequential(
            nn.Linear(13, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, 0.0) for _ in range(EVALUATOR_LAYERS)
        )
        self.norm = RMSNorm(model_dim)
        self.pool_query = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.pool_query, std=model_dim ** -0.5)
        self.readout = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, 1)
        )

    def project_condition_memory(self, condition: ConditionFeatures) -> tuple[ProjectedCondition, ...]:
        with torch.autocast(device_type=condition.tokens.device.type, dtype=NEURAL_DTYPE):
            return tuple(block.project_condition(condition) for block in self.blocks)

    def forward(
        self,
        path: Tensor,
        goal: Tensor,
        condition: ConditionFeatures,
        memory: tuple[ProjectedCondition, ...],
    ) -> Tensor:
        path = path.float()
        segments = path[:, 1:] - path[:, :-1]
        length = segments.norm(dim=-1)
        tangent = F.normalize(segments, dim=-1)
        tangent = torch.cat((tangent[:, :1], tangent), dim=1)
        arc = F.pad(length.cumsum(-1), (1, 0)) / self.horizon
        goal_length = goal.float().norm(dim=-1, keepdim=True)
        mission = torch.cat(
            (F.normalize(goal.float(), dim=-1), torch.log1p(goal_length / self.horizon)),
            dim=-1,
        )
        observed = query_configuration_field(
            condition.configuration_field, path, self.horizon
        ).observed_features
        features = torch.cat((path / self.horizon, observed[..., :1] / self.horizon,
                              observed[..., 1:], tangent, arc[..., None],
                              mission[:, None].expand(-1, path.shape[1], -1)), -1)
        relative = (condition.metric_position[:, None, :, :2].float() - path[:, :, None]) / self.horizon
        shape = (*relative.shape[:-1], 1)
        pair = torch.cat((
            relative,
            condition.metric_position[:, None, :, 2:3].float().expand(shape) / self.horizon,
            relative.norm(dim=-1, keepdim=True),
            condition.surface_hit[:, None, :, None].expand(shape).float(),
            condition.frame_age[:, None, :, None].expand(shape).float(),
            condition.motion_token[:, None, :, None].expand(shape).float(),
        ), -1)
        with torch.autocast(device_type=path.device.type, dtype=NEURAL_DTYPE):
            tokens = self.input_embedding(features)
            for block, projected in zip(self.blocks, memory, strict=True):
                tokens = block(tokens, projected, pair)
            tokens = self.norm(tokens).float()
        # Pool in FP32; the learned query can emphasize a local bottleneck.
        weights = (tokens * self.pool_query).sum(-1).softmax(-1)
        pooled = (weights[..., None] * tokens).sum(1)
        with torch.autocast(device_type=path.device.type, dtype=NEURAL_DTYPE):
            return self.readout(pooled).squeeze(-1).float()
