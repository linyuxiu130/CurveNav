"""Goal-independent scene encoding followed by explicit PointGoal conditioning."""

import math

import torch
from torch import Tensor, nn

from curvenav.layers import EncoderBlock, RMSNorm
from curvenav.types import ConditionFeatures, DepthFeatures

from .motion import HistoricalMotionEncoder
CONDITION_ENCODER_TYPE = (
    "goal_independent_scene_plus_explicit_pointgoal_token"
)


class PolicyConditionEncoder(nn.Module):
    """Preserve scene geometry and expose PointGoal as one separate token."""

    def __init__(
        self,
        configuration_encoder: nn.Module,
        *,
        observation_frames: int,
        spatial_tokens: int,
        configuration_tokens: int,
        history_horizon_m: float,
        model_dim: int = 384,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.configuration_encoder = configuration_encoder
        self.observation_frames = observation_frames
        self.spatial_tokens = spatial_tokens
        self.configuration_tokens = configuration_tokens
        self.configuration_grid_size = math.isqrt(configuration_tokens)
        if self.configuration_grid_size**2 != configuration_tokens:
            raise ValueError("configuration tokens must form one square metric grid")
        self.motion_encoder = HistoricalMotionEncoder(
            observation_frames=observation_frames,
            history_horizon_m=history_horizon_m,
            model_dim=model_dim,
        )
        self.goal_projection = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.context_blocks = nn.ModuleList(
            EncoderBlock(model_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        )
        self.memory_norm = RMSNorm(model_dim)

    def forward(
        self,
        observation: DepthFeatures,
        point_goal: Tensor,
        observation_valid: Tensor,
        observation_to_current: Tensor,
    ) -> ConditionFeatures:
        batch = point_goal.shape[0]
        if observation.tokens.shape[:2] != (batch, self.spatial_tokens):
            raise ValueError("current visual tokens do not match the condition contract")
        if observation.configuration_field.ndim != 4:
            raise ValueError("configuration field must have shape [B,C,H,W]")
        configuration = self.configuration_encoder(
            observation.configuration_field,
            observation.configuration_tokens,
            observation.configuration_points,
            observation.configuration_visual_valid,
        )
        if configuration.tokens.shape[:2] != (batch, self.configuration_tokens):
            raise ValueError(
                "configuration tokens do not match the condition contract"
            )
        motion = self.motion_encoder(observation_to_current, observation_valid)
        # Scene reasoning is deliberately goal independent.  PointGoal cannot
        # rewrite the measured obstacle field.
        tokens = torch.cat(
            (observation.tokens, motion, configuration.tokens), dim=1
        ).float()
        for block in self.context_blocks:
            tokens = block(tokens)
        tokens = self.memory_norm(tokens)
        # The goal is appended after scene self-attention, so it can condition
        # the trajectory decoder without rewriting measured obstacle tokens.
        goal_token = self.goal_projection(
            point_goal.float() / self.configuration_encoder.planning_horizon_m
        )[:, None]
        tokens = torch.cat((tokens, goal_token), dim=1)
        return ConditionFeatures(
            tokens=tokens,
            path_configuration_field=configuration.measured_field,
        )
