"""Goal-independent current vision with explicit causal motion conditioning."""

import torch
from torch import Tensor, nn

from curvenav.layers import EncoderBlock, RMSNorm
from curvenav.types import ConditionFeatures, DepthFeatures

from .motion import HistoricalMotionEncoder


CONDITION_ENCODER_TYPE = "joint_goal_visual_configuration_and_causal_motion_tokens"


class PolicyConditionEncoder(nn.Module):
    """Encode goal, current vision and causal state without latent geometry loss."""

    def __init__(
        self,
        point_goal_encoder: nn.Module,
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
        self.point_goal_encoder = point_goal_encoder
        self.configuration_encoder = configuration_encoder
        self.observation_frames = observation_frames
        self.spatial_tokens = spatial_tokens
        self.configuration_tokens = configuration_tokens
        self.motion_encoder = HistoricalMotionEncoder(
            observation_frames=observation_frames,
            history_horizon_m=history_horizon_m,
            model_dim=model_dim,
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
        goal = self.point_goal_encoder(point_goal).unsqueeze(1)
        configuration = self.configuration_encoder(
            observation.configuration_field
        )
        if configuration.shape[:2] != (batch, self.configuration_tokens):
            raise ValueError(
                "configuration tokens do not match the condition contract"
            )
        motion = self.motion_encoder(observation_to_current, observation_valid)
        # Keep configuration cells as first-class memory.  Folding them into
        # visual/goal queries and then discarding them destroys the spatial
        # correspondence needed by the trajectory decoder.
        tokens = torch.cat(
            (goal, observation.tokens, motion, configuration), dim=1
        ).float()
        for block in self.context_blocks:
            tokens = block(tokens)
        return ConditionFeatures(
            tokens=self.memory_norm(tokens),
            configuration_field=observation.configuration_field.float(),
            point_goal=point_goal.float(),
        )
