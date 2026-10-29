"""Target-independent fused metric memory and terminal PointGoal intent."""

import torch
from torch import nn

from curvenav.layers import RMSNorm
from curvenav.types import ConditionFeatures

from .motion import HistoricalMotionEncoder


CONDITION_ENCODER_TYPE = "observed_cspace_visual_bev_motion_terminal_intent"


class PolicyConditionEncoder(nn.Module):
    """Build target-independent scene memory and separate metric goal intent."""

    def __init__(
        self,
        *,
        observation_frames: int,
        planning_horizon_m: float,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.planning_horizon_m = float(planning_horizon_m)
        self.motion_encoder = HistoricalMotionEncoder(
            observation_frames=observation_frames,
            translation_scale_m=planning_horizon_m,
            model_dim=model_dim,
        )
        self.memory_norm = RMSNorm(model_dim)


    def encode_configuration(self, configuration, field, point_goal, observation_valid,
                             observation_to_current, observation_age_s):
        """Consume the same BEV whether reconstructed offline or updated online."""
        batch = point_goal.shape[0]
        motion = self.motion_encoder(
            observation_to_current, observation_age_s
        )
        tokens = torch.cat((configuration.tokens, motion), dim=1)
        configuration_valid = torch.ones(
            batch,
            configuration.tokens.shape[1],
            device=point_goal.device,
            dtype=torch.bool,
        )
        token_valid = torch.cat((configuration_valid, observation_valid[:, :-1]), dim=1)
        past_position = observation_to_current[:, :-1, :3, 3].float()
        metric_position = torch.cat(
            (configuration.metric_position, past_position), dim=1
        )
        surface_hit = torch.cat(
            (
                configuration.observed_fraction,
                torch.zeros_like(observation_valid[:, :-1]),
            ),
            dim=1,
        )
        frame_age = torch.cat(
            (
                torch.zeros_like(configuration.observed_fraction),
                observation_age_s[:, :-1],
            ),
            dim=1,
        )
        motion_token = torch.cat(
            (
                torch.zeros_like(configuration_valid),
                torch.ones_like(observation_valid[:, :-1]),
            ),
            dim=1,
        )
        terminal_goal = point_goal.float()
        return ConditionFeatures(
            tokens=self.memory_norm(tokens) * token_valid[..., None],
            token_valid=token_valid,
            metric_position=metric_position,
            surface_hit=surface_hit,
            frame_age=frame_age,
            motion_token=motion_token,
            terminal_goal=terminal_goal,
            goal_present=torch.ones_like(point_goal[:, :1]),
            configuration_field=field,
        )
