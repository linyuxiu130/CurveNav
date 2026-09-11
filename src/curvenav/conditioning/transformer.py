"""Target-independent fused metric memory and a PointGoal reference."""

import torch
from torch import Tensor, nn

from curvenav.layers import RMSNorm
from curvenav.trajectory import local_terminal_goal, metric_horizon_reference
from curvenav.types import ConditionFeatures, DepthFeatures

from .motion import HistoricalMotionEncoder


CONDITION_ENCODER_TYPE = "observed_cspace_visual_bev_motion_plus_metric_horizon_slots"


class PolicyConditionEncoder(nn.Module):
    """Build target-independent scene memory and separate metric goal intent."""

    def __init__(
        self,
        configuration_encoder: nn.Module,
        *,
        observation_frames: int,
        planning_horizon_m: float,
        control_tokens: int,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.configuration_encoder = configuration_encoder
        self.planning_horizon_m = float(planning_horizon_m)
        if control_tokens != 7:
            raise ValueError("goal reference must match seven B-spline controls")
        self.motion_encoder = HistoricalMotionEncoder(
            observation_frames=observation_frames,
            translation_scale_m=planning_horizon_m,
            model_dim=model_dim,
        )
        self.memory_norm = RMSNorm(model_dim)

    def _metric_reference(self, batch_size: int, device: torch.device) -> Tensor:
        """Return target-independent forward slots for metric scene retrieval."""
        return metric_horizon_reference(
            batch_size,
            self.planning_horizon_m,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self,
        observation: DepthFeatures,
        point_goal: Tensor,
        observation_valid: Tensor,
        observation_to_current: Tensor,
        observation_age_s: Tensor,
    ) -> ConditionFeatures:
        batch = point_goal.shape[0]
        if observation.tokens.ndim != 3 or observation.tokens.shape[0] != batch:
            raise ValueError("depth tokens must have shape [B,N,D]")
        if observation.token_valid.shape != observation.tokens.shape[:2]:
            raise ValueError("depth token validity must have shape [B,N]")
        configuration = self.configuration_encoder(
            observation.configuration_field,
            observation.tokens,
            observation.metric_position,
            observation.token_valid,
        )
        motion = self.motion_encoder(
            observation_to_current, observation_valid, observation_age_s
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
                configuration.observed_fraction > 0.0,
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
        metric_reference = self._metric_reference(batch, point_goal.device)
        terminal_goal = local_terminal_goal(point_goal, self.planning_horizon_m)
        return ConditionFeatures(
            tokens=self.memory_norm(tokens),
            token_valid=token_valid,
            metric_position=metric_position,
            surface_hit=surface_hit,
            frame_age=frame_age,
            motion_token=motion_token,
            metric_reference=metric_reference,
            terminal_goal=terminal_goal,
            configuration_field=observation.configuration_field,
        )
