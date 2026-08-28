"""PointGoal-conditioned Flow Matching generation of one executable curve."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from curvenav.trajectory import MetricCurvatureTrajectory
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)

from .safety import configuration_space_clearance_loss


TRAINING_LOSS_NAMES = ("loss", "flow_loss", "clearance_loss")
INFERENCE_SOURCE_SEED = 20_260_828
FLOW_SOURCE_ENDPOINT_PROBABILITY = 1.0 / 9.0


@dataclass
class CurveNavLoss:
    loss: Tensor
    flow_loss: Tensor
    clearance_loss: Tensor

    def logging_values(self) -> tuple[Tensor, ...]:
        return self.loss, self.flow_loss, self.clearance_loss


class CurveNavPolicy(nn.Module):
    """Learn a conditional transport from Gaussian curve coordinates."""

    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_decoder: nn.Module,
        curve_codec: MetricCurvatureTrajectory,
        flow_steps: int,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_decoder = trajectory_decoder
        self.curve_codec = curve_codec
        self.flow_steps = flow_steps
        if trajectory_decoder.curve_tokens != curve_codec.num_curve_tokens:
            raise ValueError("trajectory decoder and curve token counts differ")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(INFERENCE_SOURCE_SEED)
        inference_source = torch.randn(
            curve_codec.num_curve_tokens,
            generator=generator,
        )
        inference_source.mul_(
            math.sqrt(curve_codec.num_curve_tokens) / inference_source.norm()
        )
        self.register_buffer(
            "inference_source",
            inference_source.unsqueeze(0),
            persistent=True,
        )

    @property
    def planning_horizon_m(self) -> float:
        return self.trajectory_decoder.planning_horizon_m

    def encode_condition(self, condition: PolicyCondition) -> ConditionFeatures:
        condition.validate()
        depth = torch.where(
            condition.observation_valid[:, :, None, None, None],
            condition.depth,
            torch.zeros_like(condition.depth),
        )
        observation = self.depth_encoder(
            depth,
            condition.observation_to_current.float(),
            condition.observation_valid,
        )
        return self.condition_encoder(
            observation, condition.point_goal, condition.observation_valid
        )

    def _predict_velocity(
        self,
        state: Tensor,
        time: Tensor,
        condition: ConditionFeatures,
    ) -> Tensor:
        path, heading, curvature = self.curve_codec.decode(state)
        return self.trajectory_decoder(
            state,
            time,
            condition,
            path,
            heading,
            curvature,
        )

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
    ) -> CurveNavLoss:
        target.validate()
        clean = self.curve_codec.coordinates_from_values(target.curve_values.float())
        if clean.shape[1] != self.curve_codec.num_curve_tokens:
            raise ValueError("target curve values do not match the production codec")
        source = torch.randn_like(clean)
        time = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype)
        source_endpoint = torch.rand_like(time) < FLOW_SOURCE_ENDPOINT_PROBABILITY
        time = torch.where(source_endpoint, torch.zeros_like(time), time)
        state = (1.0 - time[:, None]) * source + time[:, None] * clean
        target_velocity = clean - source
        encoded = self.encode_condition(condition)
        predicted_velocity = self._predict_velocity(state, time, encoded)
        velocity_error = predicted_velocity.float() - target_velocity
        flow_loss = velocity_error.square().mean()
        predicted_clean = state + (1.0 - time[:, None]) * predicted_velocity.float()
        predicted_path, _, _ = self.curve_codec.decode(predicted_clean)
        clearance_loss = configuration_space_clearance_loss(
            predicted_path,
            encoded.current_points,
            encoded.current_obstacle_valid,
        )
        return CurveNavLoss(
            loss=flow_loss + clearance_loss,
            flow_loss=flow_loss,
            clearance_loss=clearance_loss,
        )

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        state = self.inference_source.expand(condition.point_goal.shape[0], -1).clone()
        step_size = 1.0 / self.flow_steps
        for index in range(self.flow_steps):
            time = torch.full(
                (state.shape[0],), index * step_size, device=state.device
            )
            velocity = self._predict_velocity(state, time, encoded).float()
            predictor = state + step_size * velocity
            next_time = torch.full_like(time, (index + 1) * step_size)
            next_velocity = self._predict_velocity(
                predictor, next_time, encoded
            ).float()
            state = state + 0.5 * step_size * (velocity + next_velocity)
        path, heading, curvature = self.curve_codec.decode(state)
        return TrajectoryPrediction(path=path, heading=heading, curvature=curvature)

    def forward(self, condition: PolicyCondition, target: TrajectoryTarget) -> CurveNavLoss:
        return self.training_loss(condition, target)
