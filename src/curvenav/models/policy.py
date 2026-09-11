"""PointGoal-conditioned Flow Matching in physical curve coordinates."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from curvenav.trajectory import IncrementalBSplineTrajectory
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)


TRAINING_LOSS_NAMES = ("loss",)
INFERENCE_SOURCE_SEED = 20_260_828
FLOW_TIME_SAMPLING = "single_draw_logit_normal"
FLOW_LOGIT_NORMAL_MEAN = -0.4
FLOW_LOGIT_NORMAL_STD = 1.0


@dataclass
class CurveNavLoss:
    loss: Tensor

    def logging_values(self) -> tuple[Tensor, ...]:
        return (self.loss,)


class CurveNavPolicy(nn.Module):
    """Learn conditional curve transport; deploy one reproducible source sample."""

    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_decoder: nn.Module,
        curve_codec: IncrementalBSplineTrajectory,
        planning_horizon_m: float,
        integration_steps: int,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_decoder = trajectory_decoder
        self.curve_codec = curve_codec
        self._planning_horizon_m = float(planning_horizon_m)
        self.integration_steps = integration_steps
        if (
            trajectory_decoder.control_tokens != curve_codec.num_control_tokens
            or trajectory_decoder.coordinate_dim != curve_codec.coordinate_dim
        ):
            raise ValueError("trajectory decoder and curve token counts differ")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(INFERENCE_SOURCE_SEED)
        inference_source = torch.randn(
            curve_codec.coordinate_dim,
            generator=generator,
        )
        inference_source.mul_(
            math.sqrt(curve_codec.coordinate_dim) / inference_source.norm()
        )
        self.register_buffer(
            "inference_source",
            inference_source.unsqueeze(0),
            persistent=True,
        )

    @property
    def planning_horizon_m(self) -> float:
        return self._planning_horizon_m

    def encode_condition(self, condition: PolicyCondition) -> ConditionFeatures:
        condition.validate()
        observation = self.depth_encoder(condition)
        return self.condition_encoder(
            observation,
            condition.point_goal,
            condition.observation_valid,
            condition.observation_to_current,
            condition.observation_age_s,
        )

    def _predict_velocity(self, state: Tensor, time: Tensor, memory) -> Tensor:
        return self.trajectory_decoder(state, time, memory).float()

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
    ) -> CurveNavLoss:
        target.validate()
        clean = self.curve_codec.coordinates_from_values(target.curve_values.float())
        if source.shape != clean.shape:
            raise ValueError(
                "flow source must match the standardized curve coordinates"
            )
        source = source.float()
        encoded = self.encode_condition(condition)
        memory = self.trajectory_decoder.project_condition_memory(
            encoded, self.curve_codec
        )
        time = torch.sigmoid(
            torch.randn(len(clean), device=clean.device) * FLOW_LOGIT_NORMAL_STD
            + FLOW_LOGIT_NORMAL_MEAN
        )
        state = (1 - time[:, None]) * clean + time[:, None] * source
        velocity = self._predict_velocity(state, time, memory)
        return CurveNavLoss((velocity - (source - clean)).square().mean())

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        memory = self.trajectory_decoder.project_condition_memory(
            encoded, self.curve_codec
        )
        state = self.inference_source.expand(len(condition.point_goal), -1)
        # Reverse the data-to-noise interpolant, reusing the same scene memory.
        for index in range(self.integration_steps):
            time = torch.full(
                (len(state),), 1 - index / self.integration_steps, device=state.device
            )
            state = (
                state
                - self._predict_velocity(state, time, memory) / self.integration_steps
            )
        path, _ = self.curve_codec.decode(state)
        return TrajectoryPrediction(path=path)

    def forward(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
    ) -> CurveNavLoss:
        return self.training_loss(condition, target, source)
