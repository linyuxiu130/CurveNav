"""PointGoal-conditioned one-step MeanFlow generation of an executable curve."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from curvenav.trajectory import MetricHeadingTrajectory
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)

from .safety import configuration_space_risk_loss


TRAINING_LOSS_NAMES = ("loss", "mean_flow_loss", "safety_loss")
INFERENCE_SOURCE_SEED = 20_260_828


@dataclass
class CurveNavLoss:
    loss: Tensor
    mean_flow_loss: Tensor
    safety_loss: Tensor

    def logging_values(self) -> tuple[Tensor, ...]:
        return self.loss, self.mean_flow_loss, self.safety_loss


class CurveNavPolicy(nn.Module):
    """Learn a deterministic conditional transport to one executable curve."""

    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_decoder: nn.Module,
        curve_codec: MetricHeadingTrajectory,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_decoder = trajectory_decoder
        self.curve_codec = curve_codec
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
            observation,
            condition.point_goal,
            condition.observation_valid,
            condition.observation_to_current,
        )

    def _predict_mean_velocity(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        condition: ConditionFeatures,
    ) -> Tensor:
        return self.trajectory_decoder(
            state,
            start_time,
            end_time,
            condition,
            self.curve_codec,
        )

    @staticmethod
    def _closed_interval_times(batch_size: int, reference: Tensor) -> Tensor:
        """Deterministically collocate the complete data-to-noise interval."""
        if batch_size == 1:
            return torch.ones(1, device=reference.device, dtype=reference.dtype)
        return torch.linspace(
            0.0,
            1.0,
            batch_size,
            device=reference.device,
            dtype=reference.dtype,
        )

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
    ) -> CurveNavLoss:
        target.validate()
        clean = self.curve_codec.coordinates_from_values(target.curve_values.float())
        if clean.shape[1] != self.curve_codec.num_curve_tokens:
            raise ValueError("target curve values do not match the production codec")
        if source.shape != clean.shape:
            raise ValueError("flow source must match the standardized curve coordinates")
        source = source.float()
        time = self._closed_interval_times(clean.shape[0], clean)
        state = (1.0 - time[:, None]) * clean + time[:, None] * source
        conditional_velocity = source - clean
        encoded = self.encode_condition(condition)

        def mean_velocity(
            flow_state: Tensor,
            start_time: Tensor,
            end_time: Tensor,
        ) -> Tensor:
            return self._predict_mean_velocity(
                flow_state,
                start_time,
                end_time,
                encoded,
            )

        instantaneous_velocity = mean_velocity(state, time, time)
        zero = torch.zeros_like(time)
        with sdpa_kernel([SDPBackend.MATH]):
            average_velocity, total_time_derivative = torch.func.jvp(
                mean_velocity,
                (state, zero, time),
                (
                    instantaneous_velocity.detach(),
                    zero,
                    torch.ones_like(time),
                ),
            )
        reparameterized_velocity = average_velocity + time[:, None] * (
            total_time_derivative.detach()
        )
        instantaneous_error = (
            instantaneous_velocity.float() - conditional_velocity
        )
        average_error = reparameterized_velocity.float() - conditional_velocity
        mean_flow_loss = 0.5 * (
            instantaneous_error.square().mean() + average_error.square().mean()
        )

        predicted_clean = state - time[:, None] * average_velocity.float()
        predicted_path, _ = self.curve_codec.decode_path(predicted_clean)
        safety_loss = configuration_space_risk_loss(
            predicted_path,
            encoded.configuration_field,
            self.planning_horizon_m,
        )
        return CurveNavLoss(
            loss=mean_flow_loss + safety_loss,
            mean_flow_loss=mean_flow_loss,
            safety_loss=safety_loss,
        )

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        state = self.inference_source.expand(condition.point_goal.shape[0], -1).clone()
        start_time = torch.zeros(state.shape[0], device=state.device)
        end_time = torch.ones_like(start_time)
        average_velocity = self._predict_mean_velocity(
            state,
            start_time,
            end_time,
            encoded,
        ).float()
        state = state - average_velocity
        path, _ = self.curve_codec.decode(state)
        return TrajectoryPrediction(path=path)

    def forward(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
    ) -> CurveNavLoss:
        return self.training_loss(condition, target, source)
