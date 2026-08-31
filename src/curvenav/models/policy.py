"""PointGoal-conditioned one-step improved MeanFlow trajectory generation."""

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

from .blocks import ProjectedCondition
from .safety import observed_clearance_loss
TRAINING_LOSS_NAMES = (
    "loss",
    "mean_flow_loss",
    "visible_clearance_loss",
)
INFERENCE_SOURCE_SEED = 20_260_828
MEAN_FLOW_TIME_SAMPLING = (
    "half_diagonal_quarter_logit_normal_interval_quarter_deployment_boundary"
)
MEAN_FLOW_LOGIT_NORMAL_MEAN = -0.4
MEAN_FLOW_LOGIT_NORMAL_STD = 1.0
@dataclass
class CurveNavLoss:
    loss: Tensor
    mean_flow_loss: Tensor
    visible_clearance_loss: Tensor

    def logging_values(self) -> tuple[Tensor, ...]:
        return (
            self.loss,
            self.mean_flow_loss,
            self.visible_clearance_loss,
        )


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

    def _predict_stage_velocities(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        condition: ConditionFeatures,
        projected_condition: tuple[ProjectedCondition, ...],
    ) -> tuple[Tensor, Tensor]:
        return self.trajectory_decoder(
            state,
            start_time,
            end_time,
            condition,
            projected_condition,
            self.curve_codec,
        )

    def _mean_flow_total_time_derivative(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        instantaneous_velocity: Tensor,
        condition: ConditionFeatures,
    ) -> Tensor:
        """Evaluate the stopped MeanFlow material derivative in float32."""
        projected_condition = self.trajectory_decoder.project_condition_memory(
            condition.tokens
        )

        def average_velocity(
            flow_state: Tensor,
            start_time: Tensor,
            end_time: Tensor,
        ) -> Tensor:
            return self._predict_stage_velocities(
                flow_state,
                start_time,
                end_time,
                condition,
                projected_condition,
            )[0]

        return torch.func.jvp(
            average_velocity,
            (state, start_time, end_time),
            (
                instantaneous_velocity,
                torch.zeros_like(start_time),
                torch.ones_like(end_time),
            ),
        )[1]

    @staticmethod
    def _training_intervals(
        flow_interval_group: Tensor,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample the iMF diagonal/interior law plus the exact deployed boundary."""
        if (
            flow_interval_group.shape != reference.shape[:1]
            or flow_interval_group.device != reference.device
        ):
            raise ValueError("flow_interval_group must have shape [B] on the flow device")
        batch_size = reference.shape[0]
        group = flow_interval_group.long()
        deployment = group == 0
        diagonal = group >= 2
        sampled = torch.sigmoid(
            torch.randn(batch_size, 2, device=reference.device, dtype=reference.dtype)
            * MEAN_FLOW_LOGIT_NORMAL_STD
            + MEAN_FLOW_LOGIT_NORMAL_MEAN
        )
        start_time = sampled.min(dim=1).values
        end_time = sampled.max(dim=1).values
        start_time = torch.where(diagonal, end_time, start_time)
        start_time = torch.where(deployment, torch.zeros_like(start_time), start_time)
        end_time = torch.where(deployment, torch.ones_like(end_time), end_time)
        return start_time, end_time, deployment

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
        flow_interval_group: Tensor,
    ) -> CurveNavLoss:
        target.validate()
        clean = self.curve_codec.coordinates_from_values(target.curve_values.float())
        if clean.shape[1] != self.curve_codec.num_curve_tokens:
            raise ValueError("target curve values do not match the production codec")
        if source.shape != clean.shape:
            raise ValueError(
                "flow source must match the standardized curve coordinates"
            )
        source = source.float()
        start_time, end_time, deployment = self._training_intervals(
            flow_interval_group, clean
        )
        flow_source = source.clone()
        flow_source[deployment] = self.inference_source.to(flow_source)
        state = (1.0 - end_time[:, None]) * clean + end_time[:, None] * flow_source
        conditional_velocity = flow_source - clean
        encoded = self.encode_condition(condition)
        projected_condition = self.trajectory_decoder.project_condition_memory(
            encoded.tokens
        )

        def stage_velocities(
            flow_state: Tensor,
            start_time: Tensor,
            end_time: Tensor,
        ) -> tuple[Tensor, Tensor]:
            return self._predict_stage_velocities(
                flow_state,
                start_time,
                end_time,
                encoded,
                projected_condition,
            )

        # Only the forward-mode tangent needs the full float32/MATH route.
        # It is a stopped regression target, so retaining its reverse-mode
        # graph wastes memory and prevents the trainable primal evaluations
        # from using the surrounding mixed-precision autocast route.
        _, diagonal_instantaneous_velocities = stage_velocities(
            state, end_time, end_time
        )
        jvp_tangent = diagonal_instantaneous_velocities[:, -1]
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=state.device.type,
                enabled=False,
            ),
        ):
            with sdpa_kernel([SDPBackend.MATH]):
                total_time_derivatives = self._mean_flow_total_time_derivative(
                    state,
                    start_time,
                    end_time,
                    jvp_tangent.detach(),
                    encoded,
                )
        average_velocities, instantaneous_velocities = stage_velocities(
            state, start_time, end_time
        )
        interval = end_time - start_time
        reparameterized_velocities = average_velocities + interval[:, None, None] * (
            total_time_derivatives.detach()
        )
        stage_target = conditional_velocity[:, None]
        instantaneous_error = instantaneous_velocities.float() - stage_target
        average_error = reparameterized_velocities.float() - stage_target
        instantaneous_squared_error = instantaneous_error.square()
        average_squared_error = average_error.square()
        mean_flow_loss = 0.5 * (
            instantaneous_squared_error.mean() + average_squared_error.mean()
        )
        predicted_deployment_coordinates = (
            flow_source - average_velocities[:, -1].float()
        )
        predicted_deployment_path, _ = self.curve_codec.decode(
            predicted_deployment_coordinates[deployment]
        )
        visible_clearance_loss = observed_clearance_loss(
            predicted_deployment_path,
            encoded.path_configuration_field[deployment],
            self.planning_horizon_m,
        ).sum().mul(4.0 / clean.shape[0])
        return CurveNavLoss(
            loss=mean_flow_loss + visible_clearance_loss,
            mean_flow_loss=mean_flow_loss,
            visible_clearance_loss=visible_clearance_loss,
        )

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        state = self.inference_source.expand(condition.point_goal.shape[0], -1).clone()
        start_time = torch.zeros(state.shape[0], device=state.device)
        end_time = torch.ones_like(start_time)
        average_velocities, _ = self._predict_stage_velocities(
            state,
            start_time,
            end_time,
            encoded,
            self.trajectory_decoder.project_condition_memory(encoded.tokens),
        )
        state = state - average_velocities[:, -1].float()
        path, _ = self.curve_codec.decode(state)
        return TrajectoryPrediction(path=path)

    def forward(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
        flow_interval_group: Tensor,
    ) -> CurveNavLoss:
        return self.training_loss(
            condition,
            target,
            source,
            flow_interval_group,
        )
