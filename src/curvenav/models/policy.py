"""PointGoal-conditioned one-step improved MeanFlow trajectory generation."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from curvenav.precision import GEOMETRY_DTYPE
from curvenav.trajectory import IncrementalBSplineTrajectory
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
        curve_codec: IncrementalBSplineTrajectory,
        planning_horizon_m: float,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_decoder = trajectory_decoder
        self.curve_codec = curve_codec
        self._planning_horizon_m = float(planning_horizon_m)
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

    def _trainable_velocity_primal(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        condition: ConditionFeatures,
        projected_condition: tuple[ProjectedCondition, ...],
    ) -> tuple[Tensor, Tensor]:
        """Expose the same decoder as one separately compiled reverse-mode graph."""
        return self._predict_stage_velocities(
            state,
            start_time,
            end_time,
            condition,
            projected_condition,
        )

    def _mean_flow_total_time_derivative(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        instantaneous_velocity: Tensor,
        condition: ConditionFeatures,
        projected_condition: tuple[ProjectedCondition, ...],
    ) -> Tensor:
        """Evaluate the stopped MeanFlow material derivative in float32."""

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
            raise ValueError(
                "flow_interval_group must have shape [B] on the flow device"
            )
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
        encoded = self.encode_condition(condition)
        clean = self.curve_codec.coordinates_from_values(
            target.curve_values.float(),
        )
        if clean.shape[1] != self.curve_codec.coordinate_dim:
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
        projected_condition = self.trajectory_decoder.project_condition_memory(encoded)

        def stage_velocities(
            flow_state: Tensor,
            start_time: Tensor,
            end_time: Tensor,
        ) -> tuple[Tensor, Tensor]:
            return self._trainable_velocity_primal(
                flow_state,
                start_time,
                end_time,
                encoded,
                projected_condition,
            )

        average_velocities, instantaneous_velocities = stage_velocities(
            state, start_time, end_time
        )
        # The MeanFlow material derivative is multiplied by (t-r), hence it is
        # identically absent on the diagonal half of the training law.  Evaluate
        # its stopped JVP only for positive-width rows; this is algebraically
        # identical to a full-batch JVP followed by the zero interval product.
        positive_width = flow_interval_group.long() < 2
        positive_condition = ConditionFeatures(
            tokens=encoded.tokens[positive_width],
            token_valid=encoded.token_valid[positive_width],
            metric_position=encoded.metric_position[positive_width],
            surface_hit=encoded.surface_hit[positive_width],
            frame_age=encoded.frame_age[positive_width],
            motion_token=encoded.motion_token[positive_width],
            goal_reference=encoded.goal_reference[positive_width],
            configuration_field=encoded.configuration_field[positive_width],
        )
        positive_projected_condition = tuple(
            tuple(value[positive_width] for value in projected)
            for projected in projected_condition
        )
        # The decoder makes v(z_t,t) structurally independent of r, so the
        # trainable primal already provides the exact JVP tangent; no second
        # diagonal decoder evaluation is needed.
        jvp_tangent = instantaneous_velocities[positive_width, -1].float()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=state.device.type,
                enabled=False,
            ),
        ):
            with sdpa_kernel([SDPBackend.MATH]):
                total_time_derivatives = self._mean_flow_total_time_derivative(
                    state[positive_width],
                    start_time[positive_width],
                    end_time[positive_width],
                    jvp_tangent.detach(),
                    positive_condition,
                    positive_projected_condition,
                )
        # The material derivative is a geometric/Flow quantity.  Keep its
        # interval integration in float32 instead of quantizing it back into
        # the neural autocast dtype before forming the MeanFlow target.
        interval_correction = torch.zeros_like(average_velocities, dtype=GEOMETRY_DTYPE)
        interval_correction[positive_width] = (end_time - start_time)[
            positive_width, None, None
        ] * total_time_derivatives.detach()
        reparameterized_velocities = average_velocities.float() + interval_correction
        stage_target = conditional_velocity[:, None]
        instantaneous_error = instantaneous_velocities.float() - stage_target
        average_error = reparameterized_velocities.float() - stage_target
        instantaneous_squared_error = instantaneous_error.square()
        average_squared_error = average_error.square()
        mean_flow_loss = 0.5 * (
            instantaneous_squared_error.mean() + average_squared_error.mean()
        )
        deployment_coordinates = (
            flow_source[deployment] - average_velocities[deployment, -1].float()
        )
        deployment_path, _ = self.curve_codec.decode(deployment_coordinates)
        visible_clearance_loss = observed_clearance_loss(
            deployment_path,
            encoded.configuration_field[deployment],
            self.planning_horizon_m,
        ).sum().mul(4.0 / clean.shape[0])
        return CurveNavLoss(
            loss=mean_flow_loss + visible_clearance_loss,
            mean_flow_loss=mean_flow_loss,
            visible_clearance_loss=visible_clearance_loss,
        )

    @torch.no_grad()
    def _deployment_transport(
        self, condition: PolicyCondition
    ) -> tuple[Tensor, Tensor]:
        """Return final and internal clean-proposal coordinates from one call."""
        encoded = self.encode_condition(condition)
        state = self.inference_source.expand(condition.point_goal.shape[0], -1).clone()
        start_time = torch.zeros(state.shape[0], device=state.device)
        end_time = torch.ones_like(start_time)
        average_velocities, instantaneous_velocities = self._predict_stage_velocities(
            state,
            start_time,
            end_time,
            encoded,
            self.trajectory_decoder.project_condition_memory(encoded),
        )
        final = state - average_velocities[:, -1].float()
        proposal = state - instantaneous_velocities[:, -1].float()
        return final, proposal

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        final, proposal = self._deployment_transport(condition)
        path, _ = self.curve_codec.decode(final)
        return TrajectoryPrediction(path=path, proposal_coordinates=proposal)

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
