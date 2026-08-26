"""History-conditioned generation of one executable local curve."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from curvenav.models.flow import FLOW_SELF_CONSISTENCY_WEIGHT
from curvenav.trajectory import BoundedCurvatureTrajectory, PlanarBSplineCodec
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)


TRAINING_LOSS_NAMES = (
    "loss",
    "flow_loss",
    "flow_velocity_loss",
    "flow_endpoint_loss",
    "flow_consistency_loss",
    "path_loss",
    "tangent_loss",
    "proposal_loss",
    "proposal_coordinate_loss",
    "proposal_path_loss",
    "proposal_tangent_loss",
)


@dataclass
class CurveNavLoss:
    loss: Tensor
    flow_loss: Tensor
    flow_velocity_loss: Tensor
    flow_endpoint_loss: Tensor
    flow_consistency_loss: Tensor
    path_loss: Tensor
    tangent_loss: Tensor
    proposal_loss: Tensor
    proposal_coordinate_loss: Tensor
    proposal_path_loss: Tensor
    proposal_tangent_loss: Tensor

    def logging_values(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name) for name in TRAINING_LOSS_NAMES)


class CurveNavPolicy(nn.Module):
    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        curve_proposal: nn.Module,
        trajectory_flow: nn.Module,
        curve_codec: BoundedCurvatureTrajectory,
        target_codec: PlanarBSplineCodec,
        integration_steps: int,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.curve_proposal = curve_proposal
        self.trajectory_flow = trajectory_flow
        self.curve_codec = curve_codec
        self.target_codec = target_codec
        self.integration_steps = integration_steps
        if self.trajectory_flow.future_tokens != self.curve_codec.num_curve_tokens:
            raise ValueError("flow and curve token counts differ")
        if self.curve_proposal.curve_tokens != self.curve_codec.num_curve_tokens:
            raise ValueError("proposal and curve token counts differ")

        progress = torch.linspace(0.0, 1.0, curve_codec.num_path_points)
        near_weights = 0.25 + torch.exp(-4.0 * progress)
        self.register_buffer(
            "near_path_weights",
            near_weights / near_weights.mean(),
            persistent=True,
        )

    @property
    def planning_horizon_m(self) -> float:
        return self.curve_codec.planning_horizon_m

    def encode_condition(self, condition: PolicyCondition) -> ConditionFeatures:
        condition.validate()
        valid_depth = condition.observation_valid[:, :, None, None, None]
        depth = torch.where(
            valid_depth, condition.depth, torch.zeros_like(condition.depth)
        )
        observation = self.depth_encoder(
            depth,
            condition.observation_to_current.float(),
            condition.observation_valid,
        )
        return self.condition_encoder(
            observation,
            condition.point_goal,
            condition.observation_to_current.float(),
            condition.observation_valid,
        )

    def _decode(
        self,
        curve_coordinates: Tensor,
        point_goal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.curve_codec.decode(
            curve_coordinates.float(),
            point_goal.float(),
        )

    def _path_loss(self, predicted_path: Tensor, reference_path: Tensor) -> Tensor:
        point_loss = (
            torch.linalg.vector_norm(
                predicted_path - reference_path,
                dim=-1,
            )
            / self.planning_horizon_m
        )
        weights = self.near_path_weights.to(
            device=point_loss.device,
            dtype=point_loss.dtype,
        )
        return (point_loss * weights).mean()

    def _tangent_loss(self, predicted_path: Tensor, reference_path: Tensor) -> Tensor:
        predicted_delta = predicted_path[:, 1:] - predicted_path[:, :-1]
        reference_delta = reference_path[:, 1:] - reference_path[:, :-1]
        reference_length = torch.linalg.vector_norm(reference_delta, dim=-1)
        predicted_direction = F.normalize(predicted_delta, dim=-1, eps=1e-6)
        reference_direction = F.normalize(reference_delta, dim=-1, eps=1e-6)
        direction_error = 1.0 - (predicted_direction * reference_direction).sum(dim=-1)
        valid = reference_length > 1e-5
        weights = self.near_path_weights[1:].to(
            device=direction_error.device,
            dtype=direction_error.dtype,
        )
        weighted = direction_error * weights * valid
        return weighted.sum() / (weights * valid).sum().clamp_min(1.0)

    @staticmethod
    def _coordinate_loss(
        predicted: Tensor,
        target: Tensor,
        free_mask: Tensor,
    ) -> Tensor:
        error = (predicted - target).square() * free_mask
        return error.sum() / free_mask.sum().clamp_min(1.0)

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
    ) -> CurveNavLoss:
        if target.reference_path.shape[1:] != (
            self.curve_codec.num_path_points,
            2,
        ):
            raise ValueError("target reference path does not match the policy curve")
        if target.control_points.shape[1:] != (
            self.target_codec.num_control_points,
            2,
        ):
            raise ValueError("target controls do not match the expert smoother")

        encoded = self.encode_condition(condition)
        smoothed_target_path = self.target_codec.decode_equal_arc(
            target.control_points.float()
        )
        free_mask = self.curve_codec.free_mask.to(
            device=condition.point_goal.device,
            dtype=torch.bool,
        )[None].expand(condition.point_goal.shape[0], -1, -1)
        proposal_state = self.curve_proposal(
            encoded,
            free_mask.to(encoded.tokens.dtype),
        )
        clean_future = self.trajectory_flow.normalize_curve_coordinates(
            self.curve_codec.encode_target(
                smoothed_target_path,
                condition.point_goal.float(),
            )
        )
        flow_state, time, target_velocity = self.trajectory_flow.training_path(
            clean_future,
            proposal_state.detach(),
            free_mask.to(clean_future.dtype),
            self.integration_steps,
        )
        flow_prediction = self.trajectory_flow(
            flow_state,
            time,
            encoded.tokens,
            encoded.route_token,
        )
        predicted_velocity = flow_prediction.velocity * free_mask
        predicted_endpoint = flow_prediction.endpoint * free_mask
        flow_velocity_loss = self._coordinate_loss(
            predicted_velocity,
            target_velocity,
            free_mask,
        )
        flow_endpoint_loss = self._coordinate_loss(
            predicted_endpoint,
            clean_future,
            free_mask,
        )

        reconstructed = self.trajectory_flow.reconstruct_clean(
            flow_state,
            time,
            predicted_velocity,
        )
        flow_consistency_loss = self._coordinate_loss(
            predicted_endpoint,
            reconstructed,
            free_mask,
        )
        flow_loss = (
            flow_velocity_loss
            + flow_endpoint_loss
            + FLOW_SELF_CONSISTENCY_WEIGHT * flow_consistency_loss
        )
        decoded_path, _, _ = self.curve_codec.decode(
            self.trajectory_flow.denormalize_curve_coordinates(
                torch.cat((reconstructed, proposal_state), dim=0)
            ),
            torch.cat((condition.point_goal, condition.point_goal), dim=0).float(),
        )
        predicted_path, proposal_path = decoded_path.chunk(2, dim=0)
        reference_path = target.reference_path.float()
        path_loss = self._path_loss(predicted_path, reference_path)
        tangent_loss = self._tangent_loss(predicted_path, reference_path)
        proposal_coordinate_loss = self._coordinate_loss(
            proposal_state,
            clean_future,
            free_mask,
        )
        proposal_path_loss = self._path_loss(proposal_path, smoothed_target_path)
        proposal_tangent_loss = self._tangent_loss(
            proposal_path,
            smoothed_target_path,
        )
        proposal_loss = (
            proposal_coordinate_loss + proposal_path_loss + proposal_tangent_loss
        )

        loss = flow_loss + path_loss + tangent_loss + proposal_loss
        return CurveNavLoss(
            loss=loss,
            flow_loss=flow_loss,
            flow_velocity_loss=flow_velocity_loss,
            flow_endpoint_loss=flow_endpoint_loss,
            flow_consistency_loss=flow_consistency_loss,
            path_loss=path_loss,
            tangent_loss=tangent_loss,
            proposal_loss=proposal_loss,
            proposal_coordinate_loss=proposal_coordinate_loss,
            proposal_path_loss=proposal_path_loss,
            proposal_tangent_loss=proposal_tangent_loss,
        )

    @torch.no_grad()
    def sample(
        self,
        condition: PolicyCondition,
    ) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        free_mask = self.curve_codec.free_mask.to(
            device=condition.point_goal.device,
            dtype=torch.bool,
        )[None].expand(condition.point_goal.shape[0], -1, -1)
        proposal_state = self.curve_proposal(
            encoded,
            free_mask.to(encoded.tokens.dtype),
        )
        curve_coordinates = self.trajectory_flow.integrate(
            encoded,
            proposal_state,
            self.integration_steps,
            free_mask,
        )
        path, heading, curvature = self._decode(
            self.trajectory_flow.denormalize_curve_coordinates(curve_coordinates),
            condition.point_goal,
        )
        return TrajectoryPrediction(
            path=path,
            heading=heading,
            curvature=curvature,
        )

    def forward(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
    ) -> CurveNavLoss:
        return self.training_loss(condition, target)
