"""History-conditioned generation of one executable local curve."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from curvenav.trajectory import BoundedCurvatureTrajectory, PlanarBSplineCodec
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)


TRAINING_LOSS_NAMES = (
    "loss",
    "coordinate_loss",
    "path_loss",
    "tangent_loss",
)


@dataclass
class CurveNavLoss:
    loss: Tensor
    coordinate_loss: Tensor
    path_loss: Tensor
    tangent_loss: Tensor

    def logging_values(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name) for name in TRAINING_LOSS_NAMES)


class CurveNavPolicy(nn.Module):
    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_decoder: nn.Module,
        curve_codec: BoundedCurvatureTrajectory,
        target_codec: PlanarBSplineCodec,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_decoder = trajectory_decoder
        self.curve_codec = curve_codec
        self.target_codec = target_codec
        if self.trajectory_decoder.curve_tokens != self.curve_codec.num_curve_tokens:
            raise ValueError("trajectory decoder and curve token counts differ")

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

    def _free_mask(self, point_goal: Tensor) -> Tensor:
        return self.curve_codec.free_mask.to(
            device=point_goal.device,
            dtype=torch.bool,
        )[None].expand(point_goal.shape[0], -1, -1)

    def _predict_normalized_coordinates(
        self,
        condition: PolicyCondition,
    ) -> Tensor:
        encoded = self.encode_condition(condition)
        free_mask = self._free_mask(condition.point_goal)
        return self.trajectory_decoder(
            encoded,
            free_mask.to(encoded.tokens.dtype),
        )

    def _decode(
        self,
        normalized_coordinates: Tensor,
        point_goal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.curve_codec.decode(
            self.trajectory_decoder.denormalize_curve_coordinates(
                normalized_coordinates
            ).float(),
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

        smoothed_target_path = self.target_codec.decode_equal_arc(
            target.control_points.float()
        )
        free_mask = self._free_mask(condition.point_goal)
        predicted_coordinates = self._predict_normalized_coordinates(condition)
        target_coordinates = self.trajectory_decoder.normalize_curve_coordinates(
            self.curve_codec.encode_target(
                smoothed_target_path,
                condition.point_goal.float(),
            )
        )
        predicted_path, _, _ = self._decode(
            predicted_coordinates,
            condition.point_goal,
        )
        coordinate_loss = self._coordinate_loss(
            predicted_coordinates,
            target_coordinates,
            free_mask,
        )
        path_loss = self._path_loss(predicted_path, smoothed_target_path)
        tangent_loss = self._tangent_loss(predicted_path, smoothed_target_path)
        loss = coordinate_loss + path_loss + tangent_loss
        return CurveNavLoss(
            loss=loss,
            coordinate_loss=coordinate_loss,
            path_loss=path_loss,
            tangent_loss=tangent_loss,
        )

    @torch.no_grad()
    def sample(
        self,
        condition: PolicyCondition,
    ) -> TrajectoryPrediction:
        normalized_coordinates = self._predict_normalized_coordinates(condition)
        path, heading, curvature = self._decode(
            normalized_coordinates,
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
