"""Past-aware conditional-flow generation of one executable local curve."""

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


CANDIDATE_SAMPLES = 8


@dataclass
class CurveNavLoss:
    loss: Tensor
    flow_loss: Tensor
    path_loss: Tensor
    tangent_loss: Tensor
    subgoal_loss: Tensor


def _deterministic_antithetic_bases(tokens: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(0xC0A7E)
    half = torch.randn(CANDIDATE_SAMPLES // 2, tokens, 2, generator=generator)
    samples = torch.cat((half, -half), dim=0)
    rms = samples.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    return samples / rms


class CurveNavPolicy(nn.Module):
    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_flow: nn.Module,
        curve_codec: BoundedCurvatureTrajectory,
        target_codec: PlanarBSplineCodec,
        integration_steps: int,
        observation_frames: int,
        history_scale_m: float,
    ) -> None:
        super().__init__()
        if observation_frames < 2 or history_scale_m <= 0:
            raise ValueError(
                "past-aware policy requires a positive observation history"
            )
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_flow = trajectory_flow
        self.curve_codec = curve_codec
        self.target_codec = target_codec
        self.integration_steps = integration_steps
        self.history_tokens = observation_frames - 1
        self.history_scale_m = float(history_scale_m)
        if self.trajectory_flow.history_tokens != self.history_tokens:
            raise ValueError("condition and flow history token counts differ")
        if self.trajectory_flow.future_tokens != self.curve_codec.num_curve_tokens:
            raise ValueError("flow and curve token counts differ")

        progress = torch.linspace(0.0, 1.0, curve_codec.num_path_points)
        near_weights = 0.25 + torch.exp(-4.0 * progress)
        self.register_buffer(
            "near_path_weights",
            near_weights / near_weights.mean(),
            persistent=True,
        )
        self.register_buffer(
            "candidate_bases",
            _deterministic_antithetic_bases(self.trajectory_flow.total_tokens),
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

    def _history_state(self, condition: PolicyCondition) -> tuple[Tensor, Tensor]:
        coordinates = (
            condition.observation_to_current[:, : self.history_tokens, :2].float()
            / self.history_scale_m
        )
        valid = condition.observation_valid[:, : self.history_tokens, None].expand(
            -1, -1, 2
        )
        return coordinates * valid, valid

    def _joint_state(
        self,
        condition: PolicyCondition,
        future: Tensor,
    ) -> tuple[Tensor, Tensor]:
        history, history_mask = self._history_state(condition)
        future_mask = self.curve_codec.free_mask.to(
            device=future.device, dtype=torch.bool
        )[None].expand(future.shape[0], -1, -1)
        return (
            torch.cat((history.to(future.dtype), future), dim=1),
            torch.cat((history_mask, future_mask), dim=1),
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
        point_loss = F.smooth_l1_loss(
            predicted_path / self.planning_horizon_m,
            reference_path / self.planning_horizon_m,
            reduction="none",
        ).mean(dim=-1)
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

    def _subgoal_loss(
        self,
        encoded: ConditionFeatures,
        reference_path: Tensor,
    ) -> Tensor:
        error = F.smooth_l1_loss(
            encoded.local_subgoal / self.planning_horizon_m,
            reference_path[:, -1] / self.planning_horizon_m,
            reduction="none",
        ).mean(dim=-1)
        return error.mean()

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
        clean_future = self.curve_codec.encode_target(
            smoothed_target_path,
            condition.point_goal.float(),
        )
        clean, free_mask = self._joint_state(condition, clean_future)
        noisy, time, target_velocity = self.trajectory_flow.training_pair(
            clean, free_mask.to(clean.dtype)
        )
        predicted_velocity = (
            self.trajectory_flow(
                noisy,
                time,
                encoded.tokens,
                encoded.route_token,
            )
            * free_mask
        )
        flow_error = (predicted_velocity - target_velocity).square()
        flow_loss = (flow_error * free_mask).sum() / free_mask.sum().clamp_min(1)

        reconstructed = self.trajectory_flow.reconstruct_clean(
            noisy,
            time,
            predicted_velocity,
        )
        reconstructed_future = reconstructed[:, self.history_tokens :]
        predicted_path, _, _ = self.curve_codec.decode(
            reconstructed_future,
            condition.point_goal.float(),
        )
        reference_path = target.reference_path.float()
        path_loss = self._path_loss(predicted_path, reference_path)
        tangent_loss = self._tangent_loss(predicted_path, reference_path)
        subgoal_loss = self._subgoal_loss(encoded, reference_path)

        loss = flow_loss + path_loss + tangent_loss + subgoal_loss
        return CurveNavLoss(
            loss=loss,
            flow_loss=flow_loss,
            path_loss=path_loss,
            tangent_loss=tangent_loss,
            subgoal_loss=subgoal_loss,
        )

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        history, history_mask = self._history_state(condition)
        future_mask = self.curve_codec.free_mask.to(
            device=history.device, dtype=torch.bool
        )[None].expand(history.shape[0], -1, -1)
        free_mask = torch.cat((history_mask, future_mask), dim=1)
        candidates = self.trajectory_flow.integrate_candidates(
            encoded,
            self.integration_steps,
            self.candidate_bases,
            free_mask,
        )

        reconstructed_past = candidates[:, :, : self.history_tokens]
        error = (reconstructed_past - history[:, None]).square()
        mask = history_mask[:, None].to(error.dtype)
        consistency = (error * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(
            1.0
        )
        selected_index = consistency.argmin(dim=1)
        batch_index = torch.arange(candidates.shape[0], device=candidates.device)
        curve_coordinates = candidates[
            batch_index,
            selected_index,
            self.history_tokens :,
        ]
        path, heading, curvature = self._decode(
            curve_coordinates,
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
        target: TrajectoryTarget | None = None,
    ) -> CurveNavLoss | TrajectoryPrediction:
        if target is None:
            return self.sample(condition)
        return self.training_loss(condition, target)
