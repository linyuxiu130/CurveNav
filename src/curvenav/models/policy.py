"""Top-level CurveNav policy orchestration."""

import torch
from torch import Tensor, nn

from curvenav.generative import RectifiedFlow, RectifiedFlowLoss
from curvenav.trajectory import PlanarBSplineCodec, PlanarScaleNormalizer
from curvenav.types import EncodedCondition, PolicyCondition, TrajectoryPrediction, TrajectoryTarget


class CurveNavPolicy(nn.Module):
    def __init__(
        self,
        depth_encoder: nn.Module,
        goal_encoder: nn.Module,
        motion_encoder: nn.Module,
        condition_encoder: nn.Module,
        rectified_flow: RectifiedFlow,
        codec: PlanarBSplineCodec,
        normalizer: PlanarScaleNormalizer,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.goal_encoder = goal_encoder
        self.motion_encoder = motion_encoder
        self.condition_encoder = condition_encoder
        self.rectified_flow = rectified_flow
        self.codec = codec
        self.normalizer = normalizer

    def encode_condition(self, condition: PolicyCondition) -> EncodedCondition:
        condition.validate()
        observation_tokens = self.depth_encoder(condition.depth)
        goal_token = self.goal_encoder(condition.task_goal)
        motion_token = self.motion_encoder(condition.motion_context)
        return self.condition_encoder(observation_tokens, goal_token, motion_token)

    def source_mean(self, task_goal: Tensor) -> Tensor:
        """Build a goal-directed prior without constraining the learned endpoint."""
        distance = torch.linalg.vector_norm(task_goal, dim=-1, keepdim=True)
        direction = task_goal / distance.clamp_min(1e-6)
        direction = torch.where(distance > 1e-6, direction, torch.zeros_like(direction))
        maximum = self.normalizer.scale_xy[0].to(
            device=task_goal.device,
            dtype=task_goal.dtype,
        )
        reference_endpoint = direction * distance.clamp_max(maximum)
        return self.codec.straight_line_controls(
            self.normalizer.normalize(reference_endpoint)
        )

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
    ) -> RectifiedFlowLoss:
        target.validate()
        encoded = self.encode_condition(condition)
        normalized = self.normalizer.normalize(target.control_points)
        source_mean = self.source_mean(condition.task_goal)
        return self.rectified_flow.training_loss(normalized, source_mean, encoded)

    @torch.no_grad()
    def sample(
        self,
        condition: PolicyCondition,
        num_samples: int = 8,
    ) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        source_mean = self.source_mean(condition.task_goal)
        normalized = self.rectified_flow.sample(
            encoded,
            source_mean=source_mean,
            num_samples=num_samples,
        )
        control_points = self.normalizer.denormalize(normalized.float())
        control_points = self.rectified_flow.enforce_origin(control_points)
        with torch.autocast(device_type=control_points.device.type, enabled=False):
            dense_path = self.codec.decode(control_points)
            heading, curvature = self.codec.geometry(dense_path)
        return TrajectoryPrediction(control_points, dense_path, heading, curvature)

    def forward(self, condition: PolicyCondition, target: TrajectoryTarget) -> Tensor:
        return self.training_loss(condition, target).loss
