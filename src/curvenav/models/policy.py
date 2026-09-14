"""PointGoal-conditioned Flow Matching in physical curve coordinates."""

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

from curvenav.trajectory import IncrementalBSplineTrajectory
from curvenav.models.blocks import ProjectedCondition
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)


TRAINING_LOSS_NAMES = ("loss", "flow_loss", "critic_loss", "ranking_loss")
INFERENCE_SOURCE_SEED = 20_260_828
INFERENCE_CANDIDATES = 32
FLOW_TIME_SAMPLING = "single_draw_logit_normal"
FLOW_LOGIT_NORMAL_MEAN = -0.4
FLOW_LOGIT_NORMAL_STD = 1.0


@dataclass
class CurveNavTrainingOutput:
    flow_loss: Tensor
    candidate_paths: Tensor
    candidate_scores: Tensor


def repeat_condition(condition: ConditionFeatures, count: int) -> ConditionFeatures:
    return ConditionFeatures(**{
        f.name: getattr(condition, f.name).repeat_interleave(count, dim=0)
        for f in fields(condition)
    })


def repeat_memory(
    memory: tuple[ProjectedCondition, ...], count: int
) -> tuple[ProjectedCondition, ...]:
    return tuple(
        tuple(value.repeat_interleave(count, dim=0) for value in layer)
        for layer in memory
    )


class CurveNavPolicy(nn.Module):
    """Generate 32 goal-conditioned curves and select the learned route-utility argmax."""

    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_decoder: nn.Module,
        trajectory_evaluator: nn.Module,
        curve_codec: IncrementalBSplineTrajectory,
        planning_horizon_m: float,
        integration_steps: int,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_decoder = trajectory_decoder
        self.trajectory_evaluator = trajectory_evaluator
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
            INFERENCE_CANDIDATES,
            curve_codec.coordinate_dim,
            generator=generator,
        )
        self.register_buffer(
            "inference_source",
            inference_source,
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

    def _predict_velocity(
        self, state: Tensor, time: Tensor, condition: ConditionFeatures, memory
    ) -> Tensor:
        path, _ = self.curve_codec.decode(state)
        return self.trajectory_decoder(state, time, path, condition, memory).float()

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
    ) -> CurveNavTrainingOutput:
        target.validate()
        clean = self.curve_codec.coordinates_from_values(target.curve_values.float())
        if source.shape != clean.shape:
            raise ValueError(
                "flow source must match the standardized curve coordinates"
            )
        source = source.float()
        encoded = self.encode_condition(condition)
        memory = self.trajectory_decoder.project_condition_memory(encoded)
        time = torch.sigmoid(
            torch.randn(len(clean), device=clean.device) * FLOW_LOGIT_NORMAL_STD
            + FLOW_LOGIT_NORMAL_MEAN
        )
        state = (1 - time[:, None]) * clean + time[:, None] * source
        velocity = self._predict_velocity(state, time, encoded, memory)
        flow_loss = (velocity - (source - clean)).square().mean()
        # Match deployment's actual two-step proposal distribution. Expert labels
        # remain available when the early generator produces only poor proposals.
        proposals = self._generate_coordinates(
            repeat_condition(encoded, 3), repeat_memory(memory, 3),
            torch.randn(len(clean) * 3, clean.shape[-1], device=clean.device),
        ).unflatten(0, (len(clean), 3))
        candidates = torch.cat((clean[:, None], proposals), dim=1).detach()
        paths, _ = self.curve_codec.decode(candidates.flatten(0, 1))
        critic_memory = self.trajectory_evaluator.project_condition_memory(encoded)
        scores = self.trajectory_evaluator(
            paths, condition.point_goal.repeat_interleave(4, dim=0),
            repeat_condition(encoded, 4), repeat_memory(critic_memory, 4),
        )
        return CurveNavTrainingOutput(
            flow_loss, paths.unflatten(0, (len(clean), 4)), scores.unflatten(0, (len(clean), 4))
        )

    @torch.no_grad()
    def _generate_coordinates(self, encoded, memory, source: Tensor) -> Tensor:
        state = source
        for index in range(self.integration_steps):
            time = torch.full(
                (len(state),), 1 - index / self.integration_steps, device=state.device
            )
            state = state - self._predict_velocity(state, time, encoded, memory) / self.integration_steps
        return state

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        memory = self.trajectory_decoder.project_condition_memory(encoded)
        critic_memory = self.trajectory_evaluator.project_condition_memory(encoded)
        batch = len(condition.point_goal)
        repeated = repeat_condition(encoded, INFERENCE_CANDIDATES)
        state = self._generate_coordinates(
            repeated, repeat_memory(memory, INFERENCE_CANDIDATES),
            self.inference_source.repeat(batch, 1),
        )
        path, _ = self.curve_codec.decode(state)
        scores = self.trajectory_evaluator(
            path, condition.point_goal.repeat_interleave(INFERENCE_CANDIDATES, dim=0),
            repeated, repeat_memory(critic_memory, INFERENCE_CANDIDATES),
        ).unflatten(0, (batch, INFERENCE_CANDIDATES))
        candidates = path.unflatten(0, (batch, INFERENCE_CANDIDATES))
        selected = scores.argmax(dim=1)
        path = candidates[torch.arange(batch, device=state.device), selected]
        return TrajectoryPrediction(path, candidates, scores, selected)

    def forward(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
    ) -> CurveNavTrainingOutput:
        return self.training_loss(condition, target, source)
