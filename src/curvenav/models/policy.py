"""PointGoal-conditioned Flow Matching in physical curve coordinates."""

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn

from curvenav.trajectory import IncrementalBSplineTrajectory
from curvenav.models.exploration import (
    EXPLORATION_RANDOM_DIM,
    GOAL_DROPOUT_PROBABILITY,
    structured_proposals,
)
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)


TRAINING_LOSS_NAMES = ("loss", "flow_loss", "critic_loss")
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


class CurveNavPolicy(nn.Module):
    """Score 32 structured goal/no-goal proposals against the mission goal."""

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
            2,
            curve_codec.coordinate_dim,
            generator=generator,
        )
        self.register_buffer(
            "inference_source",
            inference_source,
            persistent=True,
        )
        self.register_buffer(
            "inference_exploration",
            torch.rand(INFERENCE_CANDIDATES, EXPLORATION_RANDOM_DIM, generator=generator),
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
        flow_condition = replace(
            encoded,
            goal_present=(torch.rand_like(encoded.goal_present) >= GOAL_DROPOUT_PROBABILITY).float(),
        )
        velocity = self._predict_velocity(state, time, flow_condition, memory)
        flow_loss = (velocity - (source - clean)).square().mean()
        # Same proposal law as deployment, with fresh sources and perturbations.
        # Perturbed curves supervise the critic, never become expert Flow targets.
        proposals = self._propose(
            encoded, memory,
            torch.randn(len(clean), 3, 2, clean.shape[-1], device=clean.device),
            torch.rand(len(clean), 3, EXPLORATION_RANDOM_DIM, device=clean.device),
        )
        candidates = torch.cat((target.curve_values[:, None], proposals), dim=1).detach()
        paths, _ = self.curve_codec.decode_values(candidates.flatten(0, 1))
        critic_memory = self.trajectory_evaluator.project_condition_memory(encoded)
        scores = self.trajectory_evaluator(
            paths, condition.point_goal.repeat_interleave(4, dim=0),
            repeat_condition(encoded, 4), critic_memory,
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
    def _propose(self, encoded, memory, source: Tensor, random: Tensor) -> Tensor:
        batch, count = source.shape[:2]
        paired = repeat_condition(encoded, count * 2)
        paired = replace(
            paired,
            goal_present=torch.tensor([1., 0.], device=source.device).repeat(batch * count)[:, None],
        )
        coordinates = self._generate_coordinates(
            paired, memory, source.flatten(0, 2),
        ).unflatten(0, (batch * count, 2))
        return structured_proposals(
            self.curve_codec, coordinates, random.flatten(0, 1)
        ).unflatten(0, (batch, count))

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        return self.sample_encoded(encoded, condition.point_goal)

    @torch.no_grad()
    def sample_candidate_paths(self, encoded: ConditionFeatures) -> Tensor:
        """Generate the exact fixed 32-candidate bank used by deployment."""
        memory = self.trajectory_decoder.project_condition_memory(encoded)
        batch = len(encoded.tokens)
        values = self._propose(
            encoded,
            memory,
            self.inference_source[None].expand(batch, -1, -1, -1),
            self.inference_exploration[None].expand(batch, -1, -1),
        )
        paths, _ = self.curve_codec.decode_values(values.flatten(0, 1))
        return paths.unflatten(0, (batch, INFERENCE_CANDIDATES))

    def score_candidate_paths(
        self,
        encoded: ConditionFeatures,
        point_goal: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """Score a fixed candidate bank while keeping evaluator gradients."""
        batch, count = candidates.shape[:2]
        if candidates.ndim != 4 or candidates.shape[-1] != 2:
            raise ValueError("candidates must have shape [B,K,P,2]")
        if point_goal.shape != (batch, 2):
            raise ValueError("point_goal must have shape [B,2]")
        critic_memory = self.trajectory_evaluator.project_condition_memory(encoded)
        paths = candidates.flatten(0, 1)
        return self.trajectory_evaluator(
            paths,
            point_goal.repeat_interleave(count, dim=0),
            repeat_condition(encoded, count),
            critic_memory,
        ).unflatten(0, (batch, count))

    @torch.no_grad()
    def sample_nogoal(self, condition: PolicyCondition) -> Tensor:
        """Return [B,32,P,2] scene-conditioned curves with the goal masked.

        The shared observation schema still carries point_goal, but it has no
        influence here. Return raw learned curves, without goal scoring or
        geometric perturbations that could invalidate learned avoidance.
        """
        encoded = self.encode_condition(condition)
        encoded = replace(encoded, goal_present=torch.zeros_like(encoded.goal_present))
        memory = self.trajectory_decoder.project_condition_memory(encoded)
        batch = len(encoded.tokens)
        coordinates = self._generate_coordinates(
            repeat_condition(encoded, INFERENCE_CANDIDATES),
            memory,
            self.inference_source[:, 1].repeat(batch, 1),
        )
        paths, _ = self.curve_codec.decode(coordinates)
        return paths.unflatten(0, (batch, INFERENCE_CANDIDATES))

    @torch.no_grad()
    def sample_encoded(self, encoded: ConditionFeatures, point_goal: Tensor) -> TrajectoryPrediction:
        """Decode cached scene features with their corresponding goal intent."""
        candidates = self.sample_candidate_paths(encoded)
        scores = self.score_candidate_paths(encoded, point_goal, candidates)
        selected = scores.argmax(dim=1)
        path = candidates[torch.arange(len(point_goal), device=point_goal.device), selected]
        return TrajectoryPrediction(path, candidates, scores, selected)

    def forward(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
        source: Tensor,
    ) -> CurveNavTrainingOutput:
        return self.training_loss(condition, target, source)
