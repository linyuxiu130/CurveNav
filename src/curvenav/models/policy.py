"""PointGoal-conditioned Flow Matching in physical curve coordinates."""

from dataclasses import fields, replace

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from curvenav.trajectory import IncrementalBSplineTrajectory
from curvenav.types import ConditionFeatures


INFERENCE_SOURCE_SEED = 20_260_828
INFERENCE_CANDIDATES = 16
INTEGRATION_STEPS = 2


def repeat_condition(condition: ConditionFeatures, count: int) -> ConditionFeatures:
    # Candidate queries share the scene raster; gather their pixels from it directly.
    return ConditionFeatures(**{
        f.name: (condition.configuration_field if f.name=='configuration_field' else
            getattr(condition, f.name).repeat_interleave(count, dim=0))
        for f in fields(condition)
    })


def generate_candidate_bank(policy, condition, *, checkpoint_steps=False):
    """Generate exactly sixteen task-conditioned curves through the same two Euler steps."""
    count = INFERENCE_CANDIDATES
    batch_size = len(condition.tokens)
    present = condition.goal_present.bool()
    condition = replace(condition, terminal_goal=torch.where(
        present, condition.terminal_goal, torch.zeros_like(condition.terminal_goal)))
    # Same source assignment as the 39/40 reference; select noise before generation.
    source = torch.where(present[:, :, None],
        policy.inference_source[None, :count, 0],
        policy.inference_source[None, count:2*count, 1])
    state = source.flatten(0, 1)
    queries = repeat_condition(condition, count)
    memory = policy.trajectory_decoder.project_condition_memory(condition)
    for step in range(INTEGRATION_STEPS):
        time = torch.full((len(state),), 1-step/INTEGRATION_STEPS, device=state.device)
        velocity = (checkpoint(policy._predict_velocity, state, time, queries, memory, use_reentrant=False)
                    if checkpoint_steps else policy._predict_velocity(state, time, queries, memory))
        state = state-velocity/INTEGRATION_STEPS
    values = policy.curve_codec.values_from_coordinates(state)
    paths, tangent = policy.curve_codec.decode_values(values)
    return dict(values=values.unflatten(0, (batch_size, count)),
        paths=paths.unflatten(0, (batch_size, count)), tangent=tangent.unflatten(0, (batch_size, count)))


class CurveNavPolicy(nn.Module):
    """Generate sixteen native curves and score their actual geometry."""

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
            2 * INFERENCE_CANDIDATES,
            2,
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


    def _predict_velocity(
        self, state: Tensor, time: Tensor, condition: ConditionFeatures, memory
    ) -> Tensor:
        path, _ = self.curve_codec.decode(state)
        return self.trajectory_decoder(state, time, path, condition, memory).float()


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
            torch.where(encoded.goal_present.bool(), point_goal,
                torch.zeros_like(point_goal)).repeat_interleave(count, dim=0),
            repeat_condition(encoded, count),
            critic_memory,
        ).unflatten(0, (batch, count))
