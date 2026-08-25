"""One conditional-flow generate, group-rank, spline-select policy."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from curvenav.trajectory import PlanarBSplineCodec, PlanarScaleNormalizer
from curvenav.types import (
    ConditionFeatures,
    PolicyCondition,
    TrajectoryPrediction,
    TrajectoryTarget,
)


FLOW_LOSS_WEIGHT = 1.0
PATH_LOSS_WEIGHT = 0.5
TANGENT_LOSS_WEIGHT = 0.1
RANKING_LOSS_WEIGHT = 0.1


@dataclass
class CurveNavLoss:
    loss: Tensor
    flow_loss: Tensor
    path_loss: Tensor
    tangent_loss: Tensor
    ranking_loss: Tensor


def group_ranking_loss(logits: Tensor, target_distribution: Tensor) -> Tensor:
    """Cross-entropy against a quality distribution over one candidate group."""
    if logits.ndim != 2 or target_distribution.shape != logits.shape:
        raise ValueError("logits and target_distribution must both have shape [B, C]")
    if not torch.isfinite(target_distribution).all() or not torch.allclose(
        target_distribution.sum(dim=1), torch.ones(logits.shape[0], device=logits.device)
    ):
        raise ValueError("target_distribution must be finite and row-normalized")
    return -(target_distribution * torch.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


class CurveNavPolicy(nn.Module):
    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_flow: nn.Module,
        trajectory_scorer: nn.Module,
        codec: PlanarBSplineCodec,
        normalizer: PlanarScaleNormalizer,
        integration_steps: int,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_flow = trajectory_flow
        self.trajectory_scorer = trajectory_scorer
        self.codec = codec
        self.normalizer = normalizer
        self.integration_steps = integration_steps

        progress = torch.linspace(0.0, 1.0, codec.num_path_points)
        near_weights = 0.25 + torch.exp(-4.0 * progress)
        self.register_buffer(
            "near_path_weights",
            near_weights / near_weights.mean(),
            persistent=True,
        )

    def encode_condition(self, condition: PolicyCondition) -> ConditionFeatures:
        condition.validate()
        valid_depth = condition.observation_valid[:, :, None, None, None]
        depth = torch.where(valid_depth, condition.depth, torch.zeros_like(condition.depth))
        observation = self.depth_encoder(
            depth,
            condition.observation_to_current.float(),
            condition.observation_valid,
        )
        return self.condition_encoder(
            observation,
            condition.point_goal,
            condition.observation_valid,
        )

    def _decode_candidates(self, normalized_controls: Tensor) -> tuple[Tensor, Tensor]:
        batch, candidates = normalized_controls.shape[:2]
        controls = self.normalizer.denormalize(normalized_controls.float())
        controls = controls.clone()
        controls[:, :, 0] = 0
        paths = self.codec.decode_equal_arc(controls.flatten(0, 1)).reshape(
            batch,
            candidates,
            self.codec.num_path_points,
            2,
        )
        return controls, paths

    def _path_loss(self, predicted_path: Tensor, reference_path: Tensor) -> Tensor:
        point_loss = F.smooth_l1_loss(
            predicted_path,
            reference_path,
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
        direction_error = 1.0 - (
            predicted_direction * reference_direction
        ).sum(dim=-1)
        valid = reference_length > 1e-5
        weights = self.near_path_weights[1:].to(
            device=direction_error.device,
            dtype=direction_error.dtype,
        )
        weighted = direction_error * weights * valid
        return weighted.sum() / (weights * valid).sum().clamp_min(1.0)

    def _group_candidates(self, clean: Tensor) -> Tensor:
        """Pair each context with its expert and empirical marginal negatives."""
        batch = clean.shape[0]
        candidates = self.trajectory_flow.inference_candidates
        if batch < candidates:
            raise ValueError(
                "group scorer training requires batch size "
                f">= {candidates}"
            )
        offsets = torch.arange(1, candidates, device=clean.device)
        batch_indices = torch.arange(batch, device=clean.device)[:, None]
        negative_indices = (batch_indices + offsets[None]) % batch
        negatives = clean[negative_indices]
        return torch.cat((clean[:, None], negatives), dim=1).detach()

    def training_loss(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
    ) -> CurveNavLoss:
        target.validate()
        if target.control_points.shape[1:] != (self.codec.num_control_points, 2):
            raise ValueError("target controls do not match the policy spline")
        if target.reference_path.shape[1:] != (self.codec.num_path_points, 2):
            raise ValueError("target reference path does not match the policy spline")

        encoded = self.encode_condition(condition)
        clean = self.normalizer.normalize(target.control_points.float()).clone()
        clean[:, 0] = 0
        noisy, time, target_velocity = self.trajectory_flow.training_pair(clean)
        predicted_velocity = self.trajectory_flow(noisy, time, encoded.tokens)
        flow_loss = F.mse_loss(
            predicted_velocity[:, 1:],
            target_velocity[:, 1:],
        )

        reconstructed = self.trajectory_flow.reconstruct_clean(
            noisy,
            time,
            predicted_velocity,
        )
        reconstructed = reconstructed.clone()
        reconstructed[:, 0] = 0
        reconstructed_metric = self.normalizer.denormalize(reconstructed.float())
        predicted_path = self.codec.decode_equal_arc(reconstructed_metric)
        reference_path = target.reference_path.float()
        path_loss = self._path_loss(predicted_path, reference_path)
        tangent_loss = self._tangent_loss(predicted_path, reference_path)

        scorer_candidates = self._group_candidates(clean)
        _, scorer_paths = self._decode_candidates(scorer_candidates)
        scorer_paths = scorer_paths.detach()
        reference_path = target.reference_path.float()[:, None]
        candidate_error = F.smooth_l1_loss(
            scorer_paths,
            reference_path.expand_as(scorer_paths),
            reduction="none",
        ).mean(dim=(-1, -2))
        target_distribution = torch.softmax(-candidate_error.detach(), dim=1)
        ranking_logits = self.trajectory_scorer(scorer_candidates, encoded)
        ranking_loss = group_ranking_loss(ranking_logits, target_distribution)
        loss = (
            FLOW_LOSS_WEIGHT * flow_loss
            + PATH_LOSS_WEIGHT * path_loss
            + TANGENT_LOSS_WEIGHT * tangent_loss
            + RANKING_LOSS_WEIGHT * ranking_loss
        )
        return CurveNavLoss(
            loss,
            flow_loss,
            path_loss,
            tangent_loss,
            ranking_loss,
        )

    @torch.no_grad()
    def sample(self, condition: PolicyCondition) -> TrajectoryPrediction:
        encoded = self.encode_condition(condition)
        normalized_candidates = self.trajectory_flow.sample(
            encoded,
            self.integration_steps,
        )
        candidate_log_probabilities = self.trajectory_scorer.log_probabilities(
            normalized_candidates,
            encoded,
        )
        candidate_controls, candidate_paths = self._decode_candidates(
            normalized_candidates
        )
        selected_index = candidate_log_probabilities.argmax(dim=1)
        batch_index = torch.arange(
            selected_index.shape[0],
            device=selected_index.device,
        )
        control_points = candidate_controls[batch_index, selected_index]
        path = candidate_paths[batch_index, selected_index]
        heading, curvature = self.codec.geometry(path)
        return TrajectoryPrediction(
            control_points=control_points,
            path=path,
            heading=heading,
            curvature=curvature,
            candidate_control_points=candidate_controls,
            candidate_paths=candidate_paths,
            candidate_log_probabilities=candidate_log_probabilities,
        )

    def forward(
        self,
        condition: PolicyCondition,
        target: TrajectoryTarget,
    ) -> CurveNavLoss:
        return self.training_loss(condition, target)
