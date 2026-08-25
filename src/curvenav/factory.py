"""Composition root for the one CurveNav generate-rank-select graph."""

from curvenav.config import CurveNavConfig
from curvenav.conditioning import PolicyConditionEncoder
from curvenav.encoders import DepthObservationEncoder, PointGoalEncoder
from curvenav.models import (
    CurveNavPolicy,
    SplineControlFlow,
    TrajectoryScorer,
)
from curvenav.trajectory import PlanarBSplineCodec, PlanarScaleNormalizer


def build_policy(config: CurveNavConfig) -> CurveNavPolicy:
    """Assemble the only production policy graph from validated components."""
    config.validate()
    trajectory = config.trajectory
    depth = config.depth_encoder
    point_goal = config.point_goal_encoder
    condition = config.condition_encoder
    trajectory_flow_config = config.trajectory_flow
    scorer = config.trajectory_scorer

    depth_encoder = DepthObservationEncoder(
        model_dim=depth.model_dim,
        frame_tokens_height=depth.frame_tokens_height,
        frame_tokens_width=depth.frame_tokens_width,
        dropout=depth.dropout,
        max_depth_m=config.data.max_depth_m,
        focal_x_px=config.data.canonical_focal_x_px,
    )
    point_goal_encoder = PointGoalEncoder(
        model_dim=point_goal.model_dim,
        hidden_dim=point_goal.hidden_dim,
        goal_clip_distance_m=point_goal.goal_clip_distance_m,
    )
    condition_encoder = PolicyConditionEncoder(
        point_goal_encoder,
        observation_frames=config.data.observation_frames,
        spatial_tokens=depth.frame_tokens_height * depth.frame_tokens_width,
        model_dim=condition.model_dim,
        transformer_layers=condition.transformer_layers,
        transformer_heads=condition.transformer_heads,
        dropout=condition.dropout,
    )
    trajectory_flow = SplineControlFlow(
        num_control_points=trajectory.num_control_points,
        model_dim=trajectory_flow_config.model_dim,
        layers=trajectory_flow_config.transformer_layers,
        heads=trajectory_flow_config.transformer_heads,
        dropout=trajectory_flow_config.dropout,
        inference_candidates=trajectory_flow_config.inference_candidates,
        inference_seed=trajectory_flow_config.inference_seed,
    )
    trajectory_scorer = TrajectoryScorer(
        num_control_points=trajectory.num_control_points,
        model_dim=scorer.model_dim,
        layers=scorer.transformer_layers,
        heads=scorer.transformer_heads,
        dropout=scorer.dropout,
    )
    codec = PlanarBSplineCodec(
        num_control_points=trajectory.num_control_points,
        degree=trajectory.degree,
        num_path_points=trajectory.num_path_points,
    )
    return CurveNavPolicy(
        depth_encoder=depth_encoder,
        condition_encoder=condition_encoder,
        trajectory_flow=trajectory_flow,
        trajectory_scorer=trajectory_scorer,
        codec=codec,
        normalizer=PlanarScaleNormalizer(
            (trajectory.normalization_scale_m, trajectory.normalization_scale_m)
        ),
        integration_steps=trajectory_flow_config.integration_steps,
    )
