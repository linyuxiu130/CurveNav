"""Composition root for the one CurveNav generate-select graph."""

from curvenav.config import CurveNavConfig
from curvenav.conditioning import PolicyConditionEncoder
from curvenav.encoders import DepthObservationEncoder, PointGoalEncoder
from curvenav.models import (
    CurveNavPolicy,
    GeometricTrajectoryEvaluator,
    SplineControlFlow,
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
    evaluator = config.trajectory_evaluator

    depth_encoder = DepthObservationEncoder(
        model_dim=depth.model_dim,
        frame_tokens_height=depth.frame_tokens_height,
        frame_tokens_width=depth.frame_tokens_width,
        dropout=depth.dropout,
        max_depth_m=config.data.max_depth_m,
        focal_x_px=config.data.canonical_focal_x_px,
        focal_y_px=config.data.canonical_focal_y_px,
        camera_forward_offset_m=config.data.camera_forward_offset_m,
        camera_downward_pitch_degrees=(
            config.data.camera_downward_pitch_degrees
        ),
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
    trajectory_evaluator = GeometricTrajectoryEvaluator(
        image_height=config.data.image_height,
        image_width=config.data.image_width,
        focal_x_px=config.data.canonical_focal_x_px,
        focal_y_px=config.data.canonical_focal_y_px,
        max_depth_m=config.data.max_depth_m,
        camera_forward_offset_m=config.data.camera_forward_offset_m,
        camera_height_m=config.data.camera_height_m,
        camera_downward_pitch_degrees=(
            config.data.camera_downward_pitch_degrees
        ),
        minimum_obstacle_height_m=evaluator.minimum_obstacle_height_m,
        robot_height_m=evaluator.robot_height_m,
        robot_radius_m=evaluator.robot_radius_m,
        safety_margin_m=evaluator.safety_margin_m,
        discount_factor=evaluator.discount_factor,
        clearance_weight=evaluator.clearance_weight,
        length_weight=evaluator.length_weight,
        goal_weight=evaluator.goal_weight,
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
        trajectory_evaluator=trajectory_evaluator,
        codec=codec,
        normalizer=PlanarScaleNormalizer(
            (trajectory.normalization_scale_m, trajectory.normalization_scale_m)
        ),
        integration_steps=trajectory_flow_config.integration_steps,
    )
