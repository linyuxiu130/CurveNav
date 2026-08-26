"""Composition root for the one CurveNav conditional-flow graph."""

from curvenav.config import CurveNavConfig
from curvenav.conditioning import PolicyConditionEncoder
from curvenav.encoders import DepthObservationEncoder, PointGoalEncoder
from curvenav.models import (
    CurveNavPolicy,
    CurvatureTrajectoryFlow,
)
from curvenav.trajectory import (
    BoundedCurvatureTrajectory,
    PlanarBSplineCodec,
)


def build_policy(config: CurveNavConfig) -> CurveNavPolicy:
    """Assemble the only production policy graph from validated components."""
    config.validate()
    trajectory = config.trajectory
    depth = config.depth_encoder
    point_goal = config.point_goal_encoder
    condition = config.condition_encoder
    trajectory_flow_config = config.trajectory_flow

    depth_encoder = DepthObservationEncoder(
        model_dim=depth.model_dim,
        frame_tokens_height=depth.frame_tokens_height,
        frame_tokens_width=depth.frame_tokens_width,
        dropout=depth.dropout,
        max_depth_m=config.data.max_depth_m,
        focal_x_px=config.data.canonical_focal_x_px,
        focal_y_px=config.data.canonical_focal_y_px,
        camera_forward_offset_m=config.data.camera_forward_offset_m,
        camera_downward_pitch_degrees=(config.data.camera_downward_pitch_degrees),
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
        history_scale_m=(
            (config.data.observation_frames - 1) * config.data.frame_spacing_m
        ),
        planning_horizon_m=(
            config.data.future_steps * config.data.expert_waypoint_spacing_m
        ),
        model_dim=condition.model_dim,
        transformer_layers=condition.transformer_layers,
        transformer_heads=condition.transformer_heads,
        dropout=condition.dropout,
    )
    curve_codec = BoundedCurvatureTrajectory(
        num_curvature_control_points=trajectory.num_curvature_control_points,
        degree=trajectory.curvature_spline_degree,
        num_path_points=trajectory.num_path_points,
        planning_horizon_m=(
            config.data.future_steps * config.data.expert_waypoint_spacing_m
        ),
        maximum_curvature_inv_m=trajectory.maximum_curvature_inv_m,
    )
    target_codec = PlanarBSplineCodec(
        num_control_points=trajectory.num_target_control_points,
        degree=trajectory.target_spline_degree,
        num_path_points=trajectory.num_path_points,
    )
    trajectory_flow = CurvatureTrajectoryFlow(
        future_tokens=curve_codec.num_curve_tokens,
        model_dim=trajectory_flow_config.model_dim,
        layers=trajectory_flow_config.transformer_layers,
        heads=trajectory_flow_config.transformer_heads,
        dropout=trajectory_flow_config.dropout,
    )
    return CurveNavPolicy(
        depth_encoder=depth_encoder,
        condition_encoder=condition_encoder,
        trajectory_flow=trajectory_flow,
        curve_codec=curve_codec,
        target_codec=target_codec,
        integration_steps=trajectory_flow_config.integration_steps,
    )
