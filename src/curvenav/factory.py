"""Composition root for the one CurveNav policy graph."""

from curvenav.config import CurveNavConfig
from curvenav.conditioning import PolicyConditionEncoder
from curvenav.encoders import DepthObservationEncoder, PointGoalEncoder
from curvenav.models import ConditionalCurveFlowDecoder, CurveNavPolicy
from curvenav.trajectory import MetricCurvatureTrajectory


def build_policy(config: CurveNavConfig) -> CurveNavPolicy:
    """Assemble the only production policy graph from validated components."""
    config.validate()
    trajectory = config.trajectory
    depth = config.depth_encoder
    point_goal = config.point_goal_encoder
    condition = config.condition_encoder
    decoder = config.trajectory_decoder

    depth_encoder = DepthObservationEncoder(
        model_dim=depth.model_dim,
        frame_tokens_height=depth.frame_tokens_height,
        frame_tokens_width=depth.frame_tokens_width,
        dropout=depth.dropout,
        max_depth_m=config.data.max_depth_m,
        focal_x_px=config.data.canonical_focal_x_px,
        focal_y_px=config.data.canonical_focal_y_px,
        camera_forward_offset_m=config.data.camera_forward_offset_m,
        camera_height_m=config.data.camera_height_m,
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
        planning_horizon_m=(
            config.data.future_steps * config.data.expert_waypoint_spacing_m
        ),
        max_depth_m=config.data.max_depth_m,
        model_dim=condition.model_dim,
        transformer_layers=condition.transformer_layers,
        transformer_heads=condition.transformer_heads,
        dropout=condition.dropout,
    )
    curve_codec = MetricCurvatureTrajectory(
        num_curvature_control_points=trajectory.num_curvature_control_points,
        degree=trajectory.curvature_spline_degree,
        num_path_points=trajectory.num_path_points,
        length_pretransform_mean=trajectory.length_pretransform_mean,
        length_pretransform_std=trajectory.length_pretransform_std,
        curvature_control_mean_inv_m=trajectory.curvature_control_mean_inv_m,
        curvature_control_std_inv_m=trajectory.curvature_control_std_inv_m,
    )
    trajectory_decoder = ConditionalCurveFlowDecoder(
        curve_tokens=curve_codec.num_curve_tokens,
        path_tokens=decoder.path_tokens,
        num_path_points=trajectory.num_path_points,
        planning_horizon_m=(
            config.data.future_steps * config.data.expert_waypoint_spacing_m
        ),
        curvature_scale_inv_m=trajectory.curvature_control_std_inv_m,
        model_dim=decoder.model_dim,
        layers=decoder.transformer_layers,
        heads=decoder.transformer_heads,
        dropout=decoder.dropout,
    )
    return CurveNavPolicy(
        depth_encoder=depth_encoder,
        condition_encoder=condition_encoder,
        trajectory_decoder=trajectory_decoder,
        curve_codec=curve_codec,
        flow_steps=decoder.flow_steps,
    )
