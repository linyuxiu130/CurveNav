"""Composition root for the one CurveNav policy graph."""

from curvenav.config import CurveNavConfig
from curvenav.conditioning import PolicyConditionEncoder
from curvenav.encoders import (
    ConfigurationSpaceEncoder,
    DepthObservationEncoder,
    MetricDepthProjector,
)
from curvenav.models import ConditionalCurveFlowDecoder, CurveNavPolicy
from curvenav.models.evaluator import TrajectoryEvaluator
from curvenav.trajectory import IncrementalBSplineTrajectory


def build_policy(config: CurveNavConfig) -> CurveNavPolicy:
    """Assemble the only production policy graph from validated components."""
    config.validate()
    trajectory = config.trajectory
    depth = config.depth_encoder
    condition = config.condition_encoder
    decoder = config.trajectory_decoder
    planning_horizon_m = (
        config.data.future_steps * config.data.expert_waypoint_spacing_m
    )

    depth_encoder = DepthObservationEncoder(
        model_dim=depth.model_dim,
        frame_tokens_height=depth.frame_tokens_height,
        frame_tokens_width=depth.frame_tokens_width,
        dropout=depth.dropout,
        max_depth_m=config.data.max_depth_m,
        planning_horizon_m=planning_horizon_m,
        geometry=config.data.robot_geometry,
    )
    configuration_encoder = ConfigurationSpaceEncoder(
        model_dim=condition.model_dim,
        planning_horizon_m=planning_horizon_m,
        grid_size=condition.bev_grid_size,
    )
    condition_encoder = PolicyConditionEncoder(
        configuration_encoder,
        observation_frames=config.data.observation_frames,
        planning_horizon_m=planning_horizon_m,
        model_dim=condition.model_dim,
    )
    curve_codec = IncrementalBSplineTrajectory(
        num_control_points=trajectory.num_control_points,
        degree=trajectory.spline_degree,
        num_path_points=trajectory.num_path_points,
        control_increment_mean_xy_m=trajectory.control_increment_mean_xy_m,
        control_increment_std_xy_m=trajectory.control_increment_std_xy_m,
    )
    trajectory_decoder = ConditionalCurveFlowDecoder(
        control_tokens=curve_codec.num_control_tokens,
        coordinate_dim=curve_codec.coordinate_dim,
        model_dim=decoder.model_dim,
        layers=decoder.transformer_layers,
        heads=decoder.transformer_heads,
        dropout=decoder.dropout,
        planning_horizon_m=planning_horizon_m,
        path_to_increment_weight=curve_codec.increment_basis.transpose(0, 1),
    )
    return CurveNavPolicy(
        depth_encoder=depth_encoder,
        condition_encoder=condition_encoder,
        trajectory_decoder=trajectory_decoder,
        trajectory_evaluator=TrajectoryEvaluator(decoder.model_dim, decoder.transformer_heads, planning_horizon_m),
        integration_steps=decoder.integration_steps,
        curve_codec=curve_codec,
        planning_horizon_m=planning_horizon_m,
    )


def build_evaluation_projector(config: CurveNavConfig) -> MetricDepthProjector:
    """Build the raw-depth diagnostic geometry outside the policy graph."""
    data = config.data
    return MetricDepthProjector(
        token_height=config.depth_encoder.frame_tokens_height,
        token_width=config.depth_encoder.frame_tokens_width,
        max_depth_m=data.max_depth_m,
        planning_horizon_m=data.future_steps * data.expert_waypoint_spacing_m,
        geometry=data.robot_geometry,
    )
