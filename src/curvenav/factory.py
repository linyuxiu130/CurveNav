"""Composition root for CurveNav modules."""

from curvenav.config import CurveNavConfig
from curvenav.conditioning import ConditionTransformer
from curvenav.encoders import DepthSequenceEncoder, MotionContextEncoder, TaskGoalEncoder
from curvenav.generative import RectifiedFlow
from curvenav.models import CurveNavPolicy, TransformerTrajectoryField
from curvenav.trajectory import PlanarBSplineCodec, PlanarScaleNormalizer


def build_policy(config: CurveNavConfig) -> CurveNavPolicy:
    """Assemble the one explicit CurveNav policy graph from validated components."""
    config.validate()
    trajectory = config.trajectory
    depth = config.depth_encoder
    goal = config.goal_encoder
    motion = config.motion_encoder
    condition_config = config.condition_encoder
    field_config = config.field
    rectified_flow_config = config.rectified_flow

    depth_encoder = DepthSequenceEncoder(
        model_dim=depth.model_dim,
        frame_tokens_per_side=depth.frame_tokens_per_side,
        dropout=depth.dropout,
    )
    goal_encoder = TaskGoalEncoder(model_dim=goal.model_dim, hidden_dim=goal.hidden_dim)
    motion_encoder = MotionContextEncoder(
        model_dim=motion.model_dim,
        hidden_dim=motion.hidden_dim,
    )
    condition_encoder = ConditionTransformer(
        model_dim=condition_config.model_dim,
        transformer_layers=condition_config.transformer_layers,
        transformer_heads=condition_config.transformer_heads,
        dropout=condition_config.dropout,
    )
    field = TransformerTrajectoryField(
        num_control_points=trajectory.num_control_points,
        model_dim=field_config.model_dim,
        transformer_layers=field_config.transformer_layers,
        transformer_heads=field_config.transformer_heads,
        dropout=field_config.dropout,
    )
    codec = PlanarBSplineCodec(
        num_control_points=trajectory.num_control_points,
        degree=trajectory.degree,
        num_path_points=trajectory.num_path_points,
    )
    rectified_flow = RectifiedFlow(
        field=field,
        num_control_points=trajectory.num_control_points,
        inference_steps=rectified_flow_config.inference_steps,
        source_cholesky=codec.origin_conditioned_source_cholesky(),
        source_std_xy=rectified_flow_config.source_std_xy,
    )
    return CurveNavPolicy(
        depth_encoder=depth_encoder,
        goal_encoder=goal_encoder,
        motion_encoder=motion_encoder,
        condition_encoder=condition_encoder,
        rectified_flow=rectified_flow,
        codec=codec,
        normalizer=PlanarScaleNormalizer(trajectory.scale_xy),
    )
