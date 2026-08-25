from dataclasses import replace

import torch

from curvenav import PolicyCondition, TrajectoryTarget, build_policy
from curvenav.config import (
    CurveNavConfig,
    DataConfig,
    DepthEncoderConfig,
    TrajectoryFlowConfig,
    ConditionEncoderConfig,
    PointGoalEncoderConfig,
    TrajectoryConfig,
)
from curvenav.models import GeometricTrajectoryEvaluator


def tiny_config() -> CurveNavConfig:
    return CurveNavConfig(
        data=DataConfig(observation_frames=4),
        trajectory=TrajectoryConfig(num_control_points=8, num_path_points=16),
        depth_encoder=DepthEncoderConfig(
            model_dim=32, frame_tokens_height=2, frame_tokens_width=2
        ),
        point_goal_encoder=PointGoalEncoderConfig(model_dim=32, hidden_dim=32),
        condition_encoder=ConditionEncoderConfig(
            model_dim=32, transformer_layers=1, transformer_heads=4
        ),
        trajectory_flow=TrajectoryFlowConfig(
            model_dim=32,
            transformer_layers=1,
            transformer_heads=4,
            inference_candidates=16,
            integration_steps=2,
        ),
    )


def identity_observation_transform(batch: int) -> torch.Tensor:
    transform = torch.zeros(batch, 4, 4)
    transform[..., 3] = 1.0
    return transform


def condition(batch: int = 2) -> PolicyCondition:
    point_goals = torch.stack(
        (torch.linspace(2.0, 4.0, batch), torch.linspace(-1.0, 1.0, batch)),
        dim=-1,
    )
    return PolicyCondition(
        depth=torch.rand(batch, 4, 1, 126, 224),
        point_goal=point_goals,
        observation_to_current=identity_observation_transform(batch),
        observation_valid=torch.ones(batch, 4, dtype=torch.bool),
    )


def test_policy_trains_every_module_and_returns_geometry_selected_candidate() -> None:
    torch.manual_seed(0)
    policy = build_policy(tiny_config())
    inputs = condition(batch=8)
    controls = torch.randn(8, 8, 2)
    controls[:, 0] = 0
    reference_path = policy.codec.decode_equal_arc(controls)
    losses = policy(inputs, TrajectoryTarget(controls, reference_path))
    for value in (
        losses.loss,
        losses.flow_loss,
        losses.path_loss,
        losses.tangent_loss,
    ):
        assert value.ndim == 0 and torch.isfinite(value)
    torch.testing.assert_close(
        losses.loss,
        losses.flow_loss + 0.5 * losses.path_loss + 0.1 * losses.tangent_loss,
    )
    losses.loss.backward()
    gradients = [
        parameter.grad for parameter in policy.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)

    policy.eval()
    first = policy.sample(inputs)
    torch.manual_seed(999)
    second = policy.sample(inputs)
    assert first.control_points.shape == (8, 8, 2)
    assert first.path.shape == (8, 16, 2)
    assert first.candidate_control_points.shape == (8, 16, 8, 2)
    assert first.candidate_paths.shape == (8, 16, 16, 2)
    assert first.candidate_costs.shape == (8, 16)
    assert first.candidate_minimum_clearance_m.shape == (8, 16)
    assert torch.isfinite(first.candidate_costs).all()
    assert torch.equal(first.control_points, second.control_points)
    selected = first.candidate_costs.argmin(dim=1)
    batch = torch.arange(8)
    torch.testing.assert_close(
        first.control_points, first.candidate_control_points[batch, selected]
    )
    assert torch.equal(first.control_points[:, 0], torch.zeros(8, 2))


def test_point_goal_encoder_retains_range_until_xnavdp_clip_distance() -> None:
    encoder = build_policy(tiny_config()).condition_encoder.point_goal_encoder.eval()
    with torch.no_grad():
        far = encoder(torch.tensor([[4.0, 0.0], [20.0, 0.0], [40.0, 0.0], [80.0, 0.0]]))
        near = encoder(torch.tensor([[1.0, 0.0]]))
    assert not torch.allclose(far[0], far[1])
    torch.testing.assert_close(far[2], far[3])
    assert not torch.allclose(far[0], near[0])


def test_invalid_padded_frames_cannot_change_the_prediction() -> None:
    policy = build_policy(tiny_config())
    first = condition(batch=1)
    first.observation_valid[:, :2] = False
    second = PolicyCondition(
        depth=first.depth.clone(),
        point_goal=first.point_goal,
        observation_to_current=first.observation_to_current.clone(),
        observation_valid=first.observation_valid,
    )
    second.depth[:, :2] = torch.rand_like(second.depth[:, :2]) * 100.0
    second.observation_to_current[:, :2] = torch.rand_like(
        second.observation_to_current[:, :2]
    ) * 100.0
    first_encoded = policy.encode_condition(first)
    second_encoded = policy.encode_condition(second)
    torch.testing.assert_close(first_encoded.tokens, second_encoded.tokens)


def test_eight_control_contract_is_unique() -> None:
    invalid = replace(
        tiny_config(), trajectory=replace(tiny_config().trajectory, num_control_points=12)
    )
    try:
        invalid.validate()
    except ValueError as error:
        assert "exactly eight" in str(error)
    else:
        raise AssertionError("12-control trajectory must be rejected")


def test_geometric_evaluator_prefers_clear_goal_reaching_candidate() -> None:
    evaluator = GeometricTrajectoryEvaluator(
        image_height=126,
        image_width=224,
        focal_x_px=166.80851063829786,
        focal_y_px=166.80851063829786,
        max_depth_m=5.0,
        camera_forward_offset_m=0.0,
        camera_height_m=0.30,
        camera_downward_pitch_degrees=0.0,
        minimum_obstacle_height_m=0.05,
        robot_height_m=0.70,
        robot_radius_m=0.25,
        safety_margin_m=0.10,
        discount_factor=0.95,
        clearance_weight=10.0,
        length_weight=1.0,
        goal_weight=1.0,
    )
    depth = torch.ones(1, 126, 224)
    depth[:, :, 112] = 0.2
    paths = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
                [[0.0, 0.0], [0.5, 0.2], [1.0, 0.5], [1.5, 0.5]],
            ]
        ]
    )
    costs = evaluator(paths, depth, torch.tensor([[1.5, 0.5]]))
    assert costs.minimum_clearance[0, 0] < evaluator.safe_center_distance_m
    assert costs.minimum_clearance[0, 1] > evaluator.safe_center_distance_m
    assert costs.total.argmin(dim=1).item() == 1


def test_geometric_evaluator_applies_camera_extrinsics_before_planar_distance() -> None:
    evaluator = GeometricTrajectoryEvaluator(
        image_height=126,
        image_width=224,
        focal_x_px=166.80851063829786,
        focal_y_px=166.80851063829786,
        max_depth_m=5.0,
        camera_forward_offset_m=0.28618,
        camera_height_m=0.62532,
        camera_downward_pitch_degrees=10.0,
        minimum_obstacle_height_m=0.05,
        robot_height_m=0.70,
        robot_radius_m=0.25,
        safety_margin_m=0.10,
        discount_factor=0.95,
        clearance_weight=10.0,
        length_weight=1.0,
        goal_weight=1.0,
    )
    depth = torch.ones(1, 126, 224)
    depth[:, 63, 112] = 0.2
    points, valid = evaluator._obstacle_points(depth)
    center = points[0, 63, 112]
    assert valid[0, 63, 112]
    torch.testing.assert_close(
        center,
        torch.tensor([0.28618 + torch.cos(torch.deg2rad(torch.tensor(10.0))), 0.0]),
    )
    clear = evaluator(
        torch.tensor([[[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]]]),
        torch.ones(1, 126, 224),
        torch.tensor([[0.6, 0.0]]),
    )
    assert clear.minimum_clearance.item() == evaluator.max_depth_m


def test_flow_endpoint_reconstruction_matches_linear_path_identity() -> None:
    flow = build_policy(tiny_config()).trajectory_flow
    clean = torch.randn(3, 8, 2)
    clean[:, 0] = 0
    noisy, time, velocity = flow.training_pair(clean)
    reconstructed = flow.reconstruct_clean(noisy, time, velocity)
    torch.testing.assert_close(reconstructed, clean)


def test_sand_spatial_tokens_and_navdp_compression_have_fixed_contract() -> None:
    policy = build_policy(tiny_config())
    depth = torch.full((1, 4, 1, 126, 224), 0.4)
    observation = policy.depth_encoder(
        depth,
        identity_observation_transform(1),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert observation.tokens.shape == (1, 4, 4, 32)

    encoded = policy.encode_condition(condition(batch=1))
    assert policy.condition_encoder.compressed_tokens == 4 * 16
    assert encoded.tokens.shape == (1, 1 + 4 * 16, 32)


def test_planar_backprojection_uses_observation_transform() -> None:
    policy = build_policy(tiny_config()).eval()
    depth = torch.full((1, 4, 1, 126, 224), 0.4)
    identity = identity_observation_transform(1)
    translated = identity.clone()
    translated[:, 0, 0] = -1.0
    with torch.no_grad():
        first = policy.depth_encoder(depth, identity, torch.ones(1, 4, dtype=torch.bool))
        second = policy.depth_encoder(depth, translated, torch.ones(1, 4, dtype=torch.bool))
    torch.testing.assert_close(
        second.planar_points[:, 0, :, 0],
        first.planar_points[:, 0, :, 0] - 1.0,
    )
