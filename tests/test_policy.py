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
from curvenav.training.optimizer import build_optimizer
from curvenav.models import FlowPrediction, TRAINING_LOSS_NAMES
from curvenav.types import ConditionFeatures


def tiny_config() -> CurveNavConfig:
    return CurveNavConfig(
        data=DataConfig(observation_frames=4),
        trajectory=TrajectoryConfig(num_path_points=16),
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


def test_policy_trains_every_module_and_returns_one_deterministic_trajectory() -> None:
    torch.manual_seed(0)
    policy = build_policy(tiny_config())
    inputs = condition(batch=8)
    curve_coordinates = 0.2 * torch.randn(8, 8, 2)
    reference_path, _, _ = policy.curve_codec(
        curve_coordinates,
        inputs.point_goal,
    )
    target_controls = policy.target_codec.encode(reference_path)
    losses = policy(inputs, TrajectoryTarget(target_controls, reference_path))
    for value in (
        losses.loss,
        losses.flow_loss,
        losses.flow_velocity_loss,
        losses.flow_endpoint_loss,
        losses.flow_consistency_loss,
        losses.path_loss,
        losses.tangent_loss,
        losses.proposal_loss,
    ):
        assert value.ndim == 0 and torch.isfinite(value)
    assert len(losses.logging_values()) == len(TRAINING_LOSS_NAMES)
    torch.testing.assert_close(
        losses.loss,
        losses.flow_loss
        + losses.path_loss
        + losses.tangent_loss
        + losses.proposal_loss,
    )
    torch.testing.assert_close(
        losses.flow_loss,
        losses.flow_velocity_loss
        + losses.flow_endpoint_loss
        + 0.1 * losses.flow_consistency_loss,
    )
    losses.loss.backward()
    gradients = [
        parameter.grad for parameter in policy.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )

    policy.eval()
    first = policy.sample(inputs)
    torch.manual_seed(999)
    second = policy.sample(inputs)
    assert first.path.shape == (8, 16, 2)
    assert first.heading.shape == (8, 16)
    assert first.curvature.shape == (8, 16)
    assert torch.equal(first.path, second.path)
    assert torch.isfinite(first.path).all()
    assert torch.equal(first.path[:, 0], torch.zeros(8, 2))
    assert first.curvature.abs().max() < 8.0


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
    second.observation_to_current[:, :2] = (
        torch.rand_like(second.observation_to_current[:, :2]) * 100.0
    )
    first_encoded = policy.encode_condition(first)
    second_encoded = policy.encode_condition(second)
    torch.testing.assert_close(first_encoded.tokens, second_encoded.tokens)


def test_depth_tokens_do_not_depend_on_other_batch_members() -> None:
    policy = build_policy(tiny_config()).train()
    inputs = condition(batch=2)
    single = PolicyCondition(
        depth=inputs.depth[:1],
        point_goal=inputs.point_goal[:1],
        observation_to_current=inputs.observation_to_current[:1],
        observation_valid=inputs.observation_valid[:1],
    )
    with torch.no_grad():
        alone = policy.encode_condition(single).tokens
        together = policy.encode_condition(inputs).tokens[:1]
    torch.testing.assert_close(alone, together, rtol=2e-5, atol=2e-5)
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in policy.modules()
    )


def test_seven_curvature_control_contract_is_unique() -> None:
    invalid = replace(
        tiny_config(),
        trajectory=replace(
            tiny_config().trajectory,
            num_curvature_control_points=8,
        ),
    )
    try:
        invalid.validate()
    except ValueError as error:
        assert "exactly seven" in str(error)
    else:
        raise AssertionError("an alternate curve token shape must be rejected")


def test_flow_endpoint_reconstruction_matches_linear_path_identity() -> None:
    flow = build_policy(tiny_config()).trajectory_flow
    clean = torch.randn(3, flow.future_tokens, 2)
    policy = build_policy(tiny_config())
    free_mask = policy.curve_codec.free_mask[None].expand(3, -1, -1)
    clean = clean * free_mask
    source = torch.randn_like(clean) * free_mask
    state, time, velocity = flow.training_path(clean, source, free_mask)
    torch.testing.assert_close(
        state,
        (1.0 - time[:, None, None]) * source + time[:, None, None] * clean,
    )
    torch.testing.assert_close(source * free_mask, source)
    reconstructed = flow.reconstruct_clean(state, time, velocity)
    torch.testing.assert_close(reconstructed, clean)


def test_flow_training_source_is_the_deterministic_conditioned_proposal() -> None:
    policy = build_policy(tiny_config())
    assert torch.count_nonzero(policy.trajectory_flow.velocity_projection.weight) > 0
    assert torch.count_nonzero(policy.trajectory_flow.velocity_projection.bias) == 0
    assert torch.count_nonzero(policy.trajectory_flow.endpoint_projection.weight) > 0
    assert torch.count_nonzero(policy.trajectory_flow.endpoint_projection.bias) == 0
    inputs = condition(batch=4)
    encoded = policy.encode_condition(inputs)
    clean = torch.randn(4, 8, 2)
    mask = policy.curve_codec.free_mask[None].expand_as(clean)
    source = policy.curve_proposal(encoded, mask)
    repeated = policy.curve_proposal(encoded, mask)
    torch.testing.assert_close(source, repeated)
    _, _, target_velocity = policy.trajectory_flow.training_path(
        clean,
        source,
        mask,
    )
    torch.testing.assert_close(clean * mask - target_velocity, source)
    assert torch.equal(
        source[:, ~policy.curve_codec.free_mask],
        torch.zeros_like(source[:, ~policy.curve_codec.free_mask]),
    )


def test_heun_integration_transports_the_conditioned_proposal() -> None:
    policy = build_policy(tiny_config())
    flow = policy.trajectory_flow
    batch = 2
    mask = policy.curve_codec.free_mask[None].expand(batch, -1, -1)
    source = torch.randn(batch, flow.future_tokens, 2) * mask
    target = torch.randn(batch, flow.future_tokens, 2) * mask
    velocity = target - source
    encoded = ConditionFeatures(
        tokens=torch.zeros(batch, 1, 32),
        route_token=torch.zeros(batch, 32),
    )

    def constant_velocity(state, time, condition_tokens, route_token):
        return FlowPrediction(velocity=velocity, endpoint=target)

    flow.forward = constant_velocity
    transported = flow.integrate(
        encoded,
        source,
        integration_steps=4,
        free_mask=mask,
    )
    torch.testing.assert_close(transported, target)


def test_flow_curve_coordinate_normalization_is_exact_and_order_one() -> None:
    flow = build_policy(tiny_config()).trajectory_flow
    coordinates = torch.tensor(
        [[[0.05, -0.02], [0.08, 0.0], [-0.04, 0.0]]]
    )
    normalized = flow.normalize_curve_coordinates(coordinates)
    torch.testing.assert_close(normalized, 8.0 * coordinates)
    torch.testing.assert_close(
        flow.denormalize_curve_coordinates(normalized),
        coordinates,
    )
    assert normalized.abs().max() >= 0.5


def test_conditioned_proposal_decodes_to_one_bounded_curve() -> None:
    policy = build_policy(tiny_config())
    inputs = condition(batch=2)
    encoded = policy.encode_condition(inputs)
    mask = policy.curve_codec.free_mask[None].expand(2, -1, -1)
    source = policy.curve_proposal(encoded, mask)
    path, heading, curvature = policy.curve_codec.decode(
        policy.trajectory_flow.denormalize_curve_coordinates(source),
        inputs.point_goal,
    )
    assert source.shape == (2, 8, 2)
    assert path.shape == (2, 16, 2)
    assert heading.shape == (2, 16)
    assert curvature.abs().max() < policy.curve_codec.maximum_curvature_inv_m


def test_future_flow_has_no_generated_history_or_candidate_set() -> None:
    policy = build_policy(tiny_config())
    flow = policy.trajectory_flow
    assert flow.future_tokens == policy.curve_codec.num_curve_tokens
    assert not hasattr(flow, "history_tokens")
    assert not hasattr(flow, "total_tokens")
    assert not hasattr(policy, "candidate_bases")
    assert policy.curve_proposal.curve_tokens == flow.future_tokens


def test_straight_target_is_an_exact_curve_projection() -> None:
    codec = build_policy(tiny_config()).curve_codec
    goal = torch.tensor([[10.0, 0.0]])
    reference = torch.zeros(1, 16, 2)
    reference[0, :, 0] = torch.linspace(0.0, 3.6, 16)
    coordinates = codec.encode_target(reference, goal)
    path, heading, curvature = codec.decode(coordinates, goal)
    torch.testing.assert_close(path, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(heading, torch.zeros_like(heading))
    torch.testing.assert_close(curvature, torch.zeros_like(curvature))


def test_pointgoal_scaling_and_zero_goal_are_structural() -> None:
    codec = build_policy(tiny_config()).curve_codec
    zero_coordinates = torch.zeros(1, codec.num_curve_tokens, 2)
    far_path, _, _ = codec(
        zero_coordinates,
        torch.tensor([[10.0, 0.0]]),
    )
    near_path, _, _ = codec(
        zero_coordinates,
        torch.tensor([[1.2, 0.0]]),
    )
    stopped, _, _ = codec(
        torch.randn_like(zero_coordinates),
        torch.zeros(1, 2),
    )
    torch.testing.assert_close(far_path[0, -1], torch.tensor([3.6, 0.0]))
    torch.testing.assert_close(near_path[0, -1], torch.tensor([1.2, 0.0]))
    torch.testing.assert_close(stopped, torch.zeros_like(stopped))


def test_curvature_bspline_has_a_hard_continuous_bound() -> None:
    codec = build_policy(tiny_config()).curve_codec
    coordinates = 2.0 * torch.randn(32, 8, 2)
    path, heading, curvature = codec.decode(
        coordinates,
        torch.tensor([[4.0, 1.0]]).expand(32, -1),
    )
    assert torch.all(heading[:, 0].abs() < torch.pi / 2)
    assert curvature.abs().max() < codec.maximum_curvature_inv_m
    torch.testing.assert_close(path[:, 0], torch.zeros_like(path[:, 0]))
    assert torch.equal(codec.free_mask[0], torch.tensor([True, True]))
    assert torch.all(codec.free_mask[1:, 0])
    assert not torch.any(codec.free_mask[1:, 1])


def test_flow_time_embedding_resolves_the_full_unit_interval() -> None:
    embedding = build_policy(tiny_config()).trajectory_flow.time_embedding
    assert embedding.frequency[0].item() == 1.0
    assert embedding.frequency[-1].item() == 1000.0


def test_sand_spatial_tokens_and_geometry_query_compression_have_fixed_contract() -> (
    None
):
    policy = build_policy(tiny_config())
    depth = torch.full((1, 4, 1, 126, 224), 0.4)
    observation = policy.depth_encoder(
        depth,
        identity_observation_transform(1),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert observation.tokens.shape == (1, 4, 4, 32)

    encoded = policy.encode_condition(condition(batch=1))
    encoder = policy.condition_encoder
    assert encoder.current_geometry_query_embedding.shape == (1, 32, 32)
    assert encoder.context_geometry_query_embedding.shape == (1, 32, 32)
    mask = encoder.geometry_attention_mask
    assert mask.shape == (64, 16)
    assert torch.all(mask[:32, :12])
    assert not torch.any(mask[:32, 12:])
    assert not torch.any(mask[32:])
    assert encoder.route_query_embedding.shape == (1, 4, 32)
    assert encoded.tokens.shape == (1, 4 + 1 + 4 + 64, 32)
    assert policy.curve_proposal.token_embedding.shape == (1, 8, 32)


def test_learned_token_embeddings_are_not_weight_decayed() -> None:
    policy = build_policy(tiny_config())
    optimizer = build_optimizer(policy, learning_rate=4e-4, weight_decay=1e-2)
    no_decay = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    for name, parameter in policy.named_parameters():
        if "embedding" in name:
            assert id(parameter) in no_decay, name


def test_planar_backprojection_uses_observation_transform() -> None:
    policy = build_policy(tiny_config()).eval()
    depth = torch.full((1, 4, 1, 126, 224), 0.4)
    identity = identity_observation_transform(1)
    translated = identity.clone()
    translated[:, 0, 0] = -1.0
    with torch.no_grad():
        first, _ = policy.depth_encoder.planar_projector(depth, identity)
        second, _ = policy.depth_encoder.planar_projector(depth, translated)
    torch.testing.assert_close(
        second[:, 0, :, 0],
        first[:, 0, :, 0] - 1.0,
    )
