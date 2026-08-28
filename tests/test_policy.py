from dataclasses import replace
import math

import torch

from curvenav import PolicyCondition, TrajectoryTarget, build_policy
from curvenav.config import (
    CurveNavConfig,
    DataConfig,
    DepthEncoderConfig,
    TrajectoryDecoderConfig,
    ConditionEncoderConfig,
    PointGoalEncoderConfig,
    TrajectoryConfig,
)
from curvenav.training.optimizer import build_optimizer
from curvenav.models import TRAINING_LOSS_NAMES


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
        trajectory_decoder=TrajectoryDecoderConfig(
            model_dim=32,
            transformer_layers=1,
            transformer_heads=4,
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
    curve_coordinates = 0.2 * torch.randn(8, 8)
    target_values = policy.curve_codec.values_from_coordinates(curve_coordinates)
    losses = policy(
        inputs,
        TrajectoryTarget(target_values),
    )
    for value in (
        losses.loss,
        losses.flow_loss,
        losses.clearance_loss,
    ):
        assert value.ndim == 0 and torch.isfinite(value)
    assert len(losses.logging_values()) == len(TRAINING_LOSS_NAMES)
    torch.testing.assert_close(
        losses.loss,
        losses.flow_loss + losses.clearance_loss,
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
    assert torch.isfinite(first.curvature).all()


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
    assert first_encoded.tokens.shape[1] == 37


def test_history_attention_mask_has_kernel_contiguous_last_dimension() -> None:
    compressor = build_policy(tiny_config()).condition_encoder.history_compressor
    observed: list[tuple[int, ...]] = []
    handle = compressor.attention.register_forward_pre_hook(
        lambda _module, inputs, kwargs: observed.append(kwargs["attn_mask"].stride()),
        with_kwargs=True,
    )
    query = torch.randn(2, 32, 32)
    memory = torch.randn(2, 13, 32)
    padding = torch.zeros(2, 13, dtype=torch.bool)
    padding[:, :4] = True
    compressor(query, memory, padding)
    handle.remove()
    assert observed[0][-1] == 1


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


def test_flow_source_and_solver_contract_is_gaussian_train_deterministic_inference() -> None:
    policy = build_policy(tiny_config())
    assert policy.flow_steps == 8
    assert policy.inference_source.shape == (1, 8)
    torch.testing.assert_close(
        torch.linalg.vector_norm(policy.inference_source),
        torch.tensor(math.sqrt(8.0)),
    )
    assert torch.count_nonzero(policy.inference_source) == 8
    assert not hasattr(policy.curve_codec, "flow_velocity_metric")


def test_training_uses_gaussian_conditional_flow_matching() -> None:
    policy = build_policy(tiny_config())
    observed: dict[str, torch.Tensor] = {}

    class RecordingVelocity(torch.nn.Module):
        curve_tokens = 8

        def forward(self, state, time, encoded, path, heading, curvature):
            observed["state"] = state.detach()
            observed["time"] = time.detach()
            return torch.zeros_like(state)

    policy.trajectory_decoder = RecordingVelocity()
    inputs = condition(batch=2)
    target_values = torch.tensor(
        [
            [0.8, 0.1, 0.05, 0.0, -0.05, -0.1, -0.05, 0.0],
            [0.6, -0.1, -0.05, 0.0, 0.05, 0.1, 0.05, 0.0],
        ]
    )
    clean = policy.curve_codec.coordinates_from_values(target_values)
    torch.manual_seed(123)
    source = torch.randn_like(clean)
    expected_time = torch.rand(clean.shape[0])
    endpoint = torch.rand_like(expected_time) < (1.0 / 9.0)
    expected_time = torch.where(endpoint, torch.zeros_like(expected_time), expected_time)
    torch.manual_seed(123)
    losses = policy.training_loss(inputs, TrajectoryTarget(target_values))

    torch.testing.assert_close(
        observed["state"],
        (1.0 - expected_time[:, None]) * source + expected_time[:, None] * clean,
    )
    torch.testing.assert_close(observed["time"], expected_time)
    expected = (clean - source).square().mean()
    torch.testing.assert_close(losses.flow_loss, expected)


def test_configuration_space_loss_is_soft_differentiable_and_masked() -> None:
    from curvenav.models.safety import configuration_space_clearance_loss

    colliding = torch.tensor(
        [[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]], requires_grad=True
    )
    obstacle = torch.tensor([[[0.5, 0.0], [10.0, 10.0]]])
    valid = torch.tensor([[True, False]])
    loss = configuration_space_clearance_loss(colliding, obstacle, valid)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(colliding.grad).all()

    safe = colliding.detach() + torch.tensor([[[0.0, 1.0]]])
    torch.testing.assert_close(
        configuration_space_clearance_loss(safe, obstacle, valid),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        configuration_space_clearance_loss(
            colliding.detach(), obstacle, torch.zeros_like(valid)
        ),
        torch.tensor(0.0),
    )


def test_conditioned_decoder_returns_one_smooth_metric_curve() -> None:
    policy = build_policy(tiny_config())
    inputs = condition(batch=2)
    encoded = policy.encode_condition(inputs)
    noisy = torch.randn(2, 8)
    time = torch.tensor([0.2, 0.7])
    predicted_velocity = policy._predict_velocity(noisy, time, encoded)
    repeated = policy._predict_velocity(noisy, time, encoded)
    torch.testing.assert_close(predicted_velocity, repeated)
    prediction = policy.sample(inputs)
    assert prediction.path.shape == (2, 16, 2)
    assert prediction.heading.shape == (2, 16)
    assert torch.isfinite(prediction.curvature).all()


def test_decoder_has_no_generated_history_candidate_or_second_stage() -> None:
    policy = build_policy(tiny_config())
    decoder = policy.trajectory_decoder
    assert decoder.curve_tokens == policy.curve_codec.num_curve_tokens
    assert not hasattr(policy, "trajectory_flow")
    assert not hasattr(policy, "curve_proposal")
    assert not hasattr(policy, "candidate_bases")


def test_curve_length_is_independent_of_pointgoal_distance() -> None:
    codec = build_policy(tiny_config()).curve_codec
    zero_coordinates = torch.zeros(1, codec.num_curve_tokens)
    path, _, _ = codec(zero_coordinates)
    values = codec.values_from_coordinates(zero_coordinates)
    torch.testing.assert_close(
        torch.linalg.vector_norm(path[:, 1:] - path[:, :-1], dim=-1).sum(1),
        values[:, 0],
    )
    assert not hasattr(codec, "planning_horizon_m")


def test_curvature_bspline_is_smooth_without_a_model_saturation() -> None:
    codec = build_policy(tiny_config()).curve_codec
    coordinates = 20.0 * torch.ones(32, 8)
    path, heading, curvature = codec.decode(coordinates)
    torch.testing.assert_close(heading[:, 0], torch.zeros_like(heading[:, 0]))
    assert curvature.abs().max() > 4.0
    torch.testing.assert_close(path[:, 0], torch.zeros_like(path[:, 0]))
    assert not hasattr(codec, "free_mask")
    assert codec.num_curve_tokens == 8


def test_metric_curve_values_round_trip_through_flow_coordinates() -> None:
    codec = build_policy(tiny_config()).curve_codec
    values = torch.tensor(
        [[2.2, -0.8, -0.4, -0.1, 0.0, 0.2, 0.5, 0.9]]
    )
    coordinates = codec.coordinates_from_values(values)
    torch.testing.assert_close(codec.values_from_coordinates(coordinates), values)


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
    projection = policy.depth_encoder.metric_projector(
        depth,
        identity_observation_transform(1),
    )
    assert observation.tokens.shape == (1, 4, 4, 32)
    assert projection.obstacle_points.shape == (1, 4, 4, 2)
    assert projection.obstacle_valid.shape == (1, 4, 4)
    assert projection.obstacle_valid.any()
    far = policy.depth_encoder.metric_projector(
        torch.ones_like(depth),
        identity_observation_transform(1),
    )
    assert not far.obstacle_valid.any()

    encoded = policy.encode_condition(condition(batch=1))
    encoder = policy.condition_encoder
    assert encoder.history_query_embedding.shape == (1, 32, 32)
    assert encoder.history_null_token.shape == (1, 1, 32)
    assert not hasattr(encoder, "state_encoder")
    assert not hasattr(encoder, "route_query_embedding")
    assert encoded.tokens.shape == (1, 1 + 4 + 32, 32)
    assert policy.trajectory_decoder.position_embedding.shape == (1, 8, 32)
    assert policy.trajectory_decoder.path_position_embedding.shape == (1, 16, 32)
    assert len(encoder.context_blocks) == 1


def test_continuous_learned_token_embeddings_are_weight_decayed() -> None:
    policy = build_policy(tiny_config())
    optimizer = build_optimizer(policy, learning_rate=4e-4, weight_decay=1e-2)
    decay = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    learned_token_names = ("position_embedding", "history_query_embedding")
    for name, parameter in policy.named_parameters():
        if name.endswith(learned_token_names):
            assert id(parameter) in decay, name


def test_flow_decoder_depends_on_state_and_time() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(condition(batch=2))
    decoder = policy.trajectory_decoder
    noisy = torch.zeros(2, 8)
    with torch.no_grad():
        reference = policy._predict_velocity(noisy, torch.zeros(2), encoded)
        changed_curve = policy._predict_velocity(noisy + 1.0, torch.zeros(2), encoded)
        changed_time = policy._predict_velocity(noisy, torch.ones(2), encoded)

    assert not torch.allclose(changed_curve, reference)
    assert not torch.allclose(changed_time, reference)


def test_history_query_radius_is_not_a_model_input() -> None:
    policy = build_policy(tiny_config()).eval()
    inputs = condition(batch=2)
    encoder = policy.condition_encoder
    with torch.no_grad():
        reference = policy.encode_condition(inputs)
        encoder.history_query_embedding.mul_(100_000.0)
        scaled = policy.encode_condition(inputs)

    torch.testing.assert_close(scaled.tokens, reference.tokens, atol=2e-5, rtol=2e-5)


def test_curve_residual_carrier_stays_float32_under_fp16_autocast() -> None:
    if not torch.cuda.is_available():
        return
    policy = build_policy(tiny_config()).cuda().eval()
    decoder = policy.trajectory_decoder
    inputs = condition(batch=2)
    inputs = PolicyCondition(
        depth=inputs.depth.cuda(),
        point_goal=inputs.point_goal.cuda(),
        observation_to_current=inputs.observation_to_current.cuda(),
        observation_valid=inputs.observation_valid.cuda(),
    )
    encoded = policy.encode_condition(inputs)
    observed_dtype: list[torch.dtype] = []
    handle = decoder.blocks[0].register_forward_pre_hook(
        lambda _module, inputs: observed_dtype.append(inputs[0].dtype)
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = policy._predict_velocity(
            torch.randn(2, 8, device="cuda"),
            torch.tensor([0.2, 0.7], device="cuda"),
            encoded,
        )
    handle.remove()

    assert observed_dtype == [torch.float32]
    assert torch.isfinite(output).all()


def test_expert_projection_uses_the_production_curve_manifold() -> None:
    codec = build_policy(tiny_config()).curve_codec
    goal = torch.tensor([[5.0, 1.0], [4.0, -2.0]])
    values = torch.tensor(
        [
            [0.8, 0.02, 0.04, 0.06, 0.08, 0.06, 0.04, 0.02],
            [0.7, -0.03, -0.05, -0.08, -0.08, -0.05, -0.03, -0.01],
        ]
    )
    source, _, _ = codec.decode_values(values)
    projected_values, projected, _, curvature = codec.project_expert(source)
    assert projected_values.shape == values.shape
    assert torch.linalg.vector_norm(projected - source, dim=-1).mean() < 0.01
    assert torch.isfinite(curvature).all()


def test_expert_projection_uses_the_regularized_heading_solve() -> None:
    codec = build_policy(tiny_config()).curve_codec
    design = codec.heading_control_matrix[1:]
    controls = codec.num_curvature_control_points
    difference = torch.zeros(controls - 1, controls)
    index = torch.arange(controls - 1)
    difference[index, index] = -1.0
    difference[index, index + 1] = 1.0
    normal = design.T @ design + 1e-3 * (difference.T @ difference)
    torch.testing.assert_close(
        normal @ codec.heading_fit_regularized_inverse,
        design.T,
        atol=2e-5,
        rtol=2e-5,
    )


def test_metric_xyz_backprojection_uses_camera_height_and_observation_transform() -> None:
    policy = build_policy(tiny_config()).eval()
    depth = torch.full((1, 4, 1, 126, 224), 0.4)
    identity = identity_observation_transform(1)
    translated = identity.clone()
    translated[:, 0, 0] = -1.0
    with torch.no_grad():
        first = policy.depth_encoder.metric_projector(depth, identity).points
        second = policy.depth_encoder.metric_projector(depth, translated).points
    assert first.shape[-1] == 3
    assert policy.depth_encoder.metric_projector.camera_height_m == 0.62532
    torch.testing.assert_close(
        second[:, 0, :, 0],
        first[:, 0, :, 0] - 1.0,
    )
    torch.testing.assert_close(second[..., 2], first[..., 2])


def test_body_obstacle_pooling_cannot_be_occluded_by_nearer_floor() -> None:
    policy = build_policy(tiny_config()).eval()
    depth = torch.ones(1, 4, 1, 126, 224)
    # These pixels share one adaptive cell.  The 1.5 m lower pixel reaches the
    # ground, while the slightly farther 1.6 m pixel intersects the body.
    depth[:, -1, 0, 109, 100] = 1.5 / 5.0
    depth[:, -1, 0, 95, 100] = 1.6 / 5.0
    projection = policy.depth_encoder.metric_projector(
        depth,
        identity_observation_transform(1),
    )
    valid = projection.obstacle_valid[:, -1]

    assert valid.sum() == 1
    torch.testing.assert_close(
        projection.depth[:, -1][valid],
        torch.tensor([1.6]),
    )


def test_nearest_depth_is_backprojected_with_its_own_pixel_ray() -> None:
    projector = build_policy(tiny_config()).depth_encoder.metric_projector
    depth = torch.ones(1, 4, 1, 126, 224)
    depth[:, :, :, 10, 20] = 0.4
    projection = projector(depth, identity_observation_transform(1))
    optical_y = 2.0 * (10.0 - 63.0) / 166.80851063829786
    pitch = math.radians(10.0)
    expected = torch.tensor(
        [
            0.28618 + math.cos(pitch) * 2.0 - math.sin(pitch) * optical_y,
            -2.0 * (20.0 - 112.0) / 166.80851063829786,
            0.62532 - math.cos(pitch) * optical_y - math.sin(pitch) * 2.0,
        ]
    )
    torch.testing.assert_close(
        projection.depth[0, :, 0],
        torch.full((4,), 2.0),
    )
    torch.testing.assert_close(
        projection.points[0, :, 0],
        expected.expand(4, -1),
    )
