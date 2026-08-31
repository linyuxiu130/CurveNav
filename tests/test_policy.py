from dataclasses import replace
import math
from types import MethodType

import torch

from curvenav import PolicyCondition, TrajectoryTarget, build_policy
from curvenav.config import (
    CurveNavConfig,
    DataConfig,
    DepthEncoderConfig,
    TrajectoryDecoderConfig,
    ConditionEncoderConfig,
    TrajectoryConfig,
)
from curvenav.training.optimizer import build_optimizer
from curvenav.models import TRAINING_LOSS_NAMES
from curvenav.models.blocks import ReusableConditionCrossAttention
from curvenav.encoders import CONFIGURATION_TOKEN_COUNT


def tiny_config() -> CurveNavConfig:
    return CurveNavConfig(
        data=DataConfig(observation_frames=4),
        trajectory=TrajectoryConfig(num_path_points=64),
        depth_encoder=DepthEncoderConfig(
            model_dim=32, frame_tokens_height=2, frame_tokens_width=2
        ),
        condition_encoder=ConditionEncoderConfig(
            model_dim=32, transformer_layers=1, transformer_heads=4
        ),
        trajectory_decoder=TrajectoryDecoderConfig(
            model_dim=32,
            transformer_layers=3,
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


def trajectory_target(values: torch.Tensor) -> TrajectoryTarget:
    return TrajectoryTarget(values)


def flow_interval_group(batch: int, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.arange(batch, device=device, dtype=torch.uint8).remainder(4)


def predict_stage_velocities(policy, state, start_time, end_time, encoded):
    return policy._predict_stage_velocities(
        state,
        start_time,
        end_time,
        encoded,
        policy.trajectory_decoder.project_condition_memory(encoded.tokens),
    )[0]


def test_policy_trains_every_module_and_returns_one_deterministic_trajectory() -> None:
    torch.manual_seed(0)
    policy = build_policy(tiny_config())
    inputs = condition(batch=8)
    curve_coordinates = 0.2 * torch.randn(8, policy.curve_codec.num_curve_tokens)
    target_values = policy.curve_codec.values_from_coordinates(curve_coordinates)
    losses = policy(
        inputs,
        trajectory_target(target_values),
        torch.randn_like(curve_coordinates),
        flow_interval_group(8),
    )
    for value in (
        losses.loss,
        losses.mean_flow_loss,
        losses.visible_clearance_loss,
    ):
        assert value.ndim == 0 and torch.isfinite(value)
    assert len(losses.logging_values()) == len(TRAINING_LOSS_NAMES)
    assert losses.loss > 0
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
    assert first.path.shape == (8, 64, 2)
    assert torch.equal(first.path, second.path)
    assert torch.isfinite(first.path).all()
    assert torch.equal(first.path[:, 0], torch.zeros(8, 2))


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
    assert first_encoded.tokens.shape[1] == 8 + CONFIGURATION_TOKEN_COUNT


def test_valid_causal_motion_changes_condition_but_current_pose_is_not_a_state_token() -> None:
    policy = build_policy(tiny_config()).eval()
    baseline = condition(batch=1)
    changed = PolicyCondition(
        depth=baseline.depth.clone(),
        point_goal=baseline.point_goal.clone(),
        observation_to_current=baseline.observation_to_current.clone(),
        observation_valid=baseline.observation_valid.clone(),
    )
    changed.observation_to_current[:, 1, :] = torch.tensor(
        [0.3, -0.2, math.sin(0.4), math.cos(0.4)]
    )
    with torch.no_grad():
        baseline_tokens = policy.encode_condition(baseline).tokens
        changed_tokens = policy.encode_condition(changed).tokens
    assert not torch.allclose(baseline_tokens, changed_tokens)

    motion = policy.condition_encoder.motion_encoder
    with torch.no_grad():
        baseline_state = motion(
            baseline.observation_to_current,
            baseline.observation_valid,
        )
        current_changed = baseline.observation_to_current.clone()
        current_changed[:, -1] = torch.tensor([10.0, -4.0, 1.0, 0.0])
        current_changed_state = motion(current_changed, baseline.observation_valid)
    torch.testing.assert_close(baseline_state, current_changed_state)


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


def test_eight_control_point_heading_contract_is_unique() -> None:
    invalid = replace(
        tiny_config(),
        trajectory=replace(
            tiny_config().trajectory,
            num_heading_control_points=7,
        ),
    )
    try:
        invalid.validate()
    except ValueError as error:
        assert "exactly eight" in str(error)
    else:
        raise AssertionError("an alternate curve token shape must be rejected")


def test_softplus_length_coordinates_are_positive_finite_and_invertible() -> None:
    codec = build_policy(tiny_config()).curve_codec
    coordinates = torch.zeros(4, codec.num_curve_tokens)
    coordinates[:, 0] = torch.tensor([-50.0, -1.0, 1.0, 10_000.0])

    values = codec.values_from_coordinates(coordinates)
    path, _ = codec.decode(coordinates)

    assert torch.all(values[:, 0] > 0)
    assert torch.isfinite(values).all()
    assert torch.isfinite(path).all()
    torch.testing.assert_close(
        codec.coordinates_from_values(values),
        coordinates,
        rtol=2e-5,
        atol=2e-5,
    )

    physical_values = torch.zeros(3, codec.num_curve_tokens)
    physical_values[:, 0] = torch.tensor([0.1315874, 2.938137, 3.599717])
    round_trip = codec.values_from_coordinates(
        codec.coordinates_from_values(physical_values)
    )
    torch.testing.assert_close(round_trip, physical_values, rtol=1e-6, atol=1e-6)


def test_typical_source_and_one_step_mean_flow_are_deterministic() -> None:
    policy = build_policy(tiny_config())
    curve_tokens = policy.curve_codec.num_curve_tokens
    assert policy.inference_source.shape == (1, curve_tokens)
    torch.testing.assert_close(
        torch.linalg.vector_norm(policy.inference_source),
        torch.tensor(math.sqrt(float(curve_tokens))),
    )
    assert torch.count_nonzero(policy.inference_source) == curve_tokens

    constant_average_velocity = torch.linspace(-0.2, 0.2, curve_tokens)

    def constant_field(self, state, start_time, end_time, encoded, projected_condition):
        average = constant_average_velocity.expand_as(state)[:, None].expand(-1, 3, -1)
        return average, torch.zeros_like(average)

    policy._predict_stage_velocities = MethodType(constant_field, policy)
    expected_state = policy.inference_source - constant_average_velocity
    expected_path, _ = policy.curve_codec.decode(expected_state)
    prediction = policy.sample(condition(batch=2))
    torch.testing.assert_close(prediction.path, expected_path.expand(2, -1, -1))


def test_training_intervals_cover_deployment_average_and_diagonal() -> None:
    policy = build_policy(tiny_config())

    class ZeroMeanVelocity(torch.nn.Module):
        curve_tokens = policy.curve_codec.num_curve_tokens
        planning_horizon_m = policy.planning_horizon_m

        def forward(
            self,
            state,
            start_time,
            end_time,
            encoded,
            projected_condition,
            curve_codec,
        ):
            velocity = state * 0.0 + (end_time - start_time)[:, None] * 0.0
            stages = velocity[:, None].expand(-1, 3, -1)
            return stages, stages

        def project_condition_memory(self, condition_tokens):
            return ()

    policy.trajectory_decoder = ZeroMeanVelocity()
    inputs = condition(batch=8)
    target_values = policy.curve_codec.values_from_coordinates(
        torch.linspace(-0.5, 0.5, 8 * policy.curve_codec.num_curve_tokens).reshape(
            8, policy.curve_codec.num_curve_tokens
        )
    )
    clean = policy.curve_codec.coordinates_from_values(target_values)
    torch.manual_seed(123)
    random_source = torch.randn_like(clean)
    losses = policy.training_loss(
        inputs,
        trajectory_target(target_values),
        random_source,
        flow_interval_group(8),
    )

    torch.manual_seed(456)
    start, end, deployment = policy._training_intervals(flow_interval_group(8), clean)
    assert torch.equal(
        deployment,
        torch.tensor([True, False, False, False, True, False, False, False]),
    )
    torch.testing.assert_close(end[deployment], torch.ones(2))
    torch.testing.assert_close(start[deployment], torch.zeros(2))
    diagonal = torch.tensor([False, False, True, True, False, False, True, True])
    torch.testing.assert_close(start[diagonal], end[diagonal])
    interior = ~deployment & ~diagonal
    assert torch.all((0.0 < start[interior]) & (start[interior] < end[interior]))
    assert torch.all(end[interior] < 1.0)
    flow_source = random_source.clone()
    flow_source[deployment] = policy.inference_source
    expected = (flow_source - clean).square().mean()
    torch.testing.assert_close(losses.mean_flow_loss, expected)


def test_improved_mean_flow_jvp_has_the_exact_sign_and_interval_tangent() -> None:
    policy = build_policy(tiny_config())

    class AnalyticMeanVelocity(torch.nn.Module):
        curve_tokens = policy.curve_codec.num_curve_tokens
        planning_horizon_m = policy.planning_horizon_m

        def forward(
            self,
            state,
            start_time,
            end_time,
            encoded,
            projected_condition,
            curve_codec,
        ):
            average = state + (end_time - start_time)[:, None]
            stage_offset = torch.tensor(
                [-0.5, 0.0, 0.5], device=state.device, dtype=state.dtype
            )[None, :, None]
            average = average[:, None] + stage_offset
            instantaneous = (
                2.0 * state[:, None]
                + (end_time - start_time)[:, None, None]
                + stage_offset
            )
            return average, instantaneous

        def project_condition_memory(self, condition_tokens):
            return ()

    policy.trajectory_decoder = AnalyticMeanVelocity()
    inputs = condition(batch=6)
    clean = torch.linspace(-0.5, 0.5, 6 * policy.curve_codec.num_curve_tokens).reshape(
        6, policy.curve_codec.num_curve_tokens
    )
    target_values = policy.curve_codec.values_from_coordinates(clean)
    torch.manual_seed(321)
    source = torch.randn_like(clean)
    torch.manual_seed(654)
    start, time, deployment = policy._training_intervals(flow_interval_group(6), clean)
    flow_source = source.clone()
    flow_source[deployment] = policy.inference_source
    state = (1.0 - time[:, None]) * clean + time[:, None] * flow_source
    conditional_velocity = flow_source - clean
    stage_offset = torch.tensor([-0.5, 0.0, 0.5])[None, :, None]
    instantaneous = (
        2.0 * state[:, None]
        + (time - start)[:, None, None]
        + stage_offset
    )
    average = state[:, None] + (time - start)[:, None, None] + stage_offset
    final_instantaneous = 2.0 * state + 0.5
    total_derivative = final_instantaneous[:, None] + 1.0
    reparameterized = average + (time - start)[:, None, None] * total_derivative
    expected = 0.5 * (
        (instantaneous - conditional_velocity[:, None]).square().mean()
        + (reparameterized - conditional_velocity[:, None]).square().mean()
    )

    torch.manual_seed(654)
    losses = policy.training_loss(
        inputs,
        trajectory_target(target_values),
        source,
        flow_interval_group(6),
    )
    torch.testing.assert_close(losses.mean_flow_loss, expected)


def test_path_geometry_reconstructs_data_endpoint_with_instantaneous_velocity() -> None:
    policy = build_policy(tiny_config())
    decoder = policy.trajectory_decoder
    state = torch.linspace(-0.7, 0.8, policy.curve_codec.num_curve_tokens)[None]
    instantaneous = torch.linspace(
        0.3, -0.2, policy.curve_codec.num_curve_tokens
    )[None]
    end_time = torch.tensor([0.65])
    captured = []

    def capture_geometry(module, inputs):
        captured.append(inputs[0].detach())

    handle = decoder.path_geometry_embedding.register_forward_pre_hook(capture_geometry)
    try:
        decoder._path_tokens(
            state,
            end_time,
            instantaneous,
            torch.zeros(1, 5, 64, 64),
            torch.zeros(1, 1, tiny_config().trajectory_decoder.model_dim),
            policy.curve_codec,
        )
    finally:
        handle.remove()

    expected_clean = state - end_time[:, None] * instantaneous
    expected_path, _ = policy.curve_codec.decode_path(expected_clean)
    expected_anchors = expected_path[:, decoder.path_indices]
    torch.testing.assert_close(
        captured[0][..., :2] * policy.planning_horizon_m,
        expected_anchors,
    )


def test_configuration_space_field_is_queried_continuously() -> None:
    from curvenav.models.safety import sample_configuration_field

    clearance = torch.ones(1, 9, 9)
    colliding = torch.tensor([[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]])
    field = torch.zeros(1, 5, 9, 9)
    field[:, 0] = clearance
    sampled = sample_configuration_field(field, colliding, 1.0)
    assert sampled.shape == (1, 3, 5)
    torch.testing.assert_close(sampled[..., 0], torch.ones(1, 3))


def test_observed_clearance_loss_moves_a_curve_toward_higher_clearance() -> None:
    from curvenav.models.safety import observed_clearance_loss

    axis = torch.linspace(-1.0, 1.0, 64)
    field = torch.zeros(1, 5, 64, 64)
    field[:, 0] = 0.05 * axis[None, None, :]
    field[:, 3] = 1.0
    path = torch.tensor([[[0.0, 0.0], [0.5, 0.0]]], requires_grad=True)

    loss = observed_clearance_loss(path, field, planning_horizon_m=1.0).sum()
    loss.backward()

    assert loss > 0.0
    assert path.grad is not None
    assert path.grad[..., 0].sum() < 0.0

    unobserved = field.clone()
    unobserved[:, 3] = 0.0
    path = path.detach().clone().requires_grad_()
    unobserved_loss = observed_clearance_loss(path, unobserved, 1.0).sum()
    unobserved_loss.backward()
    assert unobserved_loss.item() == 0.0
    torch.testing.assert_close(path.grad, torch.zeros_like(path.grad))


def test_conditioned_decoder_returns_one_smooth_metric_curve() -> None:
    policy = build_policy(tiny_config())
    inputs = condition(batch=2)
    encoded = policy.encode_condition(inputs)
    noisy = torch.randn(2, policy.curve_codec.num_curve_tokens)
    time = torch.tensor([0.2, 0.7])
    predicted_velocity = predict_stage_velocities(
        policy, noisy, torch.zeros_like(time), time, encoded
    )
    repeated = predict_stage_velocities(
        policy, noisy, torch.zeros_like(time), time, encoded
    )
    torch.testing.assert_close(predicted_velocity, repeated)
    assert predicted_velocity.shape == (2, 3, policy.curve_codec.num_curve_tokens)
    prediction = policy.sample(inputs)
    assert prediction.path.shape == (2, 64, 2)


def test_reused_condition_projection_is_exact_cross_attention() -> None:
    torch.manual_seed(17)
    reference = torch.nn.MultiheadAttention(32, 4, batch_first=True)
    candidate = ReusableConditionCrossAttention(32, 4, 0.0)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference_queries = [
        torch.randn(3, 8, 32, requires_grad=True) for _ in range(2)
    ]
    candidate_queries = [
        query.detach().clone().requires_grad_() for query in reference_queries
    ]
    reference_memory = torch.randn(3, 13, 32, requires_grad=True)
    candidate_memory = reference_memory.detach().clone().requires_grad_()
    expected = [
        reference(query, reference_memory, reference_memory, need_weights=False)[0]
        for query in reference_queries
    ]
    projected = candidate.project_condition(candidate_memory)
    actual = [candidate(query, projected) for query in candidate_queries]
    sum(output.square().mean() for output in expected).backward()
    sum(output.square().mean() for output in actual).backward()
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
    for left, right in zip(candidate_queries, reference_queries, strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    torch.testing.assert_close(candidate_memory.grad, reference_memory.grad)
    for name, parameter in candidate.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
        )


def test_decoder_has_one_internal_path_relative_refinement_route() -> None:
    policy = build_policy(tiny_config())
    decoder = policy.trajectory_decoder
    assert decoder.curve_tokens == policy.curve_codec.num_curve_tokens
    assert not hasattr(policy, "trajectory_flow")
    assert not hasattr(policy, "curve_proposal")
    assert not hasattr(policy, "candidate_bases")
    assert not hasattr(decoder, "metric_path_attention")
    assert decoder.path_geometry_embedding[0].in_features == 10
    assert decoder.layers_per_stage == 1
    assert hasattr(decoder, "average_velocity_readout")
    assert hasattr(decoder, "instantaneous_velocity_readout")


def test_observed_configuration_space_changes_condition_memory() -> None:
    encoder = build_policy(tiny_config()).condition_encoder.configuration_encoder
    empty = torch.zeros(2, 5, 64, 64)
    changed = empty.clone()
    changed[0, 0, 4:12, 48:56] = -0.2
    changed[0, 3:, 4:12, 48:56] = 1.0
    visual = torch.zeros(2, 4, 32)
    visual_points = torch.zeros(2, 4, 3)
    visual_valid = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        baseline = encoder(empty, visual, visual_points, visual_valid)
        encoded = encoder(changed, visual, visual_points, visual_valid)
    assert baseline.tokens.shape == (2, CONFIGURATION_TOKEN_COUNT, 32)
    assert baseline.measured_field.shape == (2, 5, 64, 64)
    torch.testing.assert_close(baseline.measured_field, empty)
    torch.testing.assert_close(encoded.measured_field, changed)
    assert not torch.allclose(encoded.tokens[0], baseline.tokens[0])
    torch.testing.assert_close(encoded.tokens[1], baseline.tokens[1])

    unobserved = empty.clone()
    unobserved[:, :3] = torch.randn_like(unobserved[:, :3])
    unobserved[:, 4] = 1.0
    with torch.no_grad():
        masked = encoder(unobserved, visual, visual_points, visual_valid)
    torch.testing.assert_close(masked.tokens, baseline.tokens)

    visual = torch.randn(2, 4, 32)
    visual_points = torch.tensor(
        [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]
    ).expand(2, -1, -1)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        ordered = encoder(empty, visual, visual_points, visual_valid)
        permuted = encoder(
            empty,
            visual[:, permutation],
            visual_points[:, permutation],
            visual_valid[:, permutation],
        )
    torch.testing.assert_close(permuted.tokens, ordered.tokens)


def test_obstacle_far_from_flow_source_changes_one_step_velocity() -> None:
    policy = build_policy(tiny_config()).eval()
    inputs = condition(batch=1)
    with torch.no_grad():
        observation = policy.depth_encoder(
            inputs.depth,
            inputs.observation_to_current,
            inputs.observation_valid,
        )
        baseline = policy.condition_encoder(
            observation,
            inputs.point_goal,
            inputs.observation_valid,
            inputs.observation_to_current,
        )
        changed_field = observation.configuration_field.clone()
        changed_field[:, 0, :8, 56:] = -0.2
        changed_field[:, 3:, :8, 56:] = 1.0
        changed = policy.condition_encoder(
            replace(observation, configuration_field=changed_field),
            inputs.point_goal,
            inputs.observation_valid,
            inputs.observation_to_current,
        )
        source = policy.inference_source.clone()
        zero = torch.zeros(1)
        velocity = predict_stage_velocities(policy, source, zero, zero + 1.0, baseline)
        changed_velocity = predict_stage_velocities(
            policy,
            source,
            zero,
            zero + 1.0,
            changed,
        )
    assert not torch.allclose(changed.tokens, baseline.tokens)
    assert not torch.allclose(changed_velocity, velocity)


def test_exact_path_field_changes_velocity_without_changing_context_tokens() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(condition(batch=1))
    changed_field = encoded.path_configuration_field.clone()
    changed_field[:, 0] = -0.2
    changed_field[:, 3:] = 1.0
    changed = replace(encoded, path_configuration_field=changed_field)
    source = policy.inference_source.clone()
    zero = torch.zeros(1)
    with torch.no_grad():
        baseline_velocity = predict_stage_velocities(
            policy, source, zero, zero + 1.0, encoded
        )
        changed_velocity = predict_stage_velocities(
            policy, source, zero, zero + 1.0, changed
        )
    torch.testing.assert_close(changed.tokens, encoded.tokens)
    assert not torch.allclose(changed_velocity, baseline_velocity)


def test_decoder_queries_measured_geometry() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(condition(batch=1))
    measured_field = encoded.path_configuration_field.clone()
    measured_field[:, 0] -= 0.5
    measured_field[:, 3] = 1.0
    source = policy.inference_source.clone()
    zero = torch.zeros(1)
    with torch.no_grad():
        baseline = predict_stage_velocities(
            policy, source, zero, zero + 1.0, encoded
        )
        measured = predict_stage_velocities(
            policy,
            source,
            zero,
            zero + 1.0,
            replace(encoded, path_configuration_field=measured_field),
        )

    assert not torch.allclose(measured, baseline)


def test_decoder_cannot_treat_unobserved_clearance_as_free_space() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(condition(batch=1))
    empty_field = torch.zeros_like(encoded.path_configuration_field)
    unobserved_field = empty_field.clone()
    unobserved_field[:, :3] = torch.randn_like(unobserved_field[:, :3])
    unobserved_field[:, 4] = 1.0
    source = policy.inference_source.clone()
    zero = torch.zeros(1)
    with torch.no_grad():
        baseline = predict_stage_velocities(
            policy,
            source,
            zero,
            zero + 1.0,
            replace(encoded, path_configuration_field=empty_field),
        )
        unobserved = predict_stage_velocities(
            policy,
            source,
            zero,
            zero + 1.0,
            replace(encoded, path_configuration_field=unobserved_field),
        )
    torch.testing.assert_close(unobserved, baseline)


def test_path_field_is_exactly_the_measured_configuration_space() -> None:
    policy = build_policy(tiny_config()).eval()
    inputs = condition(batch=2)
    with torch.no_grad():
        observation = policy.depth_encoder(
            inputs.depth,
            inputs.observation_to_current,
            inputs.observation_valid,
        )
        encoded = policy.condition_encoder(
            observation,
            inputs.point_goal,
            inputs.observation_valid,
            inputs.observation_to_current,
        )
    torch.testing.assert_close(
        encoded.path_configuration_field, observation.configuration_field.float()
    )


def test_point_goal_is_separate_from_scene_and_configuration_field() -> None:
    policy = build_policy(tiny_config()).eval()
    inputs = condition(batch=1)
    changed_inputs = replace(inputs, point_goal=-inputs.point_goal)

    with torch.no_grad():
        baseline = policy.encode_condition(inputs)
        changed = policy.encode_condition(changed_inputs)

    torch.testing.assert_close(
        changed.tokens[:, :-1],
        baseline.tokens[:, :-1],
    )
    torch.testing.assert_close(
        changed.path_configuration_field,
        baseline.path_configuration_field,
    )
    assert not torch.allclose(
        changed.tokens[:, -1:],
        baseline.tokens[:, -1:],
    )


def test_path_query_masks_probabilities_outside_without_erasing_raw_clearance() -> None:
    from curvenav.models.safety import sample_configuration_field

    field = torch.zeros(1, 5, 3, 3)
    field[:, 0] = 1.0
    field[:, 3:5] = 1.0
    path = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])

    sampled = sample_configuration_field(field, path, 1.0)

    torch.testing.assert_close(sampled[0, 0], field[0, :, 1, 1])
    assert sampled[0, 1, 0] == 1.0
    torch.testing.assert_close(sampled[0, 1, 3:], torch.zeros(2))


def test_heading_curve_coordinates_are_semantic_and_not_path_gram_values() -> None:
    codec = build_policy(tiny_config()).curve_codec
    coordinates = torch.randn(3, codec.num_curve_tokens)
    values = codec.values_from_coordinates(coordinates)
    assert torch.all(values[:, 0] > 0)
    torch.testing.assert_close(codec.coordinates_from_values(values), coordinates)
    assert not hasattr(codec, "path_orthonormalizer")
    assert not hasattr(codec, "planning_horizon_m")


def test_heading_spline_has_origin_forward_tangent_and_regular_arc_parameter() -> None:
    codec = build_policy(tiny_config()).curve_codec
    coordinates = torch.zeros(32, codec.num_curve_tokens)
    path, heading = codec.decode(coordinates)
    torch.testing.assert_close(heading[:, 0], torch.zeros_like(heading[:, 0]))
    torch.testing.assert_close(path[:, 0], torch.zeros_like(path[:, 0]))
    segment = path[:, 1:] - path[:, :-1]
    assert torch.all(torch.linalg.vector_norm(segment, dim=-1) > 0)
    assert codec.num_curve_tokens == 8


def test_metric_curve_values_round_trip_through_flow_coordinates() -> None:
    codec = build_policy(tiny_config()).curve_codec
    values = torch.tensor([[2.4, -0.2, -0.1, 0.0, 0.2, 0.4, 0.5, 0.6]])
    coordinates = codec.coordinates_from_values(values)
    torch.testing.assert_close(codec.values_from_coordinates(coordinates), values)


def test_spatial_history_and_causal_motion_tokens_have_fixed_contract() -> (
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
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert observation.tokens.shape == (1, 4, 32)
    assert projection.obstacle_points.shape == (1, 4, 4, 2)
    assert projection.obstacle_valid.shape == (1, 4, 4)
    assert projection.configuration_field.shape == (1, 5, 64, 64)
    assert projection.obstacle_valid.any()
    far = policy.depth_encoder.metric_projector(
        torch.ones_like(depth),
        identity_observation_transform(1),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert not far.obstacle_valid.any()
    assert not far.configuration_field[:, 4].any()

    encoded = policy.encode_condition(condition(batch=1))
    encoder = policy.condition_encoder
    assert encoder.motion_encoder.slot_embedding.shape == (1, 3, 32)
    assert encoded.tokens.shape == (
        1,
        4 + 3 + CONFIGURATION_TOKEN_COUNT + 1,
        32,
    )
    assert policy.trajectory_decoder.position_embedding.shape == (1, 8, 32)
    assert policy.trajectory_decoder.path_position_embedding.shape == (1, 32, 32)
    torch.testing.assert_close(
        policy.trajectory_decoder.path_progress.flatten(),
        policy.trajectory_decoder.path_indices.float() / 63.0,
    )
    assert len(encoder.context_blocks) == 1


def test_continuous_learned_token_embeddings_are_weight_decayed() -> None:
    policy = build_policy(tiny_config())
    optimizer = build_optimizer(policy, learning_rate=4e-4, weight_decay=1e-2)
    decay = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    learned_token_names = ("position_embedding", "slot_embedding")
    for name, parameter in policy.named_parameters():
        if name.endswith(learned_token_names):
            assert id(parameter) in decay, name


def test_mean_flow_decoder_depends_on_state_and_interval() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(condition(batch=2))
    decoder = policy.trajectory_decoder
    noisy = torch.zeros(2, policy.curve_codec.num_curve_tokens)
    with torch.no_grad():
        zero = torch.zeros(2)
        one = torch.ones(2)
        reference = predict_stage_velocities(policy, noisy, zero, zero, encoded)
        changed_curve = predict_stage_velocities(
            policy, noisy + 1.0, zero, zero, encoded
        )
        changed_time = predict_stage_velocities(policy, noisy, zero, one, encoded)

    assert not torch.allclose(changed_curve, reference)
    assert not torch.allclose(changed_time, reference)


def test_trainable_decoder_uses_bf16_but_returns_finite_curve_velocity() -> None:
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
    from curvenav.precision import cuda_precision

    with torch.no_grad(), torch.autocast(
        "cuda", dtype=cuda_precision("cuda").autocast_dtype
    ):
        output = predict_stage_velocities(
            policy,
            torch.randn(2, policy.curve_codec.num_curve_tokens, device="cuda"),
            torch.zeros(2, device="cuda"),
            torch.tensor([0.2, 0.7], device="cuda"),
            encoded,
        )
    handle.remove()

    assert observed_dtype == [cuda_precision("cuda").autocast_dtype]
    assert torch.isfinite(output).all()


def test_cuda_improved_mean_flow_has_finite_forward_and_gradients() -> None:
    if not torch.cuda.is_available():
        return
    policy = build_policy(CurveNavConfig()).cuda().train()
    inputs = condition(batch=4)
    inputs = PolicyCondition(
        depth=inputs.depth.cuda(),
        point_goal=inputs.point_goal.cuda(),
        observation_to_current=inputs.observation_to_current.cuda(),
        observation_valid=inputs.observation_valid.cuda(),
    )
    coordinates = 0.2 * torch.randn(
        4, policy.curve_codec.num_curve_tokens, device="cuda"
    )
    target = trajectory_target(
        policy.curve_codec.values_from_coordinates(coordinates)
    )
    target = TrajectoryTarget(target.curve_values.cuda())
    from curvenav.precision import cuda_precision

    with torch.autocast("cuda", dtype=cuda_precision("cuda").autocast_dtype):
        losses = policy(
            inputs,
            target,
            torch.randn_like(coordinates),
            flow_interval_group(4, device="cuda"),
        )
    losses.loss.backward()

    assert all(torch.isfinite(value) for value in losses.logging_values())
    gradients = [
        parameter.grad for parameter in policy.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_expert_projection_uses_the_production_curve_manifold() -> None:
    codec = build_policy(tiny_config()).curve_codec
    values = codec.values_from_coordinates(
        0.2 * torch.randn(2, codec.num_curve_tokens)
    )
    source, _ = codec.decode_values(values)
    projected_values, projected = codec.project_expert(source)
    assert projected_values.shape == values.shape
    projection_ade = torch.linalg.vector_norm(projected - source, dim=-1).mean()
    assert projection_ade < 2e-3
    torch.testing.assert_close(projected[:, 0], torch.zeros_like(projected[:, 0]))


def test_heading_basis_fixes_the_initial_direction_and_has_continuous_curvature() -> None:
    codec = build_policy(tiny_config()).curve_codec
    torch.testing.assert_close(
        codec.dense_basis[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    values = codec.values_from_coordinates(torch.zeros(2, codec.num_curve_tokens))
    path, heading, curvature = codec._decode_values(values)
    assert path.shape == (2, 64, 2)
    assert heading.shape == curvature.shape == (2, 64)
    assert torch.isfinite(curvature).all()


def test_metric_xyz_backprojection_uses_camera_height_and_observation_transform() -> None:
    policy = build_policy(tiny_config()).eval()
    depth = torch.full((1, 4, 1, 126, 224), 0.4)
    identity = identity_observation_transform(1)
    translated = identity.clone()
    translated[:, 0, 0] = -1.0
    with torch.no_grad():
        valid = torch.ones(1, 4, dtype=torch.bool)
        first = policy.depth_encoder.metric_projector(depth, identity, valid).points
        second = policy.depth_encoder.metric_projector(depth, translated, valid).points
    assert first.shape[-1] == 3
    assert policy.depth_encoder.metric_projector.camera_height_m == 0.62532
    torch.testing.assert_close(
        second[:, 0, :, 0],
        first[:, 0, :, 0] - 1.0,
    )
    torch.testing.assert_close(second[..., 2], first[..., 2])


def test_historical_pose_rotation_maps_points_into_the_current_body_frame() -> None:
    projector = build_policy(tiny_config()).depth_encoder.metric_projector
    center = 63 * 224 + 112
    depth_m = torch.full((1, 2, 1, 1), 2.0)
    pixel = torch.full((1, 2, 1, 1), center)
    transforms = torch.tensor(
        [[[1.0, 2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    points = projector._backproject(depth_m, pixel, 126, 224, transforms)
    current = points[0, 1, 0]
    historical = points[0, 0, 0]

    torch.testing.assert_close(historical[0], 1.0 - current[1])
    torch.testing.assert_close(historical[1], 2.0 + current[0])
    torch.testing.assert_close(historical[2], current[2])


def test_configuration_distance_transform_is_exact_for_an_obstacle_union() -> None:
    projector = build_policy(tiny_config()).depth_encoder.metric_projector
    first = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    second = torch.zeros_like(first)
    first[0, 0, 11, 17] = True
    second[0, 0, 45, 38] = True

    first_distance = projector._euclidean_distance_transform(first)
    second_distance = projector._euclidean_distance_transform(second)
    union_distance = projector._euclidean_distance_transform(first | second)

    torch.testing.assert_close(
        union_distance,
        torch.minimum(first_distance, second_distance),
    )
    assert union_distance[0, 11, 17] == 0.0
    assert union_distance[0, 45, 38] == 0.0


def test_body_obstacle_pooling_cannot_be_occluded_by_nearer_floor() -> None:
    policy = build_policy(tiny_config()).eval()
    depth = torch.ones(1, 4, 1, 126, 224)
    # These surfaces share one adaptive cell.  The 1.5 m lower pixel reaches
    # the ground, while the slightly farther 1.6 m patch is a vertical face.
    depth[:, -1, 0, 109, 100] = 1.5 / 5.0
    depth[:, -1, 0, 93:98, 98:103] = 1.6 / 5.0
    projection = policy.depth_encoder.metric_projector(
        depth,
        identity_observation_transform(1),
        torch.ones(1, 4, dtype=torch.bool),
    )
    valid = projection.obstacle_valid[:, -1]

    assert valid.sum() == 1
    torch.testing.assert_close(
        projection.depth[:, -1][valid],
        torch.tensor([1.6]),
    )
    known_risk = projection.configuration_field[:, 0] <= 0.10
    assert torch.all(projection.configuration_field[:, 3][known_risk] == 1.0)


def test_surface_below_traversable_height_is_not_a_body_obstacle() -> None:
    projector = build_policy(tiny_config()).depth_encoder.metric_projector
    row = torch.arange(126, dtype=torch.float32)
    pitch = math.radians(10.0)
    denominator = (
        math.cos(pitch) * (row - 63.0) / 166.80851063829786
        + math.sin(pitch)
    )
    depth_m = torch.where(
        denominator > 0,
        (0.62532 - 0.005) / denominator,
        torch.tensor(5.0),
    ).clamp_max(5.0)
    depth = depth_m[None, None, None, :, None].expand(1, 4, 1, 126, 224) / 5.0

    projection = projector(
        depth,
        identity_observation_transform(1),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert not projection.obstacle_valid.any()


def test_horizontal_surface_is_not_mislabeled_by_height_alone() -> None:
    projector = build_policy(tiny_config()).depth_encoder.metric_projector
    row = torch.arange(126, dtype=torch.float32)
    pitch = math.radians(10.0)
    denominator = (
        math.cos(pitch) * (row - 63.0) / 166.80851063829786
        + math.sin(pitch)
    )
    depth_m = torch.where(
        denominator > 0,
        (0.62532 - 0.05) / denominator,
        torch.tensor(5.0),
    ).clamp_max(5.0)
    depth = depth_m[None, None, None, :, None].expand(1, 4, 1, 126, 224) / 5.0

    projection = projector(
        depth,
        identity_observation_transform(1),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert not projection.obstacle_valid.any()


def test_surface_normals_separate_traversable_slope_from_steep_obstacle() -> None:
    projector = build_policy(tiny_config()).depth_encoder.metric_projector
    row, column = torch.meshgrid(
        torch.arange(8, dtype=torch.float32),
        torch.arange(12, dtype=torch.float32),
        indexing="ij",
    )
    x = column * 0.05
    y = row * 0.05
    valid = torch.ones(1, 8, 12, dtype=torch.bool)

    gentle = torch.stack(
        (x, y, math.tan(math.radians(30.0)) * x),
        dim=-1,
    ).unsqueeze(0)
    steep = torch.stack(
        (x, y, math.tan(math.radians(60.0)) * x),
        dim=-1,
    ).unsqueeze(0)

    assert projector._traversable_surface(gentle, valid).all()
    assert not projector._traversable_surface(steep, valid).any()


def test_nearest_depth_is_backprojected_with_its_own_pixel_ray() -> None:
    projector = build_policy(tiny_config()).depth_encoder.metric_projector
    depth = torch.ones(1, 4, 1, 126, 224)
    depth[:, :, :, 10, 20] = 0.4
    projection = projector(
        depth,
        identity_observation_transform(1),
        torch.ones(1, 4, dtype=torch.bool),
    )
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
