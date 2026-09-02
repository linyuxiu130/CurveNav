"""Focused mathematical and architecture tests for the one CurveNav policy."""

from dataclasses import replace

import pytest
import torch

from curvenav import PolicyCondition, TrajectoryTarget, build_policy
from curvenav.config import (
    ConditionEncoderConfig,
    CurveNavConfig,
    DataConfig,
    DepthEncoderConfig,
    TrajectoryDecoderConfig,
    TrajectoryConfig,
)
from curvenav.models import TRAINING_LOSS_NAMES
from curvenav.models.safety import observed_clearance_loss
from curvenav.encoders.configuration import observed_configuration_features
from curvenav.models.blocks import (
    ConditionalTrajectoryBlock,
    ReusableConditionCrossAttention,
)
from curvenav.precision import cuda_precision
from curvenav.trajectory import metric_goal_reference
from curvenav.training.optimizer import build_optimizer
from curvenav.training.runtime import (
    compile_static_training_functions,
    configure_cuda_training_backend,
)


def tiny_config() -> CurveNavConfig:
    return CurveNavConfig(
        data=DataConfig(observation_frames=4),
        trajectory=TrajectoryConfig(),
        depth_encoder=DepthEncoderConfig(
            model_dim=32, frame_tokens_height=2, frame_tokens_width=2
        ),
        condition_encoder=ConditionEncoderConfig(model_dim=32),
        trajectory_decoder=TrajectoryDecoderConfig(
            model_dim=32, transformer_layers=4, transformer_heads=4
        ),
    )


def identity_transform(batch: int) -> torch.Tensor:
    value = torch.zeros(batch, 4, 4)
    value[..., 3] = 1
    return value


def make_condition(batch: int = 2) -> PolicyCondition:
    return PolicyCondition(
        depth=torch.rand(batch, 4, 1, 126, 224),
        point_goal=torch.stack(
            (torch.full((batch,), 3.0), torch.linspace(-1.0, 1.0, batch)),
            dim=-1,
        ),
        observation_to_current=identity_transform(batch),
        observation_valid=torch.ones(batch, 4, dtype=torch.bool),
    )


def test_cartesian_bspline_is_smooth_origin_anchored_and_differentiable() -> None:
    codec = build_policy(tiny_config()).curve_codec
    coordinates = torch.randn(3, 14, requires_grad=True)
    path, heading = codec.decode(coordinates)
    assert path.shape == (3, 64, 2)
    assert heading.shape == (3, 64)
    torch.testing.assert_close(path[:, 0], torch.zeros(3, 2))
    assert torch.isfinite(path).all() and torch.isfinite(heading).all()
    path.square().mean().backward()
    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()


def test_expert_projection_recovers_a_curve_in_the_same_spline_family() -> None:
    codec = build_policy(tiny_config()).curve_codec
    values = codec.values_from_coordinates(torch.randn(4, 14))
    path, _ = codec.decode_values(values)
    recovered, recovered_path = codec.project_expert(path)
    torch.testing.assert_close(recovered, values, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(recovered_path, path, atol=2e-5, rtol=2e-5)


def test_control_increment_coordinates_are_exactly_invertible() -> None:
    codec = build_policy(tiny_config()).curve_codec
    coordinates = torch.randn(8, 14)
    values = codec.values_from_coordinates(coordinates)
    recovered = codec.coordinates_from_values(values)
    torch.testing.assert_close(recovered, coordinates, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        codec.control_positions_from_coordinates(coordinates),
        values.reshape(8, 7, 2),
    )


def test_greville_goal_reference_is_an_exact_straight_bspline() -> None:
    codec = build_policy(tiny_config()).curve_codec
    goals = torch.tensor([[2.0, 1.0], [10.0, 0.0], [0.0, 0.0]])
    reference = metric_goal_reference(goals, 3.6)
    path, _ = codec.decode_values(reference.flatten(1))
    endpoint = reference[:, -1]
    expected = (
        torch.linspace(0.0, 1.0, codec.num_path_points)[None, :, None]
        * endpoint[:, None]
    )
    torch.testing.assert_close(path, expected, atol=5e-6, rtol=5e-6)


def test_clamped_bspline_tangent_is_the_control_increment_operator() -> None:
    codec = build_policy(tiny_config()).curve_codec
    values = codec.values_from_coordinates(torch.randn(4, 14))
    controls = torch.cat((torch.zeros(4, 1, 2), values.reshape(4, 7, 2)), dim=1)
    tangent = torch.einsum("pc,bcd->bpd", codec.first_basis, controls)
    knot_span = 1.0 / (codec.num_control_points - codec.degree)
    scale = codec.degree / knot_span
    torch.testing.assert_close(tangent[:, 0], scale * (controls[:, 1] - controls[:, 0]))
    torch.testing.assert_close(
        tangent[:, -1], scale * (controls[:, -1] - controls[:, -2])
    )


def test_pointgoal_never_rewrites_scene_memory() -> None:
    torch.manual_seed(0)
    policy = build_policy(tiny_config()).eval()
    first = make_condition(1)
    second = PolicyCondition(
        depth=first.depth,
        point_goal=torch.tensor([[-2.0, 3.0]]),
        observation_to_current=first.observation_to_current,
        observation_valid=first.observation_valid,
    )
    first_encoded = policy.encode_condition(first)
    second_encoded = policy.encode_condition(second)
    torch.testing.assert_close(first_encoded.tokens, second_encoded.tokens)
    assert not torch.equal(
        first_encoded.goal_reference, second_encoded.goal_reference
    )
    assert first_encoded.tokens.shape[1] == 16 * 16 + 3
    torch.testing.assert_close(first_encoded.token_valid, second_encoded.token_valid)


def test_pointgoal_reference_is_metric_local_and_does_not_constrain_output() -> None:
    policy = build_policy(tiny_config()).eval()
    encoder = policy.condition_encoder
    goals = torch.tensor(
        [[1.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [0.0, 0.0]]
    )
    reference = encoder._goal_reference(goals)

    assert torch.isfinite(reference).all()
    torch.testing.assert_close(
        reference[0, -1],
        goals[0],
    )
    torch.testing.assert_close(reference[1, -1], torch.tensor([3.6, 0.0]))
    torch.testing.assert_close(reference[2, -1], torch.tensor([0.0, 3.6]))
    torch.testing.assert_close(reference[3], torch.zeros(7, 2))
    # The reference is conditioning geometry, not a codec constraint.
    decoded, _ = policy.curve_codec.decode(torch.full((4, 14), 10.0))
    assert torch.linalg.vector_norm(decoded[:, -1], dim=-1).max() > 3.6


def test_trajectory_block_promotes_reusable_condition_to_query_dtype() -> None:
    block = ConditionalTrajectoryBlock(32, 4, 0).eval()
    trajectory = torch.randn(2, 7, 32)
    encoded = build_policy(tiny_config()).encode_condition(make_condition(2))
    projected = block.project_condition(encoded)
    key, value, *geometry = projected
    pair_geometry = torch.randn(2, 7, encoded.tokens.shape[1], 7)
    output = block(
        trajectory,
        (key.bfloat16(), value.bfloat16(), *geometry),
        pair_geometry,
    )
    assert output.dtype == trajectory.dtype
    assert output.shape == trajectory.shape and torch.isfinite(output).all()


def test_pointgoal_enters_decoder_only_as_metric_reference_geometry() -> None:
    policy = build_policy(tiny_config()).eval()
    names = tuple(name for name, _ in policy.named_parameters())
    assert not any(
        word in name
        for name in names
        for word in ("intent", "goal_projection", "reference_embedding")
    )
    first = make_condition(1)
    second = replace(first, point_goal=torch.tensor([[-2.0, 3.0]]))
    first_encoded = policy.encode_condition(first)
    second_encoded = policy.encode_condition(second)
    second_with_first_reference = replace(
        second_encoded,
        goal_reference=first_encoded.goal_reference,
    )
    state = torch.randn(1, 14)
    start = torch.zeros(1)
    end = torch.ones(1)
    first_velocity = policy._predict_stage_velocities(
        state,
        start,
        end,
        first_encoded,
        policy.trajectory_decoder.project_condition_memory(first_encoded),
    )[0]
    second_velocity = policy._predict_stage_velocities(
        state,
        start,
        end,
        second_with_first_reference,
        policy.trajectory_decoder.project_condition_memory(
            second_with_first_reference
        ),
    )[0]
    torch.testing.assert_close(first_velocity, second_velocity)


def test_depth_and_pointgoal_both_condition_the_trajectory() -> None:
    torch.manual_seed(1)
    policy = build_policy(tiny_config()).eval()
    baseline = make_condition(1)
    changed_depth = PolicyCondition(
        depth=torch.flip(baseline.depth, dims=(-1,)),
        point_goal=baseline.point_goal,
        observation_to_current=baseline.observation_to_current,
        observation_valid=baseline.observation_valid,
    )
    changed_goal = PolicyCondition(
        depth=baseline.depth,
        point_goal=torch.tensor([[2.0, -2.0]]),
        observation_to_current=baseline.observation_to_current,
        observation_valid=baseline.observation_valid,
    )
    base_path = policy.sample(baseline).path
    assert not torch.equal(base_path, policy.sample(changed_depth).path)
    assert not torch.equal(base_path, policy.sample(changed_goal).path)


def test_every_valid_history_frame_has_a_learned_visual_effect() -> None:
    torch.manual_seed(2)
    policy = build_policy(tiny_config()).eval()
    baseline = make_condition(1)
    base_tokens = policy.encode_condition(baseline).tokens
    for frame in range(4):
        depth = baseline.depth.clone()
        depth[:, frame] = torch.flip(depth[:, frame], dims=(-1,))
        changed = PolicyCondition(
            depth=depth,
            point_goal=baseline.point_goal,
            observation_to_current=baseline.observation_to_current,
            observation_valid=baseline.observation_valid,
        )
        assert not torch.equal(base_tokens, policy.encode_condition(changed).tokens)


def test_max_range_depth_is_valid_free_space_evidence() -> None:
    policy = build_policy(tiny_config()).eval()
    inputs = make_condition(1)
    inputs.depth.fill_(1.0)
    depth = policy.depth_encoder(
        inputs.depth,
        inputs.observation_to_current,
        inputs.observation_valid,
    )
    assert depth.token_valid.all()
    assert depth.configuration_field[:, 3].any()
    encoded = policy.encode_condition(inputs)
    assert torch.isfinite(encoded.tokens).all()


def test_unknown_configuration_values_cannot_enter_learned_geometry() -> None:
    field = torch.randn(2, 5, 64, 64)
    field[:, 3].zero_()
    canonical = observed_configuration_features(field)
    assert torch.count_nonzero(canonical) == 0


def test_invalid_history_pixels_and_poses_cannot_enter_the_condition() -> None:
    policy = build_policy(tiny_config()).eval()
    first = make_condition(1)
    first.observation_valid[:, :2] = False
    depth = first.depth.clone()
    transform = first.observation_to_current.clone()
    depth[:, :2] = torch.rand_like(depth[:, :2]) * 100
    transform[:, :2] = torch.rand_like(transform[:, :2]) * 100
    second = PolicyCondition(
        depth=depth,
        point_goal=first.point_goal,
        observation_to_current=transform,
        observation_valid=first.observation_valid,
    )
    torch.testing.assert_close(
        policy.encode_condition(first).tokens,
        policy.encode_condition(second).tokens,
    )


def test_training_objective_trains_every_module() -> None:
    torch.manual_seed(3)
    policy = build_policy(tiny_config()).train()
    batch = 8
    coordinates = torch.randn(batch, 14) * 0.2
    condition = make_condition(batch)
    losses = policy(
        condition,
        TrajectoryTarget(
            policy.curve_codec.values_from_coordinates(coordinates)
        ),
        torch.randn_like(coordinates),
        torch.arange(batch, dtype=torch.uint8).remainder(4),
    )
    assert TRAINING_LOSS_NAMES == (
        "loss",
        "mean_flow_loss",
        "visible_clearance_loss",
    )
    assert len(losses.logging_values()) == 3
    torch.testing.assert_close(
        losses.loss,
        losses.mean_flow_loss + losses.visible_clearance_loss,
    )
    assert losses.visible_clearance_loss >= 0
    assert losses.loss.ndim == 0 and torch.isfinite(losses.loss)
    losses.loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_visible_clearance_risk_uses_only_strict_observed_support() -> None:
    x = torch.linspace(0.0, 1.0, 64)
    path = torch.stack((x, torch.zeros_like(x)), dim=-1)[None]
    field = torch.zeros(1, 5, 64, 64)
    field[:, 0] = -0.05
    field[:, 3] = 1.0
    observed_risk = observed_clearance_loss(path, field, 3.6)
    assert observed_risk.item() > 0
    field[:, 3] = 0.0
    torch.testing.assert_close(
        observed_clearance_loss(path, field, 3.6),
        torch.zeros(1),
    )


def test_visible_clearance_risk_pushes_a_curve_up_the_clearance_gradient() -> None:
    x = torch.linspace(0.0, 1.0, 64)
    path = torch.stack((x, torch.full_like(x, 0.02)), dim=-1)[None]
    path.requires_grad_()
    axis = torch.linspace(-3.6, 3.6, 64)
    field = torch.zeros(1, 5, 64, 64)
    field[:, 0] = axis[:, None]
    field[:, 2] = 1.0
    field[:, 3] = 1.0
    loss = observed_clearance_loss(path, field, 3.6).sum()
    loss.backward()
    assert loss > 0
    assert path.grad is not None
    assert path.grad[..., 1].sum() < 0


def test_visible_clearance_risk_has_no_uniform_shortening_gradient() -> None:
    x = torch.linspace(0.0, 1.0, 64)
    base = torch.stack((x, torch.full_like(x, 0.02)), dim=-1)[None]
    scale = torch.ones((), requires_grad=True)
    axis = torch.linspace(-3.6, 3.6, 64)
    field = torch.zeros(1, 5, 64, 64)
    field[:, 0] = axis[:, None]
    field[:, 3] = 1.0
    observed_clearance_loss(scale * base, field, 3.6).sum().backward()
    assert scale.grad is not None
    torch.testing.assert_close(scale.grad, torch.zeros_like(scale.grad), atol=2e-5, rtol=0)


def test_deployment_boundary_uses_the_exact_fixed_source() -> None:
    policy = build_policy(tiny_config()).eval()
    clean = torch.randn(4, 14)
    source = torch.randn_like(clean)
    start, end, deployment = policy._training_intervals(
        torch.arange(4, dtype=torch.uint8), clean
    )
    assert deployment.tolist() == [True, False, False, False]
    assert start[0] == 0 and end[0] == 1
    flow_source = source.clone()
    flow_source[deployment] = policy.inference_source
    torch.testing.assert_close(flow_source[0], policy.inference_source[0])


def test_conditional_velocity_recovers_the_clean_flow_endpoint() -> None:
    clean = torch.randn(9, 14)
    source = torch.randn_like(clean)
    end_time = torch.rand(9)
    velocity = source - clean
    state = (1.0 - end_time[:, None]) * clean + end_time[:, None] * source
    torch.testing.assert_close(state - end_time[:, None] * velocity, clean)


def test_average_field_is_conditioned_on_interval_start() -> None:
    torch.manual_seed(8)
    policy = build_policy(tiny_config()).eval()
    condition = policy.encode_condition(make_condition(2))
    projected = policy.trajectory_decoder.project_condition_memory(condition)
    state = torch.randn(2, 14)
    end = torch.full((2,), 0.7)
    first_average = policy._predict_stage_velocities(
        state, torch.zeros(2), end, condition, projected
    )[0]
    second_average = policy._predict_stage_velocities(
        state, torch.full((2,), 0.4), end, condition, projected
    )[0]
    assert not torch.equal(first_average, second_average)


def test_sampling_is_one_call_with_reference_and_proposal_geometry_queries() -> None:
    policy = build_policy(tiny_config()).eval()
    inputs = make_condition(2)
    calls = []
    geometry_inputs = []

    def record_call(module, arguments):
        calls.append((arguments[1].detach().clone(), arguments[2].detach().clone()))

    def record_geometry(module, arguments):
        geometry_inputs.append(arguments[0].detach().clone())

    handle = policy.trajectory_decoder.register_forward_pre_hook(record_call)
    geometry_handle = policy.trajectory_decoder.path_geometry_embedding.register_forward_pre_hook(
        record_geometry
    )
    first = policy.sample(inputs).path
    handle.remove()
    geometry_handle.remove()
    torch.manual_seed(999)
    second = policy.sample(inputs).path
    assert torch.equal(first, second)
    assert first.shape == (2, 64, 2)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.zeros_like(calls[0][0]))
    torch.testing.assert_close(calls[0][1], torch.ones_like(calls[0][1]))
    assert len(geometry_inputs) == 2
    assert all(value.shape == (2, 7, 7) for value in geometry_inputs)


def test_goal_reference_provides_distinct_metric_retrieval_anchors() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(make_condition(1))
    geometry = policy.trajectory_decoder._path_relative_geometry(
        encoded.goal_reference,
        encoded,
    )
    assert not torch.equal(geometry[:, 0], geometry[:, -1])


def test_goal_geometry_uses_one_terminal_goal_not_a_straight_template() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(make_condition(1))
    candidate = encoded.goal_reference.clone()
    geometry = policy.trajectory_decoder._goal_geometry(candidate, encoded)
    expected_delta = encoded.goal_reference[:, -1:, :] - candidate
    torch.testing.assert_close(
        geometry[..., 2:4] * policy.planning_horizon_m,
        expected_delta,
    )
    assert torch.count_nonzero(geometry[:, :-1, 4]) > 0
    torch.testing.assert_close(geometry[:, -1, 4], torch.zeros(1))
    changed_reference = encoded.goal_reference.clone()
    changed_reference[:, :-1] += torch.randn_like(changed_reference[:, :-1])
    changed = replace(encoded, goal_reference=changed_reference)
    torch.testing.assert_close(
        policy.trajectory_decoder._goal_geometry(candidate, changed),
        geometry,
    )


def test_reusable_cross_attention_matches_projected_call() -> None:
    torch.manual_seed(4)
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    policy = build_policy(tiny_config())
    encoded = policy.encode_condition(make_condition(2))
    projected = layer.project_condition(encoded)
    geometry = policy.trajectory_decoder._path_relative_geometry(
        encoded.goal_reference, encoded
    )
    output = layer(query, projected, geometry)
    assert output.shape == query.shape and torch.isfinite(output).all()


def test_reusable_cross_attention_excludes_invalid_history_tokens() -> None:
    torch.manual_seed(5)
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    policy = build_policy(tiny_config())
    encoded = policy.encode_condition(make_condition(2))
    valid = encoded.token_valid.clone()
    valid[:, :5] = False
    changed = encoded.tokens.clone()
    changed[:, :5] = torch.randn_like(changed[:, :5]) * 1_000
    first_condition = replace(encoded, token_valid=valid)
    second_condition = replace(encoded, tokens=changed, token_valid=valid)
    geometry = policy.trajectory_decoder._path_relative_geometry(
        encoded.goal_reference, encoded
    )
    first = layer(query, layer.project_condition(first_condition), geometry)
    second = layer(query, layer.project_condition(second_condition), geometry)
    torch.testing.assert_close(first, second)


def test_reusable_cross_attention_promotes_cached_memory_to_query_dtype() -> None:
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    policy = build_policy(tiny_config())
    encoded = policy.encode_condition(make_condition(2))
    key, value, *geometry = layer.project_condition(encoded)
    pair_geometry = policy.trajectory_decoder._path_relative_geometry(
        encoded.goal_reference, encoded
    )
    output = layer(
        query,
        (key.bfloat16(), value.bfloat16(), *geometry),
        pair_geometry,
    )
    assert output.dtype == query.dtype
    assert output.shape == query.shape and torch.isfinite(output).all()


def test_path_relative_attention_uses_physical_query_positions() -> None:
    torch.manual_seed(6)
    policy = build_policy(tiny_config())
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    encoded = policy.encode_condition(make_condition(2))
    projected = layer.project_condition(encoded)
    first_position = torch.zeros(2, 7, 2)
    first_geometry = policy.trajectory_decoder._path_relative_geometry(
        first_position, encoded
    )
    first = layer(query, projected, first_geometry)
    shifted = torch.zeros(2, 7, 2)
    shifted[..., 1] = 1.0
    second_geometry = policy.trajectory_decoder._path_relative_geometry(
        shifted, encoded
    )
    second = layer(query, projected, second_geometry)
    assert not torch.equal(first, second)


def test_flow_field_reads_observed_geometry_along_its_candidate_curve() -> None:
    torch.manual_seed(7)
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(make_condition(1))
    state = torch.randn(1, 14)
    start = torch.zeros(1)
    end = torch.ones(1)
    projected = policy.trajectory_decoder.project_condition_memory(encoded)
    baseline = policy._predict_stage_velocities(
        state, start, end, encoded, projected
    )[0]
    changed_field = encoded.configuration_field.clone()
    changed_field[:, 0].fill_(-0.2)
    changed_field[:, 1].fill_(1.0)
    changed_field[:, 2].zero_()
    changed_field[:, 3:].fill_(1.0)
    changed = replace(encoded, configuration_field=changed_field)
    intervened = policy._predict_stage_velocities(
        state, start, end, changed, projected
    )[0]
    assert not torch.equal(baseline, intervened)


def test_deployment_sample_matches_the_final_path_from_proposal_diagnostics() -> None:
    torch.manual_seed(19)
    policy = build_policy(tiny_config()).eval()
    condition = make_condition(2)
    final_coordinates, proposal_coordinates = policy._deployment_transport(condition)
    expected_path, _ = policy.curve_codec.decode(final_coordinates)
    prediction = policy.sample(condition)
    proposal, _ = policy.curve_codec.decode(prediction.proposal_coordinates)
    assert prediction.path.shape == proposal.shape == (2, 64, 2)
    torch.testing.assert_close(prediction.path, expected_path)
    torch.testing.assert_close(
        prediction.proposal_coordinates,
        proposal_coordinates,
    )
    assert torch.isfinite(prediction.path).all()
    assert torch.isfinite(proposal).all()


def test_goal_does_not_relocate_candidate_relative_scene_queries() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(make_condition(1))
    candidate_controls = policy.curve_codec.control_positions_from_coordinates(
        torch.randn(1, 14)
    )
    baseline = policy.trajectory_decoder._path_relative_geometry(
        candidate_controls, encoded
    )
    changed = replace(encoded, goal_reference=-encoded.goal_reference)
    torch.testing.assert_close(
        baseline,
        policy.trajectory_decoder._path_relative_geometry(
            candidate_controls, changed
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_cuda_training_graph_and_deployment_sample_are_finite() -> None:
    device = torch.device("cuda")
    precision = cuda_precision(device)
    configure_cuda_training_backend()
    policy = build_policy(tiny_config()).to(device).train()
    compile_static_training_functions(policy)
    optimizer = build_optimizer(policy, learning_rate=1e-4, weight_decay=1e-2)
    inputs = make_condition(4)
    inputs = PolicyCondition(
        depth=inputs.depth.to(device),
        point_goal=inputs.point_goal.to(device),
        observation_to_current=inputs.observation_to_current.to(device),
        observation_valid=inputs.observation_valid.to(device),
    )
    coordinates = torch.randn(4, 14, device=device) * 0.1
    target = TrajectoryTarget(
        policy.curve_codec.values_from_coordinates(coordinates)
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=precision.autocast_dtype):
        loss = policy(
            inputs,
            target,
            torch.randn_like(coordinates),
            torch.arange(4, device=device, dtype=torch.uint8),
        ).loss
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
    policy.eval()
    with torch.autocast(device_type="cuda", dtype=precision.autocast_dtype):
        path = policy.sample(inputs).path
    assert path.shape == (4, 64, 2) and torch.isfinite(path).all()
