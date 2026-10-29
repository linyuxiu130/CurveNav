"""Focused mathematical and architecture tests for the one CurveNav policy."""

from dataclasses import replace

import torch

from curvenav import build_policy
from curvenav.config import (
    ConditionEncoderConfig,
    CurveNavConfig,
    DataConfig,
    DepthEncoderConfig,
    TrajectoryDecoderConfig,
    TrajectoryConfig,
)
from curvenav.models.policy import generate_candidate_bank
from curvenav.encoders.configuration import metric_grid
from curvenav.types import ConfigurationFeatures
from curvenav.models.blocks import (
    ConditionalTrajectoryBlock,
    ReusableConditionCrossAttention,
)
from curvenav.trajectory import local_terminal_goal


def tiny_config() -> CurveNavConfig:
    return CurveNavConfig(
        data=DataConfig(observation_frames=6),
        trajectory=TrajectoryConfig(),
        depth_encoder=DepthEncoderConfig(
            model_dim=32, frame_tokens_height=2, frame_tokens_width=2
        ),
        condition_encoder=ConditionEncoderConfig(model_dim=32),
        trajectory_decoder=TrajectoryDecoderConfig(
            model_dim=32, transformer_layers=4, transformer_heads=4
        ),
    )


def make_encoded_condition(policy, batch=2):
    position, _ = metric_grid(32, 16, policy.planning_horizon_m)
    configuration = ConfigurationFeatures(torch.randn(batch, 256, 32), position.expand(batch, -1, -1), torch.ones(batch, 256))
    field = torch.randn(batch, 6, 64, 64)
    field[:, 3:] = field[:, 3:].sigmoid()
    return policy.condition_encoder.encode_configuration(configuration, field, torch.tensor([[3., 1.]]).expand(batch, -1),
        torch.ones(batch, 6, dtype=torch.bool), torch.eye(4).expand(batch, 6, 4, 4), torch.linspace(5., 0., 6).expand(batch, -1))


def test_no_goal_mask_removes_goal_influence_without_changing_scene():
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 2)
    memory = policy.trajectory_decoder.project_condition_memory(encoded)
    absent = replace(encoded, goal_present=torch.zeros(2, 1))
    changed = replace(absent, terminal_goal=-10 * encoded.terminal_goal)
    state, time = torch.randn(2, 14), torch.tensor([0.5, 1.])
    first = policy._predict_velocity(state, time, absent, memory)
    second = policy._predict_velocity(state, time, changed, memory)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    at_origin = replace(encoded, terminal_goal=torch.zeros_like(encoded.terminal_goal))
    present = policy._predict_velocity(state, time, at_origin, memory)
    missing = policy._predict_velocity(state, time, replace(at_origin, goal_present=torch.zeros(2, 1)), memory)
    assert not torch.equal(present, missing)


def test_entire_nogoal_flow_is_independent_of_goal_and_keeps_fixed_origin():
    torch.manual_seed(2026)
    policy = build_policy(tiny_config()).eval()
    encoded = replace(make_encoded_condition(policy, 2), goal_present=torch.zeros(2, 1))
    first = generate_candidate_bank(policy, encoded)["paths"]
    second = generate_candidate_bank(policy, replace(encoded, terminal_goal=-7 * encoded.terminal_goal))["paths"]
    assert first.shape == (2, 16, 64, 2) and torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first[:, :, 0], torch.zeros(2, 16, 2), rtol=0, atol=0)
    # Check independent noise is actually consumed; not a trained diversity claim.
    assert first.flatten(2).std(dim=1).sum() > 0


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


def test_expert_projection_preserves_forward_tangent_and_terminal_goal() -> None:
    codec = build_policy(tiny_config()).curve_codec
    t = torch.linspace(0, 1, 64)
    # A valid forward U-turn may finish behind the initial pose.
    angle = t * 4.0
    path = torch.stack((torch.sin(angle), 1 - torch.cos(angle)), dim=-1)[None]
    values, recovered = codec.project_expert(path)
    _, heading = codec.decode_values(values)
    assert heading[0, 0] == 0
    assert values[0, 0] > 0 and path[0, -1, 0] < 0
    torch.testing.assert_close(recovered[:, -1], path[:, -1])
    torch.testing.assert_close(recovered[:, 0], torch.zeros(1, 2))


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


def test_increment_geometry_matches_the_physical_curve_jacobian() -> None:
    policy = build_policy(tiny_config())
    codec = policy.curve_codec
    state = torch.randn(1, 14)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: codec.decode(value)[0][0, :, 0], state
    )[:, 0, ::2]
    # Perturbing increment i translates every control j >= i.
    expected = torch.stack(
        [codec.basis[:, i + 1:].sum(1) for i in range(7)], dim=1
    )
    torch.testing.assert_close(
        jacobian, expected * codec.control_increment_std_xy_m[::2]
    )
    torch.testing.assert_close(codec.increment_basis, expected)
    torch.testing.assert_close(
        policy.trajectory_decoder.path_to_increment_weight,
        (expected / expected.sum(0)).T,
    )
    torch.testing.assert_close(expected[-1], torch.ones(7))


def test_euler_sampling_refreshes_curve_geometry_but_caches_scene_kv(monkeypatch):
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 2)
    states, times, geometry, scene = [], [], [], []
    original = policy._predict_velocity

    def constant_velocity(state, time, encoded, memory):
        states.append(state.clone())
        times.append(time.clone())
        original(state, time, encoded, memory)
        return torch.ones_like(state)

    monkeypatch.setattr(policy, "_predict_velocity", constant_velocity)
    geometry_hook = policy.trajectory_decoder.path_geometry_embedding.register_forward_pre_hook(
        lambda module, arguments: geometry.append(arguments[0].clone())
    )
    scene_hook = policy.trajectory_decoder.blocks[0].memory_norm.register_forward_hook(
        lambda module, arguments, output: scene.append(output)
    )
    prediction = generate_candidate_bank(policy, encoded)
    geometry_hook.remove()
    scene_hook.remove()
    torch.testing.assert_close(torch.stack(times)[:, 0], torch.tensor([1.0, 0.5]))
    expected, _ = policy.curve_codec.decode(policy.inference_source[:16, 0].repeat(2, 1) - 1)
    torch.testing.assert_close(prediction["paths"], expected.unflatten(0, (2, 16)))


def test_current_curve_geometry_has_gradients_to_flow_state_and_field():
    policy = build_policy(tiny_config())
    encoded = make_encoded_condition(policy, 1)
    field = encoded.configuration_field.detach().clone()
    field[:, 3] = 1
    field.requires_grad_()
    encoded = replace(encoded, configuration_field=field)
    state = torch.randn(1, 14, requires_grad=True)
    path, _ = policy.curve_codec.decode(state)
    geometry, goal = policy.trajectory_decoder._trajectory_geometry(path, encoded)
    relative = policy.trajectory_decoder._path_relative_geometry(path, encoded)
    gradients = torch.autograd.grad(
        geometry.square().mean() + goal.square().mean() + relative.square().mean(),
        (state, field),
    )
    assert relative.shape == (1, 7, 261, 7)
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_pointwise_geometry_retains_contrasts_lost_by_raw_increment_means():
    decoder = build_policy(tiny_config()).trajectory_decoder.eval()
    weight = decoder.path_to_increment_weight.double()
    # Seven raw means cannot distinguish this signed local contrast from zero.
    _, _, vectors = torch.linalg.svd(weight, full_matrices=True)
    contrast = vectors[-1] / vectors[-1].abs().max()
    torch.testing.assert_close(weight @ contrast, torch.zeros(7, dtype=torch.float64))
    geometry = torch.zeros(2, 64, 8)
    geometry[1, :, 2] = contrast.float()
    goal = torch.zeros(2, 64, 5)
    # A representable pointwise nonlinearity must distinguish the distributions.
    with torch.no_grad():
        first, _, last = decoder.path_geometry_embedding
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.zero_()
        first.weight[0, 2] = 1
        last.weight[0, 0] = 1
    encoded, _ = decoder._embed_trajectory_geometry(geometry, goal)
    assert encoded.dtype == torch.float32
    assert (encoded[1] - encoded[0]).abs().max() > 0.01


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


def test_terminal_goal_is_metric_local_and_does_not_constrain_output() -> None:
    policy = build_policy(tiny_config()).eval()
    encoder = policy.condition_encoder
    goals = torch.tensor([[1.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [0.0, 0.0]])
    terminal = local_terminal_goal(goals, 3.6)

    assert torch.isfinite(terminal).all()
    torch.testing.assert_close(
        terminal[0],
        goals[0],
    )
    torch.testing.assert_close(terminal[1], torch.tensor([3.6, 0.0]))
    torch.testing.assert_close(terminal[2], torch.tensor([0.0, 3.6]))
    torch.testing.assert_close(terminal[3], torch.zeros(2))
    # The terminal intent is conditioning, not a codec constraint.
    decoded, _ = policy.curve_codec.decode(torch.full((4, 14), 10.0))
    assert torch.linalg.vector_norm(decoded[:, -1], dim=-1).max() > 3.6


def test_conditional_velocity_recovers_the_clean_flow_endpoint() -> None:
    clean = torch.randn(9, 14)
    source = torch.randn_like(clean)
    end_time = torch.rand(9)
    velocity = source - clean
    state = (1.0 - end_time[:, None]) * clean + end_time[:, None] * source
    torch.testing.assert_close(state - end_time[:, None] * velocity, clean)


def test_velocity_field_is_conditioned_on_flow_time():
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 2)
    memory = policy.trajectory_decoder.project_condition_memory(
        encoded
    )
    state = torch.randn(2, 14)
    assert not torch.equal(
        policy._predict_velocity(state, torch.full((2,), 0.2), encoded, memory),
        policy._predict_velocity(state, torch.full((2,), 0.7), encoded, memory),
    )


def test_goal_geometry_uses_one_terminal_goal_not_metric_slots() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 1)
    candidate, _ = policy.curve_codec.decode(torch.randn(1, 14))
    geometry = policy.trajectory_decoder._goal_geometry(candidate, encoded)
    expected_delta = encoded.terminal_goal[:, None, :] - candidate
    torch.testing.assert_close(
        geometry[..., 2:4] * policy.planning_horizon_m,
        expected_delta,
    )
    assert torch.count_nonzero(geometry[:, :-1, 4]) > 0
    terminal_candidate = encoded.terminal_goal[:, None, :].expand_as(candidate)
    terminal_geometry = policy.trajectory_decoder._goal_geometry(
        terminal_candidate, encoded
    )
    torch.testing.assert_close(terminal_geometry[:, -1, 4], torch.zeros(1))


def test_reusable_cross_attention_matches_projected_call() -> None:
    torch.manual_seed(4)
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    policy = build_policy(tiny_config())
    encoded = make_encoded_condition(policy, 2)
    projected = layer.project_condition(encoded)
    geometry = policy.trajectory_decoder._path_relative_geometry(
        policy.curve_codec.decode(torch.randn(2, 14))[0], encoded
    )
    output = layer(query, projected, geometry)
    assert output.shape == query.shape and torch.isfinite(output).all()


def test_reusable_cross_attention_excludes_invalid_history_tokens() -> None:
    torch.manual_seed(5)
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    policy = build_policy(tiny_config())
    encoded = make_encoded_condition(policy, 2)
    valid = encoded.token_valid.clone()
    valid[:, :5] = False
    changed = encoded.tokens.clone()
    changed[:, :5] = torch.randn_like(changed[:, :5]) * 1_000
    first_condition = replace(encoded, token_valid=valid)
    second_condition = replace(encoded, tokens=changed, token_valid=valid)
    geometry = policy.trajectory_decoder._path_relative_geometry(
        policy.curve_codec.decode(torch.randn(2, 14))[0], encoded
    )
    first = layer(query, layer.project_condition(first_condition), geometry)
    second = layer(query, layer.project_condition(second_condition), geometry)
    torch.testing.assert_close(first, second)


def test_shared_scene_kv_matches_repeated_kv_outputs_and_gradients():
    torch.manual_seed(17)
    policy = build_policy(tiny_config())
    encoded = make_encoded_condition(policy, 2)
    tokens = encoded.tokens.detach().requires_grad_()
    encoded = replace(encoded, tokens=tokens)
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    memory = layer.project_condition(encoded)
    query = torch.randn(6, 7, 32, requires_grad=True)
    geometry = torch.randn(6, 7, tokens.shape[1], 7, requires_grad=True)
    shared = layer(query, memory, geometry)
    repeated = layer(query, tuple(value.repeat_interleave(3, dim=0) for value in memory), geometry)
    torch.testing.assert_close(shared, repeated, atol=2e-6, rtol=2e-5)
    variables = (query, geometry, tokens, *layer.parameters())
    shared_grad = torch.autograd.grad(shared.square().sum(), variables, retain_graph=True)
    repeated_grad = torch.autograd.grad(repeated.square().sum(), variables)
    for actual, expected in zip(shared_grad, repeated_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_candidate_conditions_keep_the_original_scene_raster():
    from curvenav.models.policy import repeat_condition
    policy=build_policy(tiny_config());encoded=make_encoded_condition(policy,2)
    repeated=repeat_condition(encoded,16)
    assert repeated.configuration_field is encoded.configuration_field
    assert repeated.tokens.shape[0]==32
    torch.testing.assert_close(repeated.terminal_goal,encoded.terminal_goal.repeat_interleave(16,0),rtol=0,atol=0)


def test_path_relative_attention_uses_physical_query_positions() -> None:
    torch.manual_seed(6)
    policy = build_policy(tiny_config())
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    encoded = make_encoded_condition(policy, 2)
    projected = layer.project_condition(encoded)
    first_position = torch.zeros(2, 64, 2)
    first_geometry = policy.trajectory_decoder._path_relative_geometry(
        first_position, encoded
    )
    first = layer(query, projected, first_geometry)
    shifted = torch.zeros(2, 64, 2)
    shifted[..., 1] = 1.0
    second_geometry = policy.trajectory_decoder._path_relative_geometry(
        shifted, encoded
    )
    second = layer(query, projected, second_geometry)
    assert not torch.equal(first, second)


def test_flow_field_reads_observed_current_curve_geometry():
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 1)
    state, time = torch.randn(1, 14), torch.ones(1)
    memory = policy.trajectory_decoder.project_condition_memory(
        encoded
    )
    baseline = policy._predict_velocity(state, time, encoded, memory)
    field = encoded.configuration_field.clone()
    field[:, 0].fill_(-0.2)
    field[:, 1].fill_(1)
    field[:, 2].zero_()
    field[:, 3:].fill_(1)
    changed = replace(encoded, configuration_field=field)
    memory = policy.trajectory_decoder.project_condition_memory(
        changed
    )
    assert not torch.equal(baseline, policy._predict_velocity(state, time, changed, memory))


def test_deployment_has_one_reproducible_path():
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 2)
    first = generate_candidate_bank(policy, encoded)
    torch.manual_seed(999)
    second = generate_candidate_bank(policy, encoded)
    torch.testing.assert_close(first['paths'], second['paths'], rtol=0, atol=0)
    restored = build_policy(tiny_config()).eval()
    restored.load_state_dict(policy.state_dict(), strict=True)
    prediction = generate_candidate_bank(restored, encoded)
    for key in first:
        torch.testing.assert_close(first[key], prediction[key], rtol=0, atol=0)
    scores = policy.score_candidate_paths(encoded, encoded.terminal_goal, first['paths'])
    torch.testing.assert_close(scores, restored.score_candidate_paths(encoded, encoded.terminal_goal, prediction['paths']), rtol=0, atol=0)


def test_goal_does_not_relocate_candidate_relative_scene_queries() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 1)
    candidate_path, _ = policy.curve_codec.decode(torch.randn(1, 14))
    baseline = policy.trajectory_decoder._path_relative_geometry(
        candidate_path, encoded
    )
    changed = replace(encoded, terminal_goal=-encoded.terminal_goal)
    torch.testing.assert_close(
        baseline,
        policy.trajectory_decoder._path_relative_geometry(candidate_path, changed),
    )
