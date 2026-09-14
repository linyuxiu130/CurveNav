"""Focused mathematical and architecture tests for the one CurveNav policy."""

from dataclasses import replace, fields
from test_depth_memory import condition as depth_condition

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
from curvenav.models.policy import repeat_condition
from test_critic import criterion_loss
from curvenav.encoders.configuration import observed_configuration_features
from curvenav.models.blocks import (
    ConditionalTrajectoryBlock,
    ReusableConditionCrossAttention,
)
from curvenav.trajectory import local_terminal_goal
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


def make_condition(batch: int = 2) -> PolicyCondition:
    value = depth_condition(batch)
    value.depth = torch.rand_like(value.depth)
    value.point_goal[:, 1] = torch.linspace(-1.0, 1.0, batch)
    return value


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
    inputs = make_condition(2)
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
    prediction = policy.sample(inputs)
    geometry_hook.remove()
    scene_hook.remove()
    torch.testing.assert_close(torch.stack(times)[:, 0], torch.tensor([1.0, 0.5]))
    expected, _ = policy.curve_codec.decode(policy.inference_source.repeat(2, 1) - 1)
    torch.testing.assert_close(prediction.candidates, expected.unflatten(0, (2, 32)))
    torch.testing.assert_close(prediction.path, prediction.candidates[torch.arange(2), prediction.scores.argmax(1)])
    assert len(scene) == 1 and len(geometry) == 2
    assert not torch.equal(geometry[0], geometry[1])
    encoded = policy.encode_condition(inputs)
    encoded = repeat_condition(encoded, 32)
    for state, actual in zip(states, geometry):
        path, _ = policy.curve_codec.decode(state)
        expected, _ = policy.trajectory_decoder._trajectory_geometry(path, encoded)
        torch.testing.assert_close(actual, expected)


def test_current_curve_geometry_has_gradients_to_flow_state_and_field():
    policy = build_policy(tiny_config())
    encoded = policy.encode_condition(make_condition(1))
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
    assert relative.shape == (1, 7, 259, 7)
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_pointwise_geometry_retains_contrasts_lost_by_raw_increment_means():
    decoder = build_policy(tiny_config()).trajectory_decoder.eval()
    weight = decoder.path_to_increment_weight.double()
    # Seven raw means cannot distinguish this signed local contrast from zero.
    _, _, vectors = torch.linalg.svd(weight, full_matrices=True)
    contrast = vectors[-1] / vectors[-1].abs().max()
    torch.testing.assert_close(weight @ contrast, torch.zeros(7, dtype=torch.float64))
    geometry = torch.zeros(2, 64, 7)
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


def test_pointgoal_never_rewrites_scene_memory() -> None:
    torch.manual_seed(0)
    policy = build_policy(tiny_config()).eval()
    first = make_condition(1)
    second = replace(
        first,
        depth=first.depth,
        point_goal=torch.tensor([[-2.0, 3.0]]),
        observation_to_current=first.observation_to_current,
        observation_valid=first.observation_valid,
    )
    first_encoded = policy.encode_condition(first)
    second_encoded = policy.encode_condition(second)
    torch.testing.assert_close(first_encoded.tokens, second_encoded.tokens)
    assert not torch.equal(first_encoded.terminal_goal, second_encoded.terminal_goal)
    assert first_encoded.tokens.shape[1] == 16 * 16 + 3
    torch.testing.assert_close(first_encoded.token_valid, second_encoded.token_valid)


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


def test_pointgoal_enters_decoder_as_terminal_intent():
    policy = build_policy(tiny_config()).eval()
    inputs = make_condition(1)
    state, time = torch.randn(1, 14), torch.ones(1)

    def velocity(c):
        encoded = policy.encode_condition(c)
        memory = policy.trajectory_decoder.project_condition_memory(
            encoded
        )
        return policy._predict_velocity(state, time, encoded, memory)

    assert not torch.equal(
        velocity(inputs),
        velocity(replace(inputs, point_goal=torch.tensor([[-2.0, 3.0]]))),
    )


def test_depth_and_pointgoal_both_condition_the_trajectory() -> None:
    torch.manual_seed(1)
    policy = build_policy(tiny_config()).eval()
    baseline = make_condition(1)
    changed_depth = replace(
        baseline,
        depth=torch.flip(baseline.depth, dims=(-1,)),
        point_goal=baseline.point_goal,
        observation_to_current=baseline.observation_to_current,
        observation_valid=baseline.observation_valid,
    )
    changed_goal = replace(
        baseline,
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
        changed = replace(
            baseline,
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
    depth = policy.depth_encoder(inputs)
    assert not depth.token_valid.any()
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
    second = replace(
        first,
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
        TrajectoryTarget(policy.curve_codec.values_from_coordinates(coordinates)),
        torch.randn_like(coordinates),
    )
    assert not losses.candidate_paths.requires_grad
    losses = criterion_loss(losses)
    assert TRAINING_LOSS_NAMES == ("loss", "flow_loss", "critic_loss")
    assert len(losses.logging_values()) == 3
    assert losses.loss.ndim == 0 and torch.isfinite(losses.loss)
    losses.loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_flow_loss_matches_the_conditional_velocity_without_source_replacement(
    monkeypatch,
):
    policy = build_policy(tiny_config())
    clean, source = torch.randn(9, 14), torch.randn(9, 14)
    target = TrajectoryTarget(policy.curve_codec.values_from_coordinates(clean))
    clean = policy.curve_codec.coordinates_from_values(target.curve_values)

    def oracle(state, time, encoded, memory):
        if len(state) != len(clean):
            return torch.zeros_like(state)
        torch.testing.assert_close(
            state, (1 - time[:, None]) * clean + time[:, None] * source
        )
        assert ((time > 0) & (time < 1)).all()
        return source - clean

    monkeypatch.setattr(policy, "_predict_velocity", oracle)
    assert policy(make_condition(9), target, source).flow_loss == 0


def test_conditional_velocity_recovers_the_clean_flow_endpoint() -> None:
    clean = torch.randn(9, 14)
    source = torch.randn_like(clean)
    end_time = torch.rand(9)
    velocity = source - clean
    state = (1.0 - end_time[:, None]) * clean + end_time[:, None] * source
    torch.testing.assert_close(state - end_time[:, None] * velocity, clean)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression")
def test_deterministic_cuda_flow_primal(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        policy = build_policy(tiny_config()).cuda()
        inputs = make_condition(8)
        inputs = replace(
            inputs,
            **{
                field.name: getattr(inputs, field.name).cuda()
                for field in fields(inputs)
            },
        )
        clean = torch.randn(8, 14, device="cuda")
        loss = policy(
            inputs,
            TrajectoryTarget(policy.curve_codec.values_from_coordinates(clean)),
            torch.randn_like(clean),
        )
        assert torch.isfinite(loss.flow_loss)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_velocity_field_is_conditioned_on_flow_time():
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(make_condition(2))
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
    encoded = policy.encode_condition(make_condition(1))
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
    encoded = policy.encode_condition(make_condition(2))
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
    encoded = policy.encode_condition(make_condition(2))
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


def test_path_relative_attention_uses_physical_query_positions() -> None:
    torch.manual_seed(6)
    policy = build_policy(tiny_config())
    layer = ReusableConditionCrossAttention(32, 4, 0).eval()
    query = torch.randn(2, 7, 32)
    encoded = policy.encode_condition(make_condition(2))
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
    encoded = policy.encode_condition(make_condition(1))
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
    inputs = make_condition(2)
    first = policy.sample(inputs)
    torch.manual_seed(999)
    second = policy.sample(inputs)
    assert [f.name for f in fields(first)] == ["path", "candidates", "scores", "selected_index"]
    torch.testing.assert_close(first.path, second.path, rtol=0, atol=0)


def test_goal_does_not_relocate_candidate_relative_scene_queries() -> None:
    policy = build_policy(tiny_config()).eval()
    encoded = policy.encode_condition(make_condition(1))
    candidate_path, _ = policy.curve_codec.decode(torch.randn(1, 14))
    baseline = policy.trajectory_decoder._path_relative_geometry(
        candidate_path, encoded
    )
    changed = replace(encoded, terminal_goal=-encoded.terminal_goal)
    torch.testing.assert_close(
        baseline,
        policy.trajectory_decoder._path_relative_geometry(candidate_path, changed),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_cuda_training_graph_and_deployment_sample_are_finite() -> None:
    device = torch.device("cuda")
    configure_cuda_training_backend()
    policy = build_policy(tiny_config()).to(device).train()
    compile_static_training_functions(policy)
    optimizer = build_optimizer(policy, learning_rate=1e-4, weight_decay=1e-2)
    inputs = make_condition(4)
    inputs = PolicyCondition(
        **{f.name: getattr(inputs, f.name).to(device) for f in fields(inputs)}
    )
    coordinates = torch.randn(4, 14, device=device) * 0.1
    target = TrajectoryTarget(policy.curve_codec.values_from_coordinates(coordinates))
    optimizer.zero_grad(set_to_none=True)
    loss = policy(
        inputs,
        target,
        torch.randn_like(coordinates),
    )
    loss = criterion_loss(loss).loss
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
    policy.eval()
    path = policy.sample(inputs).path
    assert path.shape == (4, 64, 2) and torch.isfinite(path).all()
