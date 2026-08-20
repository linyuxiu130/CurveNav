import copy

import torch

from curvenav import PolicyCondition, TrajectoryTarget, build_policy
from curvenav.config import (
    ConditionEncoderConfig,
    CurveNavConfig,
    DataConfig,
    DepthEncoderConfig,
    FieldConfig,
    GoalEncoderConfig,
    MotionEncoderConfig,
    RectifiedFlowConfig,
    TrajectoryConfig,
)


def tiny_config() -> CurveNavConfig:
    return CurveNavConfig(
        data=DataConfig(sequence_length=4),
        trajectory=TrajectoryConfig(num_control_points=6, num_path_points=16),
        depth_encoder=DepthEncoderConfig(
            model_dim=32,
            frame_tokens_per_side=2,
        ),
        goal_encoder=GoalEncoderConfig(model_dim=32, hidden_dim=32),
        motion_encoder=MotionEncoderConfig(model_dim=32, hidden_dim=32),
        condition_encoder=ConditionEncoderConfig(
            model_dim=32,
            transformer_layers=1,
            transformer_heads=4,
        ),
        field=FieldConfig(model_dim=32, transformer_layers=1, transformer_heads=4),
        rectified_flow=RectifiedFlowConfig(inference_steps=2),
    )


def test_policy_training_and_sampling_contract() -> None:
    torch.manual_seed(0)
    policy = build_policy(tiny_config())
    condition = PolicyCondition(
        depth=torch.rand(2, 4, 1, 32, 32),
        task_goal=torch.tensor([[4.0, 1.0], [3.0, -2.0]]),
        motion_context=torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
    )
    target_controls = torch.randn(2, 6, 2)
    target_controls[:, 0] = 0
    target_controls[:, -1] = torch.tensor([[1.2, 0.6], [0.8, -0.7]])
    target = TrajectoryTarget(control_points=target_controls)
    loss = policy(condition, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())

    prediction = policy.sample(condition, num_samples=3)
    assert prediction.control_points.shape == (6, 6, 2)
    assert prediction.dense_path.shape == (6, 16, 2)
    assert torch.isfinite(prediction.dense_path).all()
    assert torch.equal(
        prediction.control_points[:, 0], torch.zeros_like(prediction.control_points[:, 0])
    )
    task_goals = condition.task_goal.repeat_interleave(3, dim=0)
    assert not torch.allclose(prediction.control_points[:, -1], task_goals)


def test_task_goal_only_caps_the_flow_source_prior() -> None:
    policy = build_policy(tiny_config())
    task_goal = torch.tensor([[3.0, 4.0], [12.0, 5.0]])

    source = policy.source_mean(task_goal)
    source_endpoint = policy.normalizer.denormalize(source[:, -1])

    torch.testing.assert_close(source_endpoint[0], task_goal[0])
    torch.testing.assert_close(
        torch.linalg.vector_norm(source_endpoint[1]),
        policy.normalizer.scale_xy[0],
    )
    torch.testing.assert_close(
        source_endpoint[1] / torch.linalg.vector_norm(source_endpoint[1]),
        task_goal[1] / torch.linalg.vector_norm(task_goal[1]),
    )


def test_depth_encoder_uses_the_fixed_stride4_stem() -> None:
    policy = build_policy(tiny_config())
    stem = policy.depth_encoder.backbone[0][0]
    assert isinstance(stem, torch.nn.Conv2d)
    assert stem.kernel_size == (5, 5)
    assert stem.stride == (4, 4)
    assert stem.in_channels == 4

    first_residual_stage = policy.depth_encoder.backbone[1]
    assert first_residual_stage.convolution_1.stride == (2, 2)
    final_refinement = policy.depth_encoder.backbone[3]
    assert final_refinement.convolution.kernel_size == (3, 3)
    assert final_refinement.convolution.dilation == (2, 2)
    assert final_refinement.convolution.padding == (2, 2)
    assert len(policy.depth_encoder.backbone) == 5

    tokens = policy.depth_encoder(torch.rand(2, 4, 1, 32, 32))
    assert tokens.shape == (2, 4, 32)


def test_condition_key_value_cache_matches_cross_attention() -> None:
    torch.manual_seed(1)
    policy = build_policy(tiny_config()).eval()
    reference_block = policy.rectified_flow.field.blocks[0]
    cached_block = copy.deepcopy(reference_block)
    reference_query = torch.randn(3, 6, 32, requires_grad=True)
    reference_memory = torch.randn(3, 6, 32, requires_grad=True)
    cached_query = reference_query.detach().clone().requires_grad_()
    cached_memory = reference_memory.detach().clone().requires_grad_()

    normalized_memory = reference_block.memory_norm(reference_memory)
    reference = reference_block.cross_attention(
        reference_query,
        normalized_memory,
        normalized_memory,
        need_weights=False,
    )[0]
    cached = cached_block._cross_attend(
        cached_query,
        cached_block.prepare_memory(cached_memory),
    )

    assert torch.equal(reference, cached)
    reference_variables = (
        reference_query,
        reference_memory,
        reference_block.memory_norm.weight,
        reference_block.cross_attention.in_proj_weight,
        reference_block.cross_attention.in_proj_bias,
        reference_block.cross_attention.out_proj.weight,
        reference_block.cross_attention.out_proj.bias,
    )
    cached_variables = (
        cached_query,
        cached_memory,
        cached_block.memory_norm.weight,
        cached_block.cross_attention.in_proj_weight,
        cached_block.cross_attention.in_proj_bias,
        cached_block.cross_attention.out_proj.weight,
        cached_block.cross_attention.out_proj.bias,
    )
    reference_gradients = torch.autograd.grad(reference.square().mean(), reference_variables)
    cached_gradients = torch.autograd.grad(cached.square().mean(), cached_variables)
    assert all(
        torch.equal(reference_gradient, cached_gradient)
        for reference_gradient, cached_gradient in zip(
            reference_gradients,
            cached_gradients,
        )
    )


def test_condition_key_value_is_prepared_once_across_flow_steps() -> None:
    torch.manual_seed(2)
    policy = build_policy(tiny_config()).eval()
    condition = PolicyCondition(
        depth=torch.rand(1, 4, 1, 32, 32),
        task_goal=torch.tensor([[0.8, -0.1]]),
        motion_context=torch.tensor([[1.0, 0.0, 1.0]]),
    )
    calls = 0

    def count_memory_normalization(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = policy.rectified_flow.field.blocks[0].memory_norm.register_forward_hook(
        count_memory_normalization
    )
    policy.sample(condition, num_samples=3)
    handle.remove()

    assert calls == 1
