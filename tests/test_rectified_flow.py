import torch
from torch import nn

from curvenav.generative import RectifiedFlow
from curvenav.types import EncodedCondition


class ConstantField(nn.Module):
    def prepare_condition(self, condition):
        return condition

    def forward(self, state, time, condition):
        del time, condition
        return torch.ones_like(state)


def test_rectified_flow_never_moves_the_robot_origin() -> None:
    flow = RectifiedFlow(
        field=ConstantField(),
        num_control_points=7,
        inference_steps=3,
        source_cholesky=torch.eye(6),
        source_std_xy=(0.04, 0.12),
    )
    condition = EncodedCondition(tokens=torch.zeros(2, 4, 8))
    endpoint = torch.tensor([[0.7, -0.2], [1.3, 0.4]])
    progress = torch.linspace(0.0, 1.0, 7).view(1, 7, 1)
    source_mean = progress * endpoint.unsqueeze(1)
    sample = flow.sample(condition, source_mean=source_mean, num_samples=2)
    assert torch.equal(sample[:, 0], torch.zeros_like(sample[:, 0]))
    assert not torch.equal(sample[:, -1], endpoint.repeat_interleave(2, dim=0))


def test_interpolation_and_velocity_keep_only_the_origin_fixed() -> None:
    flow = RectifiedFlow(
        field=ConstantField(),
        num_control_points=6,
        inference_steps=2,
        source_cholesky=torch.eye(5),
        source_std_xy=(0.04, 0.12),
    )
    endpoint = torch.tensor([[1.0, 0.5], [0.8, -0.4]])
    clean = flow.enforce_origin(torch.randn(2, 6, 2))
    source = flow.enforce_origin(torch.randn(2, 6, 2))
    clean[:, -1] = endpoint
    time = torch.tensor([0.2, 0.9])
    state = flow.interpolate(clean, source, time)
    velocity = flow.velocity_target(clean, source)
    assert torch.equal(state[:, 0], torch.zeros_like(state[:, 0]))
    assert torch.equal(velocity[:, 0], torch.zeros_like(velocity[:, 0]))
    assert torch.any(velocity[:, -1] != 0)


def test_source_residual_uses_calibrated_axis_scales() -> None:
    flow = RectifiedFlow(
        field=ConstantField(),
        num_control_points=5,
        inference_steps=2,
        source_cholesky=torch.eye(4),
        source_std_xy=(0.04, 0.12),
    )
    source_mean = torch.zeros(2, 5, 2)
    torch.manual_seed(7)
    expected_noise = torch.randn(2, 4, 2)
    torch.manual_seed(7)
    source = flow.draw_source(source_mean)
    expected = expected_noise * torch.tensor([0.04, 0.12])
    assert torch.allclose(source[:, 1:], expected)
