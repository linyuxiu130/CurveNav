import torch

from curvenav.trajectory import PlanarBSplineCodec, resample_path_by_arc_length


def test_basis_is_partition_of_unity_and_clamped() -> None:
    codec = PlanarBSplineCodec(num_control_points=8, degree=3, num_path_points=31)
    assert torch.allclose(codec.basis.sum(dim=1), torch.ones(31), atol=1e-6)
    assert torch.equal(codec.basis[0], torch.nn.functional.one_hot(torch.tensor(0), 8).float())
    assert torch.equal(codec.basis[-1], torch.nn.functional.one_hot(torch.tensor(7), 8).float())


def test_decode_preserves_origin_and_endpoint() -> None:
    codec = PlanarBSplineCodec(num_control_points=8, degree=3, num_path_points=31)
    control_points = torch.randn(2, 8, 2, requires_grad=True)
    path, heading, curvature = codec(control_points)
    assert path.shape == (2, 31, 2)
    assert heading.shape == (2, 31)
    assert curvature.shape == (2, 31)
    assert torch.equal(path[:, 0], torch.zeros_like(path[:, 0]))
    assert torch.allclose(path[:, -1], control_points[:, -1])
    path.square().mean().backward()
    assert control_points.grad is not None


def test_executable_decode_has_uniform_arc_progress() -> None:
    codec = PlanarBSplineCodec(num_control_points=8, degree=3, num_path_points=64)
    controls = torch.tensor(
        [[[0.0, 0.0], [0.1, 0.0], [0.3, 0.1], [0.7, 0.4],
          [1.2, 0.5], [1.8, 0.3], [2.5, 0.1], [3.2, 0.0]]]
    )
    path = codec.decode_equal_arc(controls)
    segment_length = torch.linalg.vector_norm(path[:, 1:] - path[:, :-1], dim=-1)

    assert segment_length.std() / segment_length.mean() < 0.03
    torch.testing.assert_close(path[:, -1], controls[:, -1])


def test_encode_suppresses_short_path_endpoint_ringing() -> None:
    codec = PlanarBSplineCodec(num_control_points=12, degree=3, num_path_points=64)
    vertices = torch.tensor([[[0.0, 0.0], [0.118, -0.054], [0.144, -0.106]]])
    path = resample_path_by_arc_length(vertices, num_samples=64)

    controls = codec.encode(path)
    reconstructed, _, curvature = codec(controls)

    assert torch.sqrt((reconstructed - path).square().mean()) < 0.005
    assert curvature.abs().max() < 10.0
    torch.testing.assert_close(reconstructed[:, -1], path[:, -1])


def test_encode_keeps_both_path_endpoints_exact() -> None:
    codec = PlanarBSplineCodec(num_control_points=12, degree=3, num_path_points=64)
    path = torch.randn(3, 64, 2).cumsum(dim=1)
    path = path - path[:, :1]
    controls = codec.encode(path)
    reconstructed = codec.decode_parameter_grid(controls)
    assert torch.equal(controls[:, 0], torch.zeros_like(controls[:, 0]))
    assert torch.allclose(reconstructed[:, -1], path[:, -1], atol=1e-6)
