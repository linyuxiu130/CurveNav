import torch

from curvenav.trajectory import PlanarBSplineCodec


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


def test_encode_recovers_decoded_control_points() -> None:
    codec = PlanarBSplineCodec(num_control_points=8, degree=3, num_path_points=31)
    control_points = torch.randn(2, 8, 2)
    control_points[:, 0] = 0
    dense_path = codec.decode(control_points)
    recovered = codec.encode(dense_path)
    assert torch.allclose(recovered, control_points, atol=1e-4)


def test_encode_keeps_both_path_endpoints_exact() -> None:
    codec = PlanarBSplineCodec(num_control_points=12, degree=3, num_path_points=64)
    path = torch.randn(3, 64, 2).cumsum(dim=1)
    path = path - path[:, :1]
    controls = codec.encode(path)
    reconstructed = codec.decode(controls)
    assert torch.equal(controls[:, 0], torch.zeros_like(controls[:, 0]))
    assert torch.allclose(reconstructed[:, -1], path[:, -1], atol=1e-6)


def test_greville_controls_decode_to_exact_straight_line() -> None:
    codec = PlanarBSplineCodec(num_control_points=12, degree=3, num_path_points=64)
    endpoint = torch.tensor([[3.0, -1.0], [0.7, 0.4]])
    controls = codec.straight_line_controls(endpoint)
    expected = torch.linspace(0.0, 1.0, 64).view(1, -1, 1) * endpoint.unsqueeze(1)
    assert torch.allclose(codec.decode(controls), expected, atol=1e-6)


def test_origin_conditioned_source_covariance_is_correlated_and_positive_definite() -> None:
    codec = PlanarBSplineCodec(num_control_points=12, degree=3, num_path_points=64)
    factor = codec.origin_conditioned_source_cholesky()
    covariance = factor @ factor.T
    assert factor.shape == (11, 11)
    assert torch.all(torch.diag(factor) > 0)
    assert covariance[4, 5] > 0
