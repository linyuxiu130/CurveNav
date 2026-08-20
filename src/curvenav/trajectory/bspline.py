"""Differentiable planar clamped B-spline decoding."""

import torch
from torch import Tensor, nn


def _open_uniform_knots(num_control_points: int, degree: int) -> Tensor:
    internal_count = num_control_points - degree - 1
    internal = (
        torch.linspace(0.0, 1.0, internal_count + 2, dtype=torch.float32)[1:-1]
        if internal_count > 0
        else torch.empty(0, dtype=torch.float32)
    )
    return torch.cat(
        [
            torch.zeros(degree + 1, dtype=torch.float32),
            internal,
            torch.ones(degree + 1, dtype=torch.float32),
        ]
    )


def _basis_matrix(num_control_points: int, degree: int, num_samples: int) -> Tensor:
    knots = _open_uniform_knots(num_control_points, degree)
    t = torch.linspace(0.0, 1.0, num_samples, dtype=torch.float32)
    basis = ((t[:, None] >= knots[:-1]) & (t[:, None] < knots[1:])).float()
    basis[-1].zero_()
    basis[-1, -1] = 1.0

    for order in range(1, degree + 1):
        columns = []
        output_columns = knots.numel() - order - 1
        for index in range(output_columns):
            left_den = knots[index + order] - knots[index]
            right_den = knots[index + order + 1] - knots[index + 1]
            left = torch.zeros_like(t)
            right = torch.zeros_like(t)
            if float(left_den) > 0:
                left = (t - knots[index]) / left_den * basis[:, index]
            if float(right_den) > 0:
                right = (knots[index + order + 1] - t) / right_den * basis[:, index + 1]
            columns.append(left + right)
        basis = torch.stack(columns, dim=1)

    basis[-1].zero_()
    basis[-1, -1] = 1.0
    return basis[:, :num_control_points]


def _origin_conditioned_rbf_cholesky(coordinates: Tensor) -> Tensor:
    """RBF-GP residual factor conditioned to zero only at the robot origin."""
    distance = coordinates[:, None] - coordinates[None, :]
    covariance = torch.exp(-0.5 * (distance / 0.4).square())
    boundary = torch.tensor([0], device=coordinates.device)
    interior = torch.arange(1, coordinates.numel(), device=coordinates.device)
    covariance_oo = covariance[boundary[:, None], boundary]
    covariance_io = covariance[interior[:, None], boundary]
    covariance_ii = covariance[interior[:, None], interior]
    conditional = covariance_ii - covariance_io @ torch.linalg.solve(
        covariance_oo, covariance_io.T
    )
    conditional = conditional + 1e-6 * torch.eye(
        conditional.shape[0], device=conditional.device, dtype=conditional.dtype
    )
    return torch.linalg.cholesky(conditional)


class PlanarBSplineCodec(nn.Module):
    """Decode ``[B, K, 2]`` control points to a dense planar path."""

    def __init__(
        self,
        num_control_points: int = 12,
        degree: int = 3,
        num_path_points: int = 64,
    ) -> None:
        super().__init__()
        if degree != 3:
            raise ValueError("CurveNav uses one fixed cubic B-spline degree")
        if num_control_points < degree + 1:
            raise ValueError("num_control_points must be at least degree + 1")
        if num_path_points < num_control_points:
            raise ValueError("num_path_points must be at least num_control_points")
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_path_points = num_path_points
        knots = _open_uniform_knots(num_control_points, degree)
        self.register_buffer(
            "basis",
            _basis_matrix(num_control_points, degree, num_path_points),
            persistent=True,
        )
        self.register_buffer(
            "greville",
            torch.stack(
                [
                    knots[index + 1 : index + degree + 1].mean()
                    for index in range(num_control_points)
                ]
            ),
            persistent=True,
        )
        fit_basis = self.basis[:, 1:-1]
        self.register_buffer("fit_basis_pinv", torch.linalg.pinv(fit_basis), persistent=True)

    def straight_line_controls(self, endpoint: Tensor) -> Tensor:
        """Return controls whose B-spline is exactly ``u * endpoint``."""
        if endpoint.ndim != 2 or endpoint.shape[-1] != 2:
            raise ValueError("endpoint must have shape [B, 2]")
        progress = self.greville.to(device=endpoint.device, dtype=endpoint.dtype)
        return progress.view(1, -1, 1) * endpoint.unsqueeze(1)

    def origin_conditioned_source_cholesky(self) -> Tensor:
        """Return the RBF-GP factor for every control after the fixed origin."""
        return _origin_conditioned_rbf_cholesky(self.greville)

    def decode(self, control_points: Tensor) -> Tensor:
        self._validate(control_points)
        canonical = control_points.clone()
        canonical[:, 0] = 0
        basis = self.basis.to(device=canonical.device, dtype=canonical.dtype)
        return torch.einsum("nk,bkd->bnd", basis, canonical)

    def encode(self, dense_path: Tensor) -> Tensor:
        """Endpoint-constrained least-squares fit to ``[B, N, 2]`` paths."""
        expected = (self.num_path_points, 2)
        if dense_path.ndim != 3 or tuple(dense_path.shape[1:]) != expected:
            raise ValueError(f"dense_path must have shape [B, {expected[0]}, 2]")
        fit = self.fit_basis_pinv.to(device=dense_path.device, dtype=dense_path.dtype)
        basis = self.basis.to(device=dense_path.device, dtype=dense_path.dtype)
        endpoint = dense_path[:, -1:]
        residual = dense_path - basis[:, -1].view(1, -1, 1) * endpoint
        solved = torch.einsum("kn,bnd->bkd", fit, residual)
        origin = torch.zeros(
            dense_path.shape[0], 1, 2, device=dense_path.device, dtype=dense_path.dtype
        )
        return torch.cat([origin, solved, endpoint], dim=1)

    def geometry(self, dense_path: Tensor) -> tuple[Tensor, Tensor]:
        if dense_path.ndim != 3 or dense_path.shape[-1] != 2:
            raise ValueError("dense_path must have shape [B, N, 2]")
        delta = dense_path[:, 1:] - dense_path[:, :-1]
        delta = torch.cat([delta[:, :1], delta], dim=1)
        heading = torch.atan2(delta[..., 1], delta[..., 0])

        ds = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        dtheta = torch.atan2(
            torch.sin(heading[:, 1:] - heading[:, :-1]),
            torch.cos(heading[:, 1:] - heading[:, :-1]),
        )
        curvature = dtheta / ds[:, 1:]
        curvature = torch.cat([curvature[:, :1], curvature], dim=1)
        return heading, curvature

    def forward(self, control_points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dense_path = self.decode(control_points)
        heading, curvature = self.geometry(dense_path)
        return dense_path, heading, curvature

    def _validate(self, control_points: Tensor) -> None:
        expected = (self.num_control_points, 2)
        if control_points.ndim != 3 or tuple(control_points.shape[1:]) != expected:
            raise ValueError(f"control_points must have shape [B, {expected[0]}, 2]")
