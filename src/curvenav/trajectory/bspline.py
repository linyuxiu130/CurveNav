"""Differentiable planar clamped B-spline decoding."""

import torch
from torch import Tensor, nn

from .resampling import resample_path_by_arc_length


BSPLINE_BENDING_REGULARIZATION_M4 = 1e-5
ARC_LENGTH_OVERSAMPLE_FACTOR = 4


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


def bspline_basis_matrix(
    num_control_points: int,
    degree: int,
    num_samples: int,
) -> Tensor:
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


def _second_derivative_control_matrix(
    num_control_points: int,
    degree: int,
) -> Tensor:
    """Map spline controls to the controls of its second derivative."""
    knots = _open_uniform_knots(num_control_points, degree)
    first = torch.zeros(num_control_points - 1, num_control_points)
    for index in range(num_control_points - 1):
        scale = degree / (knots[index + degree + 1] - knots[index + 1])
        first[index, index] = -scale
        first[index, index + 1] = scale

    derivative_knots = knots[1:-1]
    second = torch.zeros(num_control_points - 2, num_control_points - 1)
    for index in range(num_control_points - 2):
        scale = (degree - 1) / (
            derivative_knots[index + degree] - derivative_knots[index + 1]
        )
        second[index, index] = -scale
        second[index, index + 1] = scale
    return second @ first


class PlanarBSplineCodec(nn.Module):
    """Fit and sample planar paths represented by B-spline control points."""

    def __init__(
        self,
        num_control_points: int = 8,
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
        self.register_buffer(
            "basis",
            bspline_basis_matrix(num_control_points, degree, num_path_points),
            persistent=True,
        )
        self.register_buffer(
            "arc_basis",
            bspline_basis_matrix(
                num_control_points,
                degree,
                ARC_LENGTH_OVERSAMPLE_FACTOR * num_path_points,
            ),
            persistent=True,
        )
        fit_basis = self.basis[:, 1:-1]
        second_derivative = _second_derivative_control_matrix(
            num_control_points,
            degree,
        )
        free_second_derivative = second_derivative[:, 1:-1]
        self.register_buffer("fit_basis_transpose", fit_basis.T, persistent=True)
        self.register_buffer("fit_normal", fit_basis.T @ fit_basis, persistent=True)
        self.register_buffer(
            "fit_bending_normal",
            free_second_derivative.T @ free_second_derivative,
            persistent=True,
        )
        self.register_buffer(
            "fit_endpoint_bending",
            free_second_derivative.T @ second_derivative[:, -1],
            persistent=True,
        )

    def decode_parameter_grid(self, control_points: Tensor) -> Tensor:
        """Evaluate the spline on its uniform parameter grid."""
        self._validate(control_points)
        anchored_controls = control_points.clone()
        anchored_controls[:, 0] = 0
        basis = self.basis.to(
            device=anchored_controls.device, dtype=anchored_controls.dtype
        )
        return torch.einsum("nk,bkd->bnd", basis, anchored_controls)

    def decode_equal_arc(self, control_points: Tensor) -> Tensor:
        """Evaluate and resample the spline at uniform metric arc progress."""
        self._validate(control_points)
        anchored_controls = control_points.clone()
        anchored_controls[:, 0] = 0
        basis = self.arc_basis.to(
            device=anchored_controls.device,
            dtype=anchored_controls.dtype,
        )
        oversampled = torch.einsum("nk,bkd->bnd", basis, anchored_controls)
        return resample_path_by_arc_length(
            oversampled,
            self.num_path_points,
        )

    def encode(self, reference_path: Tensor) -> Tensor:
        """Fit an endpoint-constrained, scale-aware smooth spline."""
        expected = (self.num_path_points, 2)
        if reference_path.ndim != 3 or tuple(reference_path.shape[1:]) != expected:
            raise ValueError(f"reference_path must have shape [B, {expected[0]}, 2]")
        if not torch.isfinite(reference_path).all():
            raise ValueError("reference_path must be finite")
        basis = self.basis.to(device=reference_path.device, dtype=reference_path.dtype)
        endpoint = reference_path[:, -1:]
        residual = reference_path - basis[:, -1].view(1, -1, 1) * endpoint
        arc_length = torch.linalg.vector_norm(
            reference_path[:, 1:] - reference_path[:, :-1], dim=-1
        ).sum(dim=1)
        bending_weight = BSPLINE_BENDING_REGULARIZATION_M4 / arc_length.clamp_min(
            1e-3
        ).pow(4)
        fit_transpose = self.fit_basis_transpose.to(
            device=reference_path.device, dtype=reference_path.dtype
        )
        fit_normal = self.fit_normal.to(
            device=reference_path.device, dtype=reference_path.dtype
        )
        bending_normal = self.fit_bending_normal.to(
            device=reference_path.device, dtype=reference_path.dtype
        )
        endpoint_bending = self.fit_endpoint_bending.to(
            device=reference_path.device, dtype=reference_path.dtype
        )
        normal = fit_normal.unsqueeze(0) + bending_weight[:, None, None] * (
            bending_normal.unsqueeze(0)
        )
        right_hand_side = torch.einsum("kn,bnd->bkd", fit_transpose, residual)
        right_hand_side = right_hand_side - bending_weight[:, None, None] * (
            endpoint_bending.view(1, -1, 1) * endpoint
        )
        solved = torch.linalg.solve(normal, right_hand_side)
        origin = torch.zeros(
            reference_path.shape[0],
            1,
            2,
            device=reference_path.device,
            dtype=reference_path.dtype,
        )
        return torch.cat([origin, solved, endpoint], dim=1)

    def geometry(self, path: Tensor) -> tuple[Tensor, Tensor]:
        if path.ndim != 3 or path.shape[-1] != 2:
            raise ValueError("path must have shape [B, N, 2]")
        delta = path[:, 1:] - path[:, :-1]
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
        path = self.decode_equal_arc(control_points)
        heading, curvature = self.geometry(path)
        return path, heading, curvature

    def _validate(self, control_points: Tensor) -> None:
        expected = (self.num_control_points, 2)
        if control_points.ndim != 3 or tuple(control_points.shape[1:]) != expected:
            raise ValueError(f"control_points must have shape [B, {expected[0]}, 2]")
