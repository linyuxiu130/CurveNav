"""Regular metric curves generated from a cubic B-spline heading field."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .basis import bspline_basis_and_first_derivative


HEADING_PARAMETERIZATION_TYPE = (
    "positive_softplus_arc_length_and_cubic_heading_increment_coordinates"
)
HEADING_CONTROL_POINTS = 8
HEADING_SPLINE_DEGREE = 3
CURVE_INTEGRATION_OVERSAMPLE_FACTOR = 4
LEARNED_CURVE_VALUES = HEADING_CONTROL_POINTS


class MetricHeadingTrajectory(nn.Module):
    """Decode scale-separated Euclidean values into one regular planar curve.

    The physical values are metric arc length ``L`` followed by seven local
    heading increments.  Their cumulative sum gives the seven nonzero cubic
    heading controls while the first control remains structurally zero.  This
    is a bijective differential coordinate system for the same regular curve
    family and avoids transporting strongly correlated cumulative headings.
    """

    def __init__(
        self,
        num_heading_control_points: int,
        degree: int,
        num_path_points: int,
        length_pre_activation_mean: float,
        length_pre_activation_std: float,
        heading_increment_mean_rad: tuple[float, ...],
        heading_increment_std_rad: tuple[float, ...],
    ) -> None:
        super().__init__()
        if num_heading_control_points != HEADING_CONTROL_POINTS:
            raise ValueError("CurveNav uses exactly eight heading control points")
        if degree != HEADING_SPLINE_DEGREE:
            raise ValueError("CurveNav uses one clamped cubic heading spline")
        if num_path_points < num_heading_control_points:
            raise ValueError("num_path_points must cover the heading controls")
        if length_pre_activation_std <= 0:
            raise ValueError(
                "length pre-activation standard deviation must be positive"
            )
        if len(heading_increment_mean_rad) != HEADING_CONTROL_POINTS - 1:
            raise ValueError("heading-increment mean must contain seven values")
        if (
            len(heading_increment_std_rad) != HEADING_CONTROL_POINTS - 1
            or any(value <= 0 for value in heading_increment_std_rad)
        ):
            raise ValueError("heading-increment standard deviation must be positive")

        self.num_heading_control_points = num_heading_control_points
        self.degree = degree
        self.num_path_points = num_path_points
        self.num_curve_tokens = LEARNED_CURVE_VALUES
        self.length_pre_activation_mean = float(length_pre_activation_mean)
        self.length_pre_activation_std = float(length_pre_activation_std)

        dense_points = (
            (num_path_points - 1) * CURVE_INTEGRATION_OVERSAMPLE_FACTOR + 1
        )
        dense_basis, dense_first_basis = bspline_basis_and_first_derivative(
            num_heading_control_points,
            degree,
            dense_points,
        )
        sampled_basis = dense_basis[::CURVE_INTEGRATION_OVERSAMPLE_FACTOR, 1:]
        fit_inverse = torch.linalg.solve(
            sampled_basis.T @ sampled_basis,
            sampled_basis.T,
        )
        self.register_buffer("dense_basis", dense_basis, persistent=True)
        self.register_buffer(
            "dense_first_basis", dense_first_basis, persistent=True
        )
        self.register_buffer("heading_fit_inverse", fit_inverse, persistent=True)
        self.register_buffer(
            "heading_increment_mean_rad",
            torch.tensor(heading_increment_mean_rad),
            persistent=True,
        )
        self.register_buffer(
            "heading_increment_std_rad",
            torch.tensor(heading_increment_std_rad),
            persistent=True,
        )

    def values_from_coordinates(self, coordinates: Tensor) -> Tensor:
        """Map standardized Euclidean Flow coordinates to physical values."""
        if coordinates.ndim != 2 or coordinates.shape[1] != self.num_curve_tokens:
            raise ValueError("coordinates do not match the heading-curve codec")
        coordinates = coordinates.float()
        length = F.softplus(
            self.length_pre_activation_mean
            + self.length_pre_activation_std * coordinates[:, :1]
        )
        heading_increments = (
            self.heading_increment_mean_rad
            + self.heading_increment_std_rad * coordinates[:, 1:]
        )
        return torch.cat((length, heading_increments), dim=-1)

    def coordinates_from_values(self, values: Tensor) -> Tensor:
        """Map positive arc length and heading increments to Flow coordinates."""
        if values.ndim != 2 or values.shape[1] != self.num_curve_tokens:
            raise ValueError("values do not match the heading-curve codec")
        values = values.float()
        return torch.cat(
            (
                (
                    values[:, :1]
                    + torch.log(-torch.expm1(-values[:, :1]))
                    - self.length_pre_activation_mean
                )
                / self.length_pre_activation_std,
                (values[:, 1:] - self.heading_increment_mean_rad)
                / self.heading_increment_std_rad,
            ),
            dim=-1,
        )

    def _decode_values(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if values.ndim != 2 or values.shape[1] != self.num_curve_tokens:
            raise ValueError("values do not match the heading-curve codec")
        values = values.float()
        length = values[:, 0]
        controls = torch.cat(
            (
                torch.zeros_like(length[:, None]),
                values[:, 1:].cumsum(dim=-1),
            ),
            dim=-1,
        )
        heading = controls @ self.dense_basis.T
        heading_derivative = controls @ self.dense_first_basis.T
        delta_heading = heading[:, 1:] - heading[:, :-1]
        midpoint_heading = 0.5 * (heading[:, 1:] + heading[:, :-1])
        delta_s = length[:, None] / (heading.shape[1] - 1)
        chord_length = delta_s * torch.sinc(delta_heading / (2.0 * torch.pi))
        delta_position = chord_length[..., None] * torch.stack(
            (midpoint_heading.cos(), midpoint_heading.sin()),
            dim=-1,
        )
        dense_path = torch.cat(
            (
                torch.zeros(
                    values.shape[0],
                    1,
                    2,
                    device=values.device,
                    dtype=values.dtype,
                ),
                delta_position.cumsum(dim=1),
            ),
            dim=1,
        )
        stride = CURVE_INTEGRATION_OVERSAMPLE_FACTOR
        return (
            dense_path[:, ::stride],
            heading[:, ::stride],
            (heading_derivative / length[:, None])[:, ::stride],
        )

    def decode_path(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        path, heading, _ = self._decode_values(
            self.values_from_coordinates(coordinates)
        )
        return path, heading

    def decode_values(self, values: Tensor) -> tuple[Tensor, Tensor]:
        path, heading, _ = self._decode_values(values)
        return path, heading

    def decode(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        return self.decode_path(coordinates)

    @torch.no_grad()
    def project_expert(self, reference_path: Tensor) -> tuple[Tensor, Tensor]:
        """Project an equal-arc expert onto the production heading field."""
        if reference_path.shape[1:] != (self.num_path_points, 2):
            raise ValueError("expert path does not match the heading-curve codec")
        reference_path = reference_path.float()
        segment = reference_path[:, 1:] - reference_path[:, :-1]
        segment_length = torch.linalg.vector_norm(segment, dim=-1)
        length = segment_length.sum(dim=1)
        wrapped_heading = torch.atan2(segment[..., 1], segment[..., 0])
        heading_change = torch.atan2(
            torch.sin(wrapped_heading[:, 1:] - wrapped_heading[:, :-1]),
            torch.cos(wrapped_heading[:, 1:] - wrapped_heading[:, :-1]),
        )
        unwrapped_heading = torch.cat(
            (
                wrapped_heading[:, :1],
                wrapped_heading[:, :1] + heading_change.cumsum(dim=1),
            ),
            dim=1,
        )
        vertex_heading = torch.cat(
            (
                torch.zeros_like(length[:, None]),
                0.5 * (unwrapped_heading[:, :-1] + unwrapped_heading[:, 1:]),
                unwrapped_heading[:, -1:]
                + 0.5
                * (unwrapped_heading[:, -1:] - unwrapped_heading[:, -2:-1]),
            ),
            dim=1,
        )
        heading_controls = vertex_heading @ self.heading_fit_inverse.T
        heading_increments = torch.diff(
            torch.cat((torch.zeros_like(length[:, None]), heading_controls), dim=1),
            dim=1,
        )
        values = torch.cat((length[:, None], heading_increments), dim=1)
        path, _ = self.decode_values(values)
        return values, path

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        return self.decode(coordinates)
