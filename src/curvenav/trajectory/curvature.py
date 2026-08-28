"""Smooth metric trajectories in an unconstrained Euclidean Flow space."""

import torch
from torch import Tensor, nn

from .basis import bspline_basis_matrix


CURVATURE_PARAMETERIZATION_TYPE = (
    "shared_expert_and_policy_softplus_arc_length_cubic_curvature_bspline"
)
CURVE_INTEGRATION_OVERSAMPLE_FACTOR = 4
CURVATURE_VARIATION_REGULARIZATION = 1e-3


class MetricCurvatureTrajectory(nn.Module):
    """Encode and decode a forward arc-length curve.

    Dataset values are physical arc length in metres followed by seven physical
    cubic B-spline curvature controls in inverse metres.  Flow coordinates are
    standardized pre-softplus length and standardized curvature.  This is a
    bijection between positive length and an unconstrained Euclidean coordinate;
    no PointGoal-dependent cap or curvature saturation is part of the model.
    """

    def __init__(
        self,
        num_curvature_control_points: int,
        degree: int,
        num_path_points: int,
        length_pretransform_mean: float,
        length_pretransform_std: float,
        curvature_control_mean_inv_m: float,
        curvature_control_std_inv_m: float,
    ) -> None:
        super().__init__()
        if degree != 3:
            raise ValueError("CurveNav uses one cubic curvature B-spline")
        if num_curvature_control_points < degree + 1:
            raise ValueError("curvature controls must support the spline degree")
        if num_path_points < 3:
            raise ValueError("num_path_points must be at least three")
        if length_pretransform_std <= 0 or curvature_control_std_inv_m <= 0:
            raise ValueError("trajectory coordinate scales must be positive")

        self.num_curvature_control_points = num_curvature_control_points
        self.num_curve_tokens = num_curvature_control_points + 1
        self.degree = degree
        self.num_path_points = num_path_points
        self.length_pretransform_mean = float(length_pretransform_mean)
        self.length_pretransform_std = float(length_pretransform_std)
        self.curvature_control_mean_inv_m = float(curvature_control_mean_inv_m)
        self.curvature_control_std_inv_m = float(curvature_control_std_inv_m)
        dense_points = (num_path_points - 1) * CURVE_INTEGRATION_OVERSAMPLE_FACTOR + 1
        integration_basis = bspline_basis_matrix(
            num_curvature_control_points,
            degree,
            dense_points,
        )
        self.register_buffer(
            "integration_basis",
            integration_basis,
            persistent=True,
        )
        unit_delta_heading = 0.5 * (
            integration_basis[:-1] + integration_basis[1:]
        ) / (dense_points - 1)
        dense_heading_matrix = torch.cat(
            (
                torch.zeros(1, num_curvature_control_points),
                unit_delta_heading.cumsum(dim=0),
            ),
            dim=0,
        )
        heading_control_matrix = dense_heading_matrix[
            ::CURVE_INTEGRATION_OVERSAMPLE_FACTOR
        ]
        self.register_buffer(
            "heading_control_matrix",
            heading_control_matrix,
            persistent=True,
        )
        heading_design = heading_control_matrix[1:]
        first_difference = torch.zeros(
            num_curvature_control_points - 1,
            num_curvature_control_points,
        )
        difference_index = torch.arange(num_curvature_control_points - 1)
        first_difference[difference_index, difference_index] = -1.0
        first_difference[difference_index, difference_index + 1] = 1.0
        normal_matrix = (
            heading_design.T @ heading_design
            + CURVATURE_VARIATION_REGULARIZATION
            * (first_difference.T @ first_difference)
        )
        self.register_buffer(
            "heading_fit_regularized_inverse",
            torch.linalg.solve(normal_matrix, heading_design.T),
            persistent=True,
        )
    @staticmethod
    def path_geometry(path: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return arc length, segment heading, and vertex-centered curvature."""
        if path.ndim != 3 or path.shape[-1] != 2 or path.shape[1] < 3:
            raise ValueError("path must have shape [B,P,2] with P >= 3")
        delta = path[:, 1:] - path[:, :-1]
        segment_length = torch.linalg.vector_norm(delta, dim=-1)
        segment_heading = torch.atan2(delta[..., 1], delta[..., 0])
        turn = torch.atan2(
            torch.sin(segment_heading[:, 1:] - segment_heading[:, :-1]),
            torch.cos(segment_heading[:, 1:] - segment_heading[:, :-1]),
        )
        support_length = 0.5 * (segment_length[:, 1:] + segment_length[:, :-1])
        vertex_curvature = turn / support_length.clamp_min(1e-6)
        curvature = torch.cat(
            (
                vertex_curvature[:, :1],
                vertex_curvature,
                vertex_curvature[:, -1:],
            ),
            dim=1,
        )
        return segment_length.sum(dim=1), segment_heading, curvature

    def values_from_coordinates(self, coordinates: Tensor) -> Tensor:
        """Map Euclidean Flow coordinates to physical curve values."""
        if coordinates.ndim != 2 or coordinates.shape[1] != self.num_curve_tokens:
            raise ValueError("coordinates do not match the curve codec")
        coordinates = coordinates.float()
        length_pretransform = (
            self.length_pretransform_mean
            + self.length_pretransform_std * coordinates[:, :1]
        )
        return torch.cat(
            (
                torch.nn.functional.softplus(length_pretransform),
                self.curvature_control_mean_inv_m
                + self.curvature_control_std_inv_m * coordinates[:, 1:],
            ),
            dim=-1,
        )

    def coordinates_from_values(self, values: Tensor) -> Tensor:
        """Map positive metric length and physical curvature to Flow space."""
        if values.ndim != 2 or values.shape[1] != self.num_curve_tokens:
            raise ValueError("values do not match the curve codec")
        values = values.float()
        length = values[:, :1]
        length_pretransform = length + torch.log(-torch.expm1(-length))
        return torch.cat(
            (
                (length_pretransform - self.length_pretransform_mean)
                / self.length_pretransform_std,
                (values[:, 1:] - self.curvature_control_mean_inv_m)
                / self.curvature_control_std_inv_m,
            ),
            dim=-1,
        )

    def decode(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode Euclidean Flow coordinates into physical curve geometry."""
        return self.decode_values(self.values_from_coordinates(coordinates))

    def decode_values(
        self,
        values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode physical length and curvature controls."""
        if values.ndim != 2 or values.shape[1] != self.num_curve_tokens:
            raise ValueError("values do not match the curve codec")
        values = values.float()
        length = values[:, 0]
        curvature_controls = values[:, 1:]
        basis = self.integration_basis.to(
            device=values.device,
            dtype=values.dtype,
        )
        dense_curvature = (
            basis[None] * curvature_controls[:, None]
        ).sum(dim=-1)

        segments = dense_curvature.shape[1] - 1
        delta_s = length[:, None] / segments
        delta_heading = (
            0.5 * (dense_curvature[:, :-1] + dense_curvature[:, 1:]) * delta_s
        )
        dense_heading = torch.cat(
            (
                torch.zeros_like(length[:, None]),
                delta_heading.cumsum(dim=1),
            ),
            dim=1,
        )
        midpoint_heading = dense_heading[:, :-1] + 0.5 * delta_heading
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
            dense_heading[:, ::stride],
            dense_curvature[:, ::stride],
        )

    @torch.no_grad()
    def project_expert(
        self,
        reference_path: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project an equal-arc expert path into the production curve manifold.

        For fixed arc length, sampled heading is linear in the seven curvature
        controls.  A fixed Tikhonov solve minimizes heading error plus the
        squared first difference of adjacent curvature controls.  This removes
        non-identifiable alternating controls while retaining the production
        codec's zero-heading boundary condition.  The metric projection is
        decoded by the same function used at inference.
        """
        if reference_path.shape[1:] != (self.num_path_points, 2):
            raise ValueError("expert path does not match the curve codec")
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
        interior_heading = 0.5 * (
            unwrapped_heading[:, :-1] + unwrapped_heading[:, 1:]
        )
        terminal_heading = unwrapped_heading[:, -1:] + 0.5 * (
            unwrapped_heading[:, -1:] - unwrapped_heading[:, -2:-1]
        )
        target_heading = torch.cat(
            (
                torch.zeros_like(length[:, None]),
                interior_heading,
                terminal_heading,
            ),
            dim=1,
        )
        inverse = self.heading_fit_regularized_inverse.to(
            device=reference_path.device,
            dtype=reference_path.dtype,
        )
        curvature_controls = torch.einsum(
            "kn,bn->bk",
            inverse,
            target_heading[:, 1:] / length[:, None].clamp_min(1e-6),
        )
        values = torch.cat(
            (
                length[:, None],
                curvature_controls,
            ),
            dim=1,
        )
        path, heading, curvature = self.decode_values(values)
        return values, path, heading, curvature

    def forward(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.decode(coordinates)
