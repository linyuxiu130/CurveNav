"""Bounded-horizon trajectories with a hard continuous-curvature bound."""

import torch
from torch import Tensor, nn

from .bspline import bspline_basis_matrix


CURVATURE_PARAMETERIZATION_TYPE = (
    "bounded_local_arc_length_zero_tangent_cubic_curvature_bspline"
)
CURVE_INTEGRATION_OVERSAMPLE_FACTOR = 4
CURVATURE_TARGET_REGULARIZATION = 0.3
MAXIMUM_LOCAL_DETOUR_RATIO = 2.0
LENGTH_LOGIT_SCALE = 8.0


class BoundedCurvatureTrajectory(nn.Module):
    """Encode and decode a forward arc-length curve.

    Token zero contains bounded total arc length.  The remaining tokens contain
    the control values of a cubic B-spline curvature profile.  The initial
    tangent is the robot forward axis.

    Cubic B-spline bases are non-negative and form a partition of unity.  Since
    every curvature control is mapped through ``tanh``, the complete continuous
    profile satisfies ``abs(kappa(s)) < maximum_curvature_inv_m``.
    """

    def __init__(
        self,
        num_curvature_control_points: int,
        degree: int,
        num_path_points: int,
        planning_horizon_m: float,
        maximum_curvature_inv_m: float,
    ) -> None:
        super().__init__()
        if degree != 3:
            raise ValueError("CurveNav uses one cubic curvature B-spline")
        if num_curvature_control_points < degree + 1:
            raise ValueError("curvature controls must support the spline degree")
        if num_path_points < 3:
            raise ValueError("num_path_points must be at least three")
        if planning_horizon_m <= 0 or maximum_curvature_inv_m <= 0:
            raise ValueError("trajectory metric scales must be positive")

        self.num_curvature_control_points = num_curvature_control_points
        self.num_curve_tokens = num_curvature_control_points + 1
        self.degree = degree
        self.num_path_points = num_path_points
        self.planning_horizon_m = float(planning_horizon_m)
        self.maximum_curvature_inv_m = float(maximum_curvature_inv_m)
        dense_points = (num_path_points - 1) * CURVE_INTEGRATION_OVERSAMPLE_FACTOR + 1
        target_basis = bspline_basis_matrix(
            num_curvature_control_points,
            degree,
            num_path_points,
        )
        integration_basis = bspline_basis_matrix(
            num_curvature_control_points,
            degree,
            dense_points,
        )
        target_normal = target_basis.T @ target_basis
        target_projection = torch.linalg.solve(
            target_normal
            + CURVATURE_TARGET_REGULARIZATION * torch.eye(num_curvature_control_points),
            target_basis.T,
        )
        self.register_buffer(
            "target_curvature_projection",
            target_projection,
            persistent=True,
        )
        self.register_buffer(
            "integration_basis",
            integration_basis,
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
        initial_turn = torch.atan2(
            torch.sin(segment_heading[:, 0]),
            torch.cos(segment_heading[:, 0]),
        )
        initial_curvature = (
            2.0 * initial_turn / segment_length[:, 0].clamp_min(1e-6)
        )
        curvature = torch.cat(
            (
                initial_curvature[:, None],
                vertex_curvature,
                vertex_curvature[:, -1:],
            ),
            dim=1,
        )
        return segment_length.sum(dim=1), segment_heading, curvature

    def encode_target(self, reference_path: Tensor, point_goal: Tensor) -> Tensor:
        """Project an expert path into the single feasible curve coordinate space."""
        if reference_path.shape[1:] != (self.num_path_points, 2):
            raise ValueError("reference path does not match the curve codec")
        if point_goal.shape != (reference_path.shape[0], 2):
            raise ValueError("point_goal does not match the reference path")
        reference_path = reference_path.float()
        point_goal = point_goal.float()
        length, _, reference_curvature = self.path_geometry(
            reference_path
        )
        goal_distance = torch.linalg.vector_norm(point_goal, dim=-1)
        maximum_length = torch.minimum(
            torch.full_like(goal_distance, self.planning_horizon_m),
            MAXIMUM_LOCAL_DETOUR_RATIO * goal_distance,
        )
        positive_length = maximum_length > 0
        safe_maximum = torch.where(
            positive_length,
            maximum_length,
            torch.ones_like(maximum_length),
        )
        length_fraction = torch.where(
            positive_length,
            length / safe_maximum,
            0.5 * torch.ones_like(length),
        )
        epsilon = torch.finfo(reference_path.dtype).eps
        length_fraction = length_fraction.clamp(epsilon, 1.0 - epsilon)

        coordinates = torch.zeros(
            reference_path.shape[0],
            self.num_curve_tokens,
            device=reference_path.device,
            dtype=reference_path.dtype,
        )
        coordinates[:, 0] = torch.logit(length_fraction) / LENGTH_LOGIT_SCALE
        projection = self.target_curvature_projection.to(
            device=reference_path.device,
            dtype=reference_path.dtype,
        )
        curvature_controls = (
            projection[None] * reference_curvature[:, None]
        ).sum(dim=-1)
        bounded_fraction = (
            curvature_controls / self.maximum_curvature_inv_m
        ).clamp(-1.0 + epsilon, 1.0 - epsilon)
        coordinates[:, 1:] = torch.atanh(bounded_fraction)
        return coordinates

    def decode(
        self,
        coordinates: Tensor,
        point_goal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode coordinates into path, heading, and bounded curvature."""
        if coordinates.ndim != 2 or coordinates.shape[1] != self.num_curve_tokens:
            raise ValueError("coordinates do not match the curve codec")
        if point_goal.shape != (coordinates.shape[0], 2):
            raise ValueError("point_goal does not match the curve coordinates")
        coordinates = coordinates.float()
        point_goal = point_goal.float()
        goal_distance = torch.linalg.vector_norm(point_goal, dim=-1)
        maximum_length = torch.minimum(
            torch.full_like(goal_distance, self.planning_horizon_m),
            MAXIMUM_LOCAL_DETOUR_RATIO * goal_distance,
        )
        length = maximum_length * torch.sigmoid(
            LENGTH_LOGIT_SCALE * coordinates[:, 0]
        )
        curvature_controls = self.maximum_curvature_inv_m * torch.tanh(
            coordinates[:, 1:]
        )
        basis = self.integration_basis.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
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
                    coordinates.shape[0],
                    1,
                    2,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
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

    def forward(
        self,
        coordinates: Tensor,
        point_goal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.decode(coordinates, point_goal)
