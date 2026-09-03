"""Planar cubic B-splines in standardized physical increment space."""

import torch
from torch import Tensor, nn

from curvenav.precision import GEOMETRY_DTYPE

from .basis import bspline_basis_and_first_derivative


INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE = (
    "standardized_physical_bspline_control_increments"
)
BSPLINE_CONTROL_POINTS = 8
BSPLINE_DEGREE = 3
# Greville abscissae of the seven non-origin controls for the clamped cubic
# knot vector [0,0,0,0,.2,.4,.6,.8,1,1,1,1]. A control polygon placed at
# these fractions reproduces a linear metric horizon exactly.
METRIC_REFERENCE_PROGRESS = (
    1.0 / 15.0,
    1.0 / 5.0,
    2.0 / 5.0,
    3.0 / 5.0,
    4.0 / 5.0,
    14.0 / 15.0,
    1.0,
)


def local_terminal_goal(point_goal: Tensor, planning_horizon_m: float) -> Tensor:
    """Clip PointGoal to the local terminal position without changing its direction."""
    if point_goal.ndim != 2 or point_goal.shape[-1] != 2:
        raise ValueError("point_goal must have shape [B,2]")
    if planning_horizon_m <= 0:
        raise ValueError("planning horizon must be positive")
    point_goal = point_goal.float()
    distance = torch.linalg.vector_norm(point_goal, dim=-1, keepdim=True)
    direction = point_goal / distance.clamp_min(torch.finfo(point_goal.dtype).eps)
    endpoint = direction * distance.clamp_max(planning_horizon_m)
    return endpoint


def metric_horizon_reference(
    batch_size: int,
    planning_horizon_m: float,
    *,
    device: torch.device,
    dtype: torch.dtype = GEOMETRY_DTYPE,
) -> Tensor:
    """Return target-independent forward metric slots for scene retrieval.

    These seven controls are spatial query anchors only.  They cover the
    nominal receding-horizon distance from the robot origin and are never
    decoded as a policy output or used as a goal template.
    """
    if batch_size < 1 or planning_horizon_m <= 0:
        raise ValueError("batch size and planning horizon must be positive")
    progress = torch.tensor(
        METRIC_REFERENCE_PROGRESS,
        device=device,
        dtype=dtype,
    )
    reference = torch.zeros(
        batch_size,
        len(METRIC_REFERENCE_PROGRESS),
        2,
        device=device,
        dtype=dtype,
    )
    reference[..., 0] = progress * float(planning_horizon_m)
    return reference


class IncrementalBSplineTrajectory(nn.Module):
    """Decode standardized physical increments into one clamped cubic B-spline."""

    def __init__(
        self,
        num_control_points: int,
        degree: int,
        num_path_points: int,
        control_increment_mean_xy_m: tuple[float, ...],
        control_increment_std_xy_m: tuple[float, ...],
    ) -> None:
        super().__init__()
        if num_control_points != BSPLINE_CONTROL_POINTS:
            raise ValueError("CurveNav uses exactly eight B-spline controls")
        if degree != BSPLINE_DEGREE:
            raise ValueError("CurveNav uses one clamped cubic B-spline")
        if num_path_points < num_control_points:
            raise ValueError("path samples must cover every B-spline control")
        self.num_control_tokens = num_control_points - 1
        self.coordinate_dim = 2 * self.num_control_tokens
        if len(control_increment_mean_xy_m) != self.coordinate_dim:
            raise ValueError("control increment mean must contain fourteen values")
        if len(control_increment_std_xy_m) != self.coordinate_dim or any(
            value <= 0 for value in control_increment_std_xy_m
        ):
            raise ValueError("control increment standard deviation must be positive")

        self.num_control_points = num_control_points
        self.degree = degree
        self.num_path_points = num_path_points
        basis, first_basis = bspline_basis_and_first_derivative(
            num_control_points, degree, num_path_points
        )
        learned_basis = basis[:, 1:]
        fit_inverse = torch.linalg.solve(
            learned_basis.T @ learned_basis,
            learned_basis.T,
        )
        self.register_buffer("basis", basis, persistent=True)
        self.register_buffer("first_basis", first_basis, persistent=True)
        self.register_buffer("fit_inverse", fit_inverse, persistent=True)
        self.register_buffer(
            "control_increment_mean_xy_m",
            torch.tensor(control_increment_mean_xy_m),
            persistent=True,
        )
        self.register_buffer(
            "control_increment_std_xy_m",
            torch.tensor(control_increment_std_xy_m),
            persistent=True,
        )

    @staticmethod
    def _increments(values: Tensor) -> Tensor:
        controls = values.float().reshape(values.shape[0], -1, 2)
        origin = torch.zeros(
            values.shape[0], 1, 2, device=values.device, dtype=GEOMETRY_DTYPE
        )
        return torch.diff(torch.cat((origin, controls), dim=1), dim=1)

    def values_from_coordinates(
        self,
        coordinates: Tensor,
    ) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.coordinate_dim:
            raise ValueError("coordinates do not match the B-spline codec")
        increments = self.control_increment_mean_xy_m + (
            self.control_increment_std_xy_m * coordinates.float()
        )
        controls = increments.reshape(coordinates.shape[0], -1, 2).cumsum(dim=1)
        return controls.flatten(1)

    def coordinates_from_values(
        self,
        values: Tensor,
    ) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.coordinate_dim:
            raise ValueError("values do not match the B-spline codec")
        increments = self._increments(values).flatten(1)
        return (
            increments - self.control_increment_mean_xy_m
        ) / self.control_increment_std_xy_m

    def _controls_from_values(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.coordinate_dim:
            raise ValueError("values do not match the B-spline codec")
        learned = values.float().reshape(values.shape[0], -1, 2)
        origin = torch.zeros(
            values.shape[0], 1, 2, device=values.device, dtype=GEOMETRY_DTYPE
        )
        return torch.cat((origin, learned), dim=1)

    def decode_values(self, values: Tensor) -> tuple[Tensor, Tensor]:
        controls = self._controls_from_values(values)
        path = torch.einsum("pc,bcd->bpd", self.basis, controls)
        tangent = torch.einsum("pc,bcd->bpd", self.first_basis, controls)
        heading = torch.atan2(tangent[..., 1], tangent[..., 0])
        return path, heading

    def decode_path(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.decode_values(self.values_from_coordinates(coordinates))

    def control_positions_from_coordinates(
        self,
        coordinates: Tensor,
    ) -> Tensor:
        """Return the seven physical controls used as path-relative queries."""
        values = self.values_from_coordinates(coordinates)
        return values.reshape(values.shape[0], self.num_control_tokens, 2)

    def decode(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.decode_path(coordinates)

    @torch.no_grad()
    def project_expert(self, reference_path: Tensor) -> tuple[Tensor, Tensor]:
        if reference_path.shape[1:] != (self.num_path_points, 2):
            raise ValueError("expert path does not match the B-spline codec")
        reference_path = reference_path.float()
        if torch.count_nonzero(reference_path[:, 0]).item():
            raise ValueError("expert paths must begin at the robot origin")
        learned_controls = torch.einsum("cp,bpd->bcd", self.fit_inverse, reference_path)
        values = learned_controls.flatten(1)
        path, _ = self.decode_values(values)
        return values, path

    def forward(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.decode(coordinates)
