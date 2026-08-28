"""Fixed cubic B-spline bases shared by CurveNav's curve codec."""

import torch
from torch import Tensor


def _open_uniform_knots(num_control_points: int, degree: int) -> Tensor:
    internal_count = num_control_points - degree - 1
    internal = (
        torch.linspace(0.0, 1.0, internal_count + 2, dtype=torch.float32)[1:-1]
        if internal_count > 0
        else torch.empty(0, dtype=torch.float32)
    )
    return torch.cat(
        (
            torch.zeros(degree + 1, dtype=torch.float32),
            internal,
            torch.ones(degree + 1, dtype=torch.float32),
        )
    )


def bspline_basis_matrix(
    num_control_points: int,
    degree: int,
    num_samples: int,
) -> Tensor:
    """Evaluate one open-uniform B-spline basis on a fixed unit grid."""
    knots = _open_uniform_knots(num_control_points, degree)
    parameter = torch.linspace(0.0, 1.0, num_samples, dtype=torch.float32)
    basis = (
        (parameter[:, None] >= knots[:-1])
        & (parameter[:, None] < knots[1:])
    ).float()
    basis[-1].zero_()
    basis[-1, -1] = 1.0

    for order in range(1, degree + 1):
        columns = []
        for index in range(knots.numel() - order - 1):
            left_denominator = knots[index + order] - knots[index]
            right_denominator = knots[index + order + 1] - knots[index + 1]
            left = torch.zeros_like(parameter)
            right = torch.zeros_like(parameter)
            if float(left_denominator) > 0:
                left = (
                    (parameter - knots[index])
                    / left_denominator
                    * basis[:, index]
                )
            if float(right_denominator) > 0:
                right = (
                    (knots[index + order + 1] - parameter)
                    / right_denominator
                    * basis[:, index + 1]
                )
            columns.append(left + right)
        basis = torch.stack(columns, dim=1)

    basis[-1].zero_()
    basis[-1, -1] = 1.0
    return basis[:, :num_control_points]
