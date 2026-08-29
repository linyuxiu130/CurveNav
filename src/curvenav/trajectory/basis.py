"""Fixed clamped B-spline bases shared by CurveNav's curve codec."""

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


def bspline_basis_and_first_derivative(
    num_control_points: int,
    degree: int,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    """Return the clamped basis and its analytic first derivative."""
    if num_samples < 2:
        raise ValueError("B-spline derivatives require at least two samples")
    knots = _open_uniform_knots(num_control_points, degree)
    parameter = torch.linspace(0.0, 1.0, num_samples, dtype=torch.float32)
    evaluation_parameter = parameter.clone()
    evaluation_parameter[-1] = torch.nextafter(
        evaluation_parameter[-1], torch.tensor(0.0)
    )

    def basis_for_order(order: int) -> Tensor:
        basis = (
            (evaluation_parameter[:, None] >= knots[:-1])
            & (evaluation_parameter[:, None] < knots[1:])
        ).float()
        for current_order in range(1, order + 1):
            columns = []
            for index in range(knots.numel() - current_order - 1):
                left_denominator = knots[index + current_order] - knots[index]
                right_denominator = (
                    knots[index + current_order + 1] - knots[index + 1]
                )
                value = torch.zeros_like(evaluation_parameter)
                if float(left_denominator) > 0:
                    value = value + (
                        (evaluation_parameter - knots[index])
                        / left_denominator
                        * basis[:, index]
                    )
                if float(right_denominator) > 0:
                    value = value + (
                        (knots[index + current_order + 1] - evaluation_parameter)
                        / right_denominator
                        * basis[:, index + 1]
                    )
                columns.append(value)
            basis = torch.stack(columns, dim=1)
        return basis

    basis = basis_for_order(degree)[:, :num_control_points]
    basis[-1].zero_()
    basis[-1, -1] = 1.0
    lower = basis_for_order(degree - 1)
    first_columns = []
    for index in range(num_control_points):
        value = torch.zeros_like(parameter)
        left_denominator = knots[index + degree] - knots[index]
        right_denominator = knots[index + degree + 1] - knots[index + 1]
        if float(left_denominator) > 0:
            value = value + degree / left_denominator * lower[:, index]
        if float(right_denominator) > 0:
            value = value - degree / right_denominator * lower[:, index + 1]
        first_columns.append(value)
    first = torch.stack(first_columns, dim=1)

    return basis, first
