"""Simulator-free checks for the v2 dynamics audit mathematics."""

from __future__ import annotations

import numpy as np

from curvenav.data_generation.audit_hssd_v2_dynamics import (
    _speed_cap_for_curvature,
    _wheel_speeds,
    geometric_curvature,
)


def main() -> None:
    radius = 2.0
    angle = np.linspace(0.0, np.pi / 2.0, 101)
    quarter_circle = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    curvature = geometric_curvature(quarter_circle)
    assert np.allclose(curvature, 1.0 / radius, atol=1e-10)
    assert _speed_cap_for_curvature(0.5) == 0.5
    assert _speed_cap_for_curvature(2.0) == 0.25
    left, right = _wheel_speeds(0.0, 0.5)
    assert left < 0.0 < right
    assert np.isclose(abs(left), abs(right))
    print("CurveNav v2 dynamics audit checks passed")


if __name__ == "__main__":
    main()
