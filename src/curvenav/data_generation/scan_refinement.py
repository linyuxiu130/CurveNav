"""SCAN-Planner-inspired route-guided trajectory refinement.

This module deliberately ports the trajectory *optimization pattern*, not the
ROS/Go2 stack.  The input is already a collision-free 2-D reference route.  We
optimize resampled anchors with SCAN-style fitness, collision, smoothness and
feasibility terms, fit a cubic spline, and accept it only after the existing
hard path-safety check succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize

from curvenav.data_generation.occupancy import NavigationGrid, resample_polyline


_CONTROL_POINT_SPACING_M = 0.2
_OPTIMIZATION_CLEARANCE_M = 0.2
_MAXIMUM_DEVIATION_M = 0.35
_SMOOTHNESS_WEIGHT = 1.0
_COLLISION_WEIGHT = 2.0
_FEASIBILITY_WEIGHT = 0.1
_FITNESS_WEIGHT = 1.0
_MAXIMUM_ITERATIONS = 40


@dataclass(frozen=True)
class ScanRefinementResult:
    path_xy: np.ndarray
    diagnostics: dict[str, float | int | bool | str]


def refine_scan_style_trajectory(
    path_xy: np.ndarray,
    grid: NavigationGrid,
    *,
    minimum_clearance_m: float,
    validation_step_m: float,
    output_spacing_m: float,
) -> ScanRefinementResult:
    """Refine a safe route and conservatively fall back on any failure.

    SCAN-Planner fixes B-spline boundary controls and optimizes the remaining
    controls with four terms.  Here resampled anchors play the same role.  The
    first/last two anchors are fixed to preserve endpoints and local tangents.
    A final cubic spline is never trusted solely on its soft collision cost:
    every returned refinement must pass ``NavigationGrid.path_is_safe``.
    """

    baseline = resample_polyline(np.asarray(path_xy, dtype=np.float64), output_spacing_m)
    baseline_score = _smoothness_score(baseline)
    base_diag: dict[str, float | int | bool | str] = {
        "method": "scan_style_lbfgs",
        "accepted": False,
        "optimizer_iterations": 0,
        "baseline_length_m": _path_length(baseline),
        "refined_length_m": _path_length(baseline),
        "baseline_curvature_p95": _curvature_p95(baseline),
        "refined_curvature_p95": _curvature_p95(baseline),
        "baseline_maximum_curvature": _maximum_curvature(baseline),
        "refined_maximum_curvature": _maximum_curvature(baseline),
        "baseline_spatial_jerk_rms": baseline_score[0],
        "refined_spatial_jerk_rms": baseline_score[0],
    }
    reference = resample_polyline(baseline, _CONTROL_POINT_SPACING_M)
    if len(reference) < 7:
        base_diag["reason"] = "too_few_control_points"
        return ScanRefinementResult(baseline, base_diag)

    count = len(reference)
    fixed = np.zeros(count, dtype=bool)
    fixed[:2] = True
    fixed[-2:] = True
    variable = np.flatnonzero(~fixed)
    tangent = _reference_tangents(reference)
    difference2 = np.diff(np.eye(count), n=2, axis=0)
    difference3 = np.diff(np.eye(count), n=3, axis=0)
    sample_matrix = _anchor_and_midpoint_matrix(count)
    x0 = reference[variable].reshape(-1)
    bounds = [
        (
            float(reference[index, axis] - _MAXIMUM_DEVIATION_M),
            float(reference[index, axis] + _MAXIMUM_DEVIATION_M),
        )
        for index in variable
        for axis in range(2)
    ]

    def unpack(values: np.ndarray) -> np.ndarray:
        anchors = reference.copy()
        anchors[variable] = values.reshape(-1, 2)
        return anchors

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        anchors = unpack(values)
        gradient = np.zeros_like(anchors)

        jerk = difference3 @ anchors
        smooth_cost = float(np.mean(np.sum(jerk * jerk, axis=1)))
        gradient += (
            _SMOOTHNESS_WEIGHT
            * (2.0 / max(len(jerk), 1))
            * (difference3.T @ jerk)
        )

        acceleration = difference2 @ anchors
        feasible_cost = float(np.mean(np.sum(acceleration * acceleration, axis=1)))
        gradient += (
            _FEASIBILITY_WEIGHT
            * (2.0 / max(len(acceleration), 1))
            * (difference2.T @ acceleration)
        )

        displacement = anchors - reference
        along = np.sum(displacement * tangent, axis=1, keepdims=True)
        lateral = displacement - along * tangent
        # Match SCAN-Planner's anisotropic route fitness: movement along the
        # reference is penalized 25x less than lateral route departure.
        fitness_cost = float(np.mean(along[:, 0] ** 2 / 25.0 + np.sum(lateral**2, axis=1)))
        fitness_gradient = 2.0 * (along * tangent / 25.0 + lateral) / count
        gradient += _FITNESS_WEIGHT * fitness_gradient

        collision_samples = sample_matrix @ anchors
        clearance, clearance_gradient = _bilinear_clearance_with_gradient(
            grid, collision_samples
        )
        optimization_clearance = max(_OPTIMIZATION_CLEARANCE_M, minimum_clearance_m)
        deficit = np.maximum(optimization_clearance - clearance, 0.0)
        collision_cost = float(np.mean(deficit**2))
        sample_gradient = (
            -2.0 * deficit[:, None] * clearance_gradient / max(len(deficit), 1)
        )
        gradient += _COLLISION_WEIGHT * (sample_matrix.T @ sample_gradient)

        total = (
            _SMOOTHNESS_WEIGHT * smooth_cost
            + _COLLISION_WEIGHT * collision_cost
            + _FEASIBILITY_WEIGHT * feasible_cost
            + _FITNESS_WEIGHT * fitness_cost
        )
        return total, gradient[variable].reshape(-1)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": _MAXIMUM_ITERATIONS,
            "maxls": 20,
            "ftol": 1e-10,
            "gtol": 1e-7,
        },
    )
    base_diag["optimizer_iterations"] = int(result.nit)
    optimized = unpack(np.asarray(result.x, dtype=np.float64))
    safe_candidates: list[np.ndarray] = []
    # Soft collision penalties are not a safety certificate.  Backtracking
    # toward the already-safe reference often recovers a valid, still smoother
    # trajectory if the full optimizer step cuts a grid corner.
    for alpha in (1.0, 0.75, 0.5, 0.25):
        anchors = reference + alpha * (optimized - reference)
        anchors[[0, -1]] = reference[[0, -1]]
        candidate = _interpolate_cubic(anchors, output_spacing_m)
        if candidate is None:
            continue
        if _path_length(candidate) > _path_length(baseline) * 1.05 + 1e-6:
            continue
        if grid.path_is_safe(
            candidate,
            minimum_clearance_m=minimum_clearance_m,
            sample_step_m=validation_step_m,
        ):
            safe_candidates.append(candidate)

    if not safe_candidates:
        base_diag["reason"] = "no_hard_safe_refinement"
        return ScanRefinementResult(baseline, base_diag)

    baseline_p95 = _curvature_p95(baseline)
    baseline_maximum = _maximum_curvature(baseline)
    curvature_compatible = []
    for candidate in safe_candidates:
        candidate_p95 = _curvature_p95(candidate)
        candidate_maximum = _maximum_curvature(candidate)
        p95_compatible = candidate_p95 <= max(
            baseline_p95 * 1.25, baseline_p95 + 0.15
        )
        # Rounding a single sharp corner can intentionally turn an almost-zero
        # p95 into a finite value while greatly reducing the unsafe peak.  That
        # is a genuine improvement, so accept it when peak curvature falls by
        # at least 20 percent.
        peak_substantially_better = candidate_maximum <= baseline_maximum * 0.8
        peak_compatible = candidate_maximum <= max(
            baseline_maximum * 1.05, baseline_maximum + 0.2
        )
        if (p95_compatible or peak_substantially_better) and peak_compatible:
            curvature_compatible.append(candidate)
    if not curvature_compatible:
        base_diag["reason"] = "curvature_not_improved"
        return ScanRefinementResult(baseline, base_diag)

    refined = min(curvature_compatible, key=_smoothness_score)
    refined_score = _smoothness_score(refined)
    # Do not accept a nominal optimizer improvement that is less smooth after
    # actual spline interpolation and uniform output resampling.
    if refined_score >= baseline_score:
        base_diag["reason"] = "no_sampled_smoothness_improvement"
        return ScanRefinementResult(baseline, base_diag)

    base_diag.update(
        {
            "accepted": True,
            "reason": "accepted",
            "refined_length_m": _path_length(refined),
            "refined_curvature_p95": _curvature_p95(refined),
            "refined_maximum_curvature": _maximum_curvature(refined),
            "refined_spatial_jerk_rms": refined_score[0],
        }
    )
    return ScanRefinementResult(refined, base_diag)


def _reference_tangents(reference: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(reference)
    tangent[0] = reference[1] - reference[0]
    tangent[-1] = reference[-1] - reference[-2]
    tangent[1:-1] = reference[2:] - reference[:-2]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    fallback = np.array([[1.0, 0.0]])
    return np.where(norm > 1e-9, tangent / np.maximum(norm, 1e-9), fallback)


def _anchor_and_midpoint_matrix(count: int) -> np.ndarray:
    identity = np.eye(count)
    midpoint = np.zeros((count - 1, count), dtype=np.float64)
    rows = np.arange(count - 1)
    midpoint[rows, rows] = 0.5
    midpoint[rows, rows + 1] = 0.5
    return np.vstack([identity, midpoint])


def _bilinear_clearance_with_gradient(
    grid: NavigationGrid, points_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the center-safe clearance field and its analytic gradient."""

    coordinates = (points_xy - grid.origin_xy) / grid.cell_size_m - 0.5
    lower = np.floor(coordinates).astype(np.int64)
    fraction = coordinates - lower
    valid = (
        (lower[:, 0] >= 0)
        & (lower[:, 1] >= 0)
        & (lower[:, 0] + 1 < grid.clearance_m.shape[0])
        & (lower[:, 1] + 1 < grid.clearance_m.shape[1])
    )
    value = np.zeros(len(points_xy), dtype=np.float64)
    gradient = np.zeros_like(points_xy, dtype=np.float64)
    if not np.any(valid):
        return value, gradient

    index = lower[valid]
    weight = fraction[valid]
    ix, iy = index[:, 0], index[:, 1]
    wx, wy = weight[:, 0], weight[:, 1]
    field = grid.clearance_m
    v00 = field[ix, iy].astype(np.float64)
    v10 = field[ix + 1, iy].astype(np.float64)
    v01 = field[ix, iy + 1].astype(np.float64)
    v11 = field[ix + 1, iy + 1].astype(np.float64)
    value[valid] = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    gradient[valid, 0] = (
        (1.0 - wy) * (v10 - v00) + wy * (v11 - v01)
    ) / grid.cell_size_m
    gradient[valid, 1] = (
        (1.0 - wx) * (v01 - v00) + wx * (v11 - v10)
    ) / grid.cell_size_m
    return value, gradient


def _interpolate_cubic(anchors: np.ndarray, spacing_m: float) -> np.ndarray | None:
    segment = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    if not np.all(np.isfinite(segment)) or float(segment.sum()) <= 1e-8:
        return None
    keep = np.concatenate([[True], segment > 1e-7])
    anchors = anchors[keep]
    if len(anchors) < 2:
        return None
    segment = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    parameter = cumulative / cumulative[-1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spline, _ = splprep(
                anchors.T,
                u=parameter,
                k=min(3, len(anchors) - 1),
                s=0.0,
            )
        count = max(2, int(math.ceil(cumulative[-1] / spacing_m)) + 1)
        curve = np.column_stack(splev(np.linspace(0.0, 1.0, count), spline))
    except (TypeError, ValueError):
        return None
    curve[[0, -1]] = anchors[[0, -1]]
    return np.asarray(curve, dtype=np.float64)


def _path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _curvature(path: np.ndarray) -> np.ndarray:
    if len(path) < 3:
        return np.zeros(1, dtype=np.float64)
    first = path[1:-1] - path[:-2]
    second = path[2:] - path[1:-1]
    chord = path[2:] - path[:-2]
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(chord, axis=1)
    ).clip(1e-8)
    twice_area = 2.0 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    values = twice_area / denominator
    trim = max(1, int(round(len(values) * 0.03)))
    return values[trim:-trim] if len(values) > 2 * trim + 2 else values


def _curvature_p95(path: np.ndarray) -> float:
    return float(np.percentile(_curvature(path), 95))


def _maximum_curvature(path: np.ndarray) -> float:
    return float(np.max(_curvature(path)))


def _smoothness_score(path: np.ndarray) -> tuple[float, float, float]:
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    spacing = max(float(np.mean(segment)), 1e-6)
    if len(path) < 4:
        jerk_rms = 0.0
    else:
        jerk = np.diff(path, n=3, axis=0) / spacing**3
        jerk_rms = float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))
    return jerk_rms, _curvature_p95(path), _path_length(path)
