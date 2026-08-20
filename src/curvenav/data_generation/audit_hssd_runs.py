"""Audit generated HSSD runs after the exact CurveNav B-spline codec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from curvenav.data_generation.occupancy import NavigationGrid
from curvenav.trajectory import PlanarBSplineCodec


CONTINUOUS_CLEARANCE_STEP_M = 0.025


def _resample_fixed(path: np.ndarray, count: int) -> np.ndarray:
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, path[:, axis]) for axis in range(2)]
    ).astype(np.float32)


def _grid_from_npz(path: Path) -> NavigationGrid:
    values = np.load(path)
    return NavigationGrid(
        free=values["free"],
        clearance_m=values["clearance_m"],
        origin_xy=values["origin_xy"],
        cell_size_m=float(values["cell_size_m"]),
    )


def _continuous_clearance(
    grid: NavigationGrid,
    paths: np.ndarray,
    *,
    sample_step_m: float = CONTINUOUS_CLEARANCE_STEP_M,
) -> np.ndarray:
    """Sample every decoded segment densely enough for the planner safety contract."""
    segments = np.diff(paths, axis=1)
    subdivisions = max(
        1,
        int(np.ceil(np.linalg.norm(segments, axis=2).max() / sample_step_m)),
    )
    fractions = np.arange(subdivisions, dtype=np.float64) / subdivisions
    dense = paths[:, :-1, None] + segments[:, :, None] * fractions[None, None, :, None]
    dense = np.concatenate(
        [dense.reshape(len(paths), -1, 2), paths[:, -1:, :]], axis=1
    )
    return grid.sample_clearance(dense.reshape(-1, 2)).reshape(len(paths), -1)


def _audit_run(
    run_dir: Path,
    dataset_name: str,
    grid: NavigationGrid,
    codec: PlanarBSplineCodec,
    min_gap: int,
    max_gap: int,
    minimum_clearance_m: float,
) -> dict[str, object]:
    world = np.load(run_dir / "traj_xyz.npy")[:, :2].astype(np.float64)
    yaw = np.load(run_dir / "traj_yaw.npy").astype(np.float64)
    canonical = []
    starts = []
    rotations = []
    windows = []
    for start in range(len(world) - min_gap):
        for end in range(start + min_gap, min(start + max_gap, len(world) - 1) + 1):
            delta = world[start : end + 1] - world[start]
            cosine = float(np.cos(yaw[start]))
            sine = float(np.sin(yaw[start]))
            local = np.column_stack(
                [
                    delta[:, 0] * cosine + delta[:, 1] * sine,
                    -delta[:, 0] * sine + delta[:, 1] * cosine,
                ]
            )
            canonical.append(_resample_fixed(local, codec.num_path_points))
            starts.append(world[start])
            rotations.append((cosine, sine))
            windows.append((start, end))

    canonical_tensor = torch.from_numpy(np.stack(canonical))
    with torch.no_grad():
        decoded = codec.decode(codec.encode(canonical_tensor)).numpy()
    canonical_array = canonical_tensor.numpy()
    fit_error = np.linalg.norm(decoded - canonical_array, axis=2)

    starts_array = np.asarray(starts)
    rotations_array = np.asarray(rotations)
    cosine = rotations_array[:, 0, None]
    sine = rotations_array[:, 1, None]
    world_x = starts_array[:, 0, None] + decoded[:, :, 0] * cosine - decoded[:, :, 1] * sine
    world_y = starts_array[:, 1, None] + decoded[:, :, 0] * sine + decoded[:, :, 1] * cosine
    decoded_world = np.stack([world_x, world_y], axis=-1)
    clearance = _continuous_clearance(grid, decoded_world)
    minimum_per_window = clearance.min(axis=1)
    unsafe = np.flatnonzero(minimum_per_window + 1e-9 < minimum_clearance_m)
    worst_index = int(np.argmin(minimum_per_window))
    return {
        "dataset": dataset_name,
        "run_id": run_dir.name,
        "windows": len(windows),
        "unsafe_windows": int(len(unsafe)),
        "unsafe_fraction": float(len(unsafe) / len(windows)),
        "minimum_decoded_clearance_m": float(minimum_per_window[worst_index]),
        "worst_window": list(windows[worst_index]),
        "fit_rmse_m": float(np.sqrt(np.mean(fit_error**2))),
        "fit_error_p99_m": float(np.percentile(fit_error, 99)),
        "maximum_endpoint_error_m": float(
            np.max(np.linalg.norm(decoded[:, -1] - canonical_array[:, -1], axis=1))
        ),
    }


def audit_dataset_root(
    dataset_root: Path,
    *,
    min_gap: int = 5,
    max_gap: int = 42,
    minimum_clearance_m: float = 0.10,
) -> dict[str, object]:
    codec = PlanarBSplineCodec(
        num_control_points=12, degree=3, num_path_points=64
    ).eval()
    runs = []
    for dataset_dir in sorted(dataset_root.glob("dataset_hssd_*")):
        grid = _grid_from_npz(dataset_dir / "navigation_grid.npz")
        for run_dir in sorted(dataset_dir.glob("run_*")):
            runs.append(
                _audit_run(
                    run_dir,
                    dataset_dir.name,
                    grid,
                    codec,
                    min_gap,
                    max_gap,
                    minimum_clearance_m,
                )
            )
    if not runs:
        raise RuntimeError(f"no generated HSSD runs found in {dataset_root}")
    report = {
        "contract": {
            "min_gap": min_gap,
            "max_gap": max_gap,
            "num_control_points": codec.num_control_points,
            "num_path_points": codec.num_path_points,
            "clearance_sample_step_m": CONTINUOUS_CLEARANCE_STEP_M,
            "minimum_clearance_m": minimum_clearance_m,
        },
        "runs": runs,
        "total_windows": sum(int(run["windows"]) for run in runs),
        "unsafe_windows": sum(int(run["unsafe_windows"]) for run in runs),
        "minimum_decoded_clearance_m": min(
            float(run["minimum_decoded_clearance_m"]) for run in runs
        ),
        "maximum_fit_error_p99_m": max(float(run["fit_error_p99_m"]) for run in runs),
        "maximum_endpoint_error_m": max(
            float(run["maximum_endpoint_error_m"]) for run in runs
        ),
    }
    (dataset_root / "codec_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args(argv)
    report = audit_dataset_root(args.dataset_root.expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["unsafe_windows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
