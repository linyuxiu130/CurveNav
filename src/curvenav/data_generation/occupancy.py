"""Navigation point-cloud loading and conservative 2-D grid construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """Read XYZ from an ASCII or binary-little-endian scalar-vertex PLY."""

    path = Path(path)
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt: str | None = None
        vertex_count: int | None = None
        vertex_properties: list[tuple[str, str]] = []
        current_element: str | None = None
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"PLY header has no end_header: {path}")
            line = raw.decode("ascii").strip()
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("list-valued vertex PLY properties are unsupported")
                vertex_properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break

        if fmt is None or vertex_count is None:
            raise ValueError(f"PLY is missing format or vertex count: {path}")
        names = [name for name, _ in vertex_properties]
        missing = {"x", "y", "z"}.difference(names)
        if missing:
            raise ValueError(f"PLY vertex is missing coordinates: {sorted(missing)}")

        if fmt == "binary_little_endian":
            try:
                dtype = np.dtype(
                    [(name, _PLY_DTYPES[property_type]) for name, property_type in vertex_properties]
                )
            except KeyError as error:
                raise ValueError(f"unsupported PLY scalar type: {error.args[0]}") from error
            vertices = np.fromfile(stream, dtype=dtype, count=vertex_count)
            if len(vertices) != vertex_count:
                raise ValueError(f"truncated PLY vertex block: {path}")
            xyz = np.column_stack([vertices[axis] for axis in "xyz"])
        elif fmt == "ascii":
            values = np.loadtxt(stream, max_rows=vertex_count, ndmin=2)
            if values.shape[0] != vertex_count:
                raise ValueError(f"truncated PLY vertex block: {path}")
            xyz = values[:, [names.index(axis) for axis in "xyz"]]
        else:
            raise ValueError(f"unsupported PLY format {fmt!r}; expected little-endian or ASCII")

    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.size == 0:
        raise ValueError(f"PLY has no finite XYZ points: {path}")
    return xyz


@dataclass(frozen=True)
class NavigationGrid:
    """A center-safe free mask and its additional-clearance field.

    ``navigable.ply`` is treated as a set of already footprint-safe robot
    center positions.  Clearance therefore measures distance to the boundary
    of that safe set; it must not be interpreted as raw obstacle ESDF.
    """

    free: np.ndarray
    clearance_m: np.ndarray
    origin_xy: np.ndarray
    cell_size_m: float

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        return np.floor((points - self.origin_xy) / self.cell_size_m).astype(np.int32)

    def grid_to_world(self, indices: np.ndarray) -> np.ndarray:
        cells = np.asarray(indices, dtype=np.float64)
        return self.origin_xy + (cells + 0.5) * self.cell_size_m

    def in_bounds(self, index: np.ndarray | tuple[int, int]) -> bool:
        x, y = int(index[0]), int(index[1])
        return 0 <= x < self.free.shape[0] and 0 <= y < self.free.shape[1]

    def snap_to_free(self, xy: np.ndarray, max_distance_m: float = 0.25) -> np.ndarray:
        index = self.world_to_grid(np.asarray(xy, dtype=np.float64))
        if self.in_bounds(index) and self.free[tuple(index)]:
            return index
        radius = max(1, int(np.ceil(max_distance_m / self.cell_size_m)))
        x0, y0 = int(index[0]), int(index[1])
        best: tuple[float, int, int] | None = None
        for x in range(max(0, x0 - radius), min(self.free.shape[0], x0 + radius + 1)):
            for y in range(max(0, y0 - radius), min(self.free.shape[1], y0 + radius + 1)):
                if not self.free[x, y]:
                    continue
                distance = float(np.linalg.norm(self.grid_to_world(np.array([x, y])) - xy))
                if distance <= max_distance_m and (best is None or distance < best[0]):
                    best = (distance, x, y)
        if best is None:
            raise ValueError(f"no navigable cell within {max_distance_m:.2f} m of {xy.tolist()}")
        return np.array(best[1:], dtype=np.int32)

    def sample_clearance(self, path_xy: np.ndarray) -> np.ndarray:
        indices = self.world_to_grid(path_xy)
        result = np.full(len(indices), -np.inf, dtype=np.float64)
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.free.shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.free.shape[1])
        )
        valid_positions = np.flatnonzero(valid)
        if len(valid_positions):
            valid_indices = indices[valid]
            is_free = self.free[valid_indices[:, 0], valid_indices[:, 1]]
            free_positions = valid_positions[is_free]
            free_indices = valid_indices[is_free]
            result[free_positions] = self.clearance_m[
                free_indices[:, 0], free_indices[:, 1]
            ]
        return result

    def path_is_safe(
        self,
        path_xy: np.ndarray,
        *,
        minimum_clearance_m: float,
        sample_step_m: float | None = None,
    ) -> bool:
        dense = resample_polyline(
            path_xy,
            spacing_m=sample_step_m or min(0.025, self.cell_size_m / 4.0),
        )
        # Uniform resampling need not pass through every polyline vertex.  A
        # single corner on an obstacle boundary must not disappear between
        # samples, so validate the union of original and dense points.
        clearance = self.sample_clearance(
            np.concatenate([np.asarray(path_xy, dtype=np.float64), dense], axis=0)
        )
        return bool(np.isfinite(clearance).all() and np.all(clearance + 1e-9 >= minimum_clearance_m))


def resample_polyline(path_xy: np.ndarray, spacing_m: float) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_xy must have shape [N, 2] with N >= 2")
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9])
    path = path[keep]
    if len(path) < 2:
        raise ValueError("path has zero length")
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    count = max(2, int(np.ceil(cumulative[-1] / spacing_m)) + 1)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, path[:, axis]) for axis in range(2)]
    )


def build_navigation_grid(
    navigable_ply: str | Path,
    *,
    cell_size_m: float = 0.1,
    fill_single_cell_holes: bool = True,
    border_cells: int = 1,
) -> NavigationGrid:
    """Build a conservative grid matching X-NavDP's 0.1 m metadata contract."""

    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    points = read_ply_xyz(navigable_ply)
    xy = points[:, :2]
    # Match X-NavDP: its grid origin is the exact navigable point-cloud minimum.
    origin = xy.min(axis=0)
    indices = np.floor((xy - origin) / cell_size_m + 1e-8).astype(np.int32)
    shape = tuple((indices.max(axis=0) + 1).tolist())
    free = np.zeros(shape, dtype=bool)
    free[indices[:, 0], indices[:, 1]] = True

    if fill_single_cell_holes:
        neighbors = convolve(free.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
        free |= (~free) & (neighbors >= 7)
    if border_cells > 0:
        border = min(border_cells, *free.shape)
        free[:border, :] = False
        free[-border:, :] = False
        free[:, :border] = False
        free[:, -border:] = False
    if not free.any():
        raise ValueError(f"navigation grid is empty after filtering: {navigable_ply}")

    padded = np.pad(free, 1, constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * cell_size_m
    clearance[~free] = 0.0
    return NavigationGrid(
        free=free,
        clearance_m=clearance.astype(np.float32),
        origin_xy=origin.astype(np.float64),
        cell_size_m=float(cell_size_m),
    )
