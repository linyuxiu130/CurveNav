"""Training-only obstacle-aware goal distances on immutable robot C-space."""

import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from curvenav.data.privileged import SourceConfigurationSpaceQuery


class GoalDistanceQuery:
    def __init__(self, source: SourceConfigurationSpaceQuery):
        self.source = source
        self.graphs: dict[int, csr_matrix] = {}
        self.fields: dict[tuple[int, int, int], np.ndarray] = {}

    def _graph(self, index: int) -> csr_matrix:
        if index not in self.graphs:
            grid = self.source.grids[index]
            free = grid.signed_clearance_m >= 0
            ids = np.arange(free.size).reshape(free.shape)
            rows, cols, lengths = [], [], []
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                x0, x1 = slice(0, free.shape[0] - dx), slice(dx, free.shape[0])
                y0 = slice(max(0, -dy), min(free.shape[1], free.shape[1] - dy))
                y1 = slice(max(0, dy), min(free.shape[1], free.shape[1] + dy))
                valid = free[x0, y0] & free[x1, y1]
                if dx and dy:
                    # Closed-cell contact, identical to the source path query:
                    # a diagonal touches both side cells at their shared corner.
                    valid &= free[x1, y0] & free[x0, y1]
                a, b = ids[x0, y0][valid], ids[x1, y1][valid]
                rows.extend((a, b))
                cols.extend((b, a))
                edge_length = np.full(len(a), np.hypot(dx, dy) * grid.cell_size_m)
                lengths.extend((edge_length, edge_length))
            self.graphs[index] = coo_matrix(
                (np.concatenate(lengths), (np.concatenate(rows), np.concatenate(cols))),
                shape=(free.size, free.size),
            ).tocsr()
        return self.graphs[index]

    def _field(self, index: int, cell: np.ndarray) -> np.ndarray:
        grid = self.source.grids[index]
        if np.any(cell < 0) or np.any(cell >= grid.signed_clearance_m.shape):
            raise ValueError("training mission goal lies outside source C-space")
        if grid.signed_clearance_m[tuple(cell)] < 0:
            raise ValueError("training mission goal is not traversable in source C-space")
        key = (index, *cell.tolist())
        if key not in self.fields:
            if os.environ.get("CURVENAV_GOAL_DISTANCE_BACKEND", "scipy") == "numba":
                from curvenav.data.grid_dijkstra import grid_distances
                distances = grid_distances(
                    grid.signed_clearance_m >= 0, grid.cell_size_m,
                    int(cell[0]), int(cell[1]),
                )
            else:
                node = np.ravel_multi_index(cell, grid.signed_clearance_m.shape)
                distances = dijkstra(self._graph(index), directed=True, indices=int(node))
            self.fields[key] = distances.astype(np.float32).reshape(
                grid.signed_clearance_m.shape
            )
        return self.fields[key]

    def query(self, index: int, goal_cell: np.ndarray, cells: np.ndarray) -> np.ndarray:
        grid = self.source.grids[index]
        field = self._field(index, goal_cell)
        values = np.full(cells.shape[:-1], np.inf, dtype=np.float32)
        inside = (cells >= 0).all(-1) & (cells < grid.signed_clearance_m.shape).all(-1)
        values[inside] = field[cells[inside, 0], cells[inside, 1]]
        return values
