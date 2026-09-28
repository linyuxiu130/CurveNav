"""Exact eight-neighbour grid distances without materializing a sparse graph."""

import heapq
import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def grid_distances(free: np.ndarray, cell_size: float, gx: int, gy: int) -> np.ndarray:
    height, width = free.shape
    distances = np.full((height, width), np.inf, dtype=np.float64)
    distances[gx, gy] = 0.0
    queue = [(0.0, gx * width + gy)]
    while queue:
        distance, node = heapq.heappop(queue)
        x, y = node // width, node % width
        if distance != distances[x, y]:
            continue
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= height or ny >= width or not free[nx, ny]:
                    continue
                if dx != 0 and dy != 0 and (not free[nx, y] or not free[x, ny]):
                    continue
                edge = (math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0) * cell_size
                candidate = distance + edge
                if candidate < distances[nx, ny]:
                    distances[nx, ny] = candidate
                    heapq.heappush(queue, (candidate, nx * width + ny))
    return distances
