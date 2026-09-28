"""Causal local occupied voxels from calibrated depth, shared by train and deploy."""

import itertools
import math
import numpy as np

from curvenav.physical import (
    BODY_OBSTACLE_MIN_Z_M, ROBOT_COLLISION_TOP_Z_M, ROBOT_FOOTPRINT_RADIUS_M,
    EXTRA_CLEARANCE_M, PATH_CONFIGURATION_QUERY_SPACING_M,
)

GRID_SIZE = 64
VOXEL_M = PATH_CONFIGURATION_QUERY_SPACING_M
MEMORY_CONTRACT = "local_world_voxels_025m_all_corners_depth_clearing_v1"


def memory_grid_shape(horizon_m, radius_m=ROBOT_FOOTPRINT_RADIUS_M):
    resolution = 2 * horizon_m / (GRID_SIZE - 1)
    padding = math.ceil((radius_m + EXTRA_CLEARANCE_M) / resolution) + 1
    return GRID_SIZE + 2 * padding


def _rectangle_minimum(image, x0, y0, x1, y1):
    """Exact inclusive rectangle minima using four overlapping dyadic blocks."""
    height, width = y1 - y0 + 1, x1 - x0 + 1
    ky = np.floor(np.log2(height)).astype(np.int64)
    kx = np.floor(np.log2(width)).astype(np.int64)
    result = np.empty(len(x0), dtype=image.dtype)
    rows = image
    for j in range(int(ky.max(initial=0)) + 1):
        if j:
            offset = 1 << (j - 1)
            rows = np.minimum(rows[:-offset], rows[offset:])
        blocks = rows
        for i in range(int(kx.max(initial=0)) + 1):
            if i:
                offset = 1 << (i - 1)
                blocks = np.minimum(blocks[:, :-offset], blocks[:, offset:])
            selected = (ky == j) & (kx == i)
            left, top = x0[selected], y0[selected]
            right = x1[selected] - (1 << i) + 1
            bottom = y1[selected] - (1 << j) + 1
            result[selected] = np.minimum(
                np.minimum(blocks[top, left], blocks[top, right]),
                np.minimum(blocks[bottom, left], blocks[bottom, right]),
            )
    return result


class ObstacleMemory:
    """Retain measured occupied volumes until observed clear or outside the local map."""

    def __init__(self, horizon_m, max_depth_m, geometry=(ROBOT_FOOTPRINT_RADIUS_M, BODY_OBSTACLE_MIN_Z_M, ROBOT_COLLISION_TOP_Z_M)):
        self.radius, self.min_z, self.max_z = geometry
        self.horizon = horizon_m
        self.maximum = max_depth_m
        self.resolution = 2 * horizon_m / (GRID_SIZE - 1)
        self.size = memory_grid_shape(horizon_m, self.radius)
        self.padding = (self.size - GRID_SIZE) // 2
        self.extent = horizon_m + self.padding * self.resolution
        self.voxels = np.empty((0, 3), dtype=np.int64)
        self.corners = np.asarray(list(itertools.product((0., 1.), repeat=3)))

    def update(self, depth, intrinsic, camera_to_body, body_to_world):
        depth = np.asarray(depth, dtype=np.float32)
        camera_to_world = body_to_world @ camera_to_body
        if len(self.voxels):
            world = (self.voxels[:, None] + self.corners) * VOXEL_M
            local = (world - body_to_world[:3, 3]) @ body_to_world[:3, :3]
            nearby = (
                (local[..., :2].min(1) <= self.extent).all(1)
                & (local[..., :2].max(1) >= -self.extent).all(1)
            )
            self.voxels = self.voxels[nearby]
            world = world[nearby]
            optical = (world-camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
            z = optical[..., 2]
            # Behind-camera corners cannot clear a voxel; only project positive Z.
            positive_z = np.where(z > 0, z, 1.0)
            u = intrinsic[0,0]*optical[...,0]/positive_z + intrinsic[0,2] - .5
            v = intrinsic[1,1]*optical[...,1]/positive_z + intrinsic[1,2] - .5
            h, w = depth.shape
            inside = (z > 0) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)
            x0 = np.floor(np.clip(u, 0, w-1)).astype(np.int64)
            y0 = np.floor(np.clip(v, 0, h-1)).astype(np.int64)
            x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
            measured = _rectangle_minimum(
                depth, x0.min(1), y0.min(1), x1.max(1), y1.max(1)
            ) * self.maximum
            # The entire projected voxel rectangle must be observed free.
            # This tolerance covers normalized FP16 depth storage, not scene motion.
            clear = inside.all(1) & (
                measured > z.max(1) + self.maximum * np.finfo(np.float16).eps
            )
            self.voxels = self.voxels[~clear]

        y, x = np.meshgrid(np.arange(depth.shape[0])+.5, np.arange(depth.shape[1])+.5, indexing='ij')
        rays = np.stack(((x-intrinsic[0,2])/intrinsic[0,0], (y-intrinsic[1,2])/intrinsic[1,1], np.ones_like(x)), -1)
        points = (rays * (depth*self.maximum)[...,None]) @ camera_to_body[:3,:3].T + camera_to_body[:3,3]
        hit = (depth > 0) & (depth < 1)
        hit &= (points[...,2] >= self.min_z) & (points[...,2] <= self.max_z)
        hit &= (np.abs(points[...,:2]) <= self.extent).all(-1)
        world = points[hit] @ body_to_world[:3,:3].T + body_to_world[:3,3]
        new = np.floor(world / VOXEL_M).astype(np.int64)
        self.voxels = np.unique(np.concatenate((self.voxels, new)), axis=0)
        world = (self.voxels[:,None] + self.corners) * VOXEL_M
        local = (world-body_to_world[:3,3]) @ body_to_world[:3,:3]
        intersects = (
            (local[..., 2].min(1) <= self.max_z)
            & (local[..., 2].max(1) >= self.min_z)
        )
        local = local[intersects,...,:2]
        # Rasterize the entire projected voxel extent. No point replacement or
        # warp accumulation may shrink the occupied volume between observations.
        lo = np.floor((local.min(1)+self.horizon)/self.resolution + .5).astype(int) + self.padding
        hi = np.floor((local.max(1)+self.horizon)/self.resolution + .5).astype(int) + self.padding
        raster = np.zeros((self.size, self.size), dtype=bool)
        span = (hi-lo).max(axis=0, initial=0)
        for dx, dy in itertools.product(range(span[0]+1), range(span[1]+1)):
            index = lo + (dx, dy)
            valid = (index <= hi).all(1) & (index >= 0).all(1) & (index < self.size).all(1)
            raster[index[valid,1], index[valid,0]] = True
        return raster
