"""Stateful CurveNav inference with spatial history and depth-safe selection."""

from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np
import torch

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data.depth import preprocess_metric_depth
from curvenav.factory import build_policy
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.types import PolicyCondition


RAW_FRAME_SPACING_M = 0.15
BENCHMARK_FOCAL_PX = 1.4 / 1.88 * 640.0
BENCHMARK_CAMERA_HEIGHT_M = 0.30
ROBOT_CLEARANCE_M = 0.35
SELECTOR_GRID_M = 0.05


def load_policy(checkpoint_path, config_path, device: str):
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_policy_contract(checkpoint, config)
    policy = build_policy(config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy)
    policy.eval().to(device)
    policy.compile(mode="default")
    return config, policy


def _yaw_from_xyzw(quaternions: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SpatialDepthHistory:
    """Select depth history by traveled distance, matching the expert frame spacing."""

    def __init__(self, config: CurveNavConfig) -> None:
        data = config.data
        frame_step = data.frame_skip + 1
        self.offsets_m = np.arange(
            data.sequence_length - 1, -1, -1, dtype=np.float64
        ) * (frame_step * RAW_FRAME_SPACING_M)
        self.height = data.image_height
        self.width = data.image_width
        self.maximum_m = data.max_depth_m
        self.samples: list[deque[tuple[float, np.ndarray]]] = []
        self.distances = np.empty(0, dtype=np.float64)
        self.previous_positions = np.empty((0, 2), dtype=np.float32)

    def reset(self, batch_size: int) -> None:
        self.samples = [deque() for _ in range(batch_size)]
        self.distances = np.zeros(batch_size, dtype=np.float64)
        self.previous_positions = np.full((batch_size, 2), np.nan, dtype=np.float32)

    def reset_env(self, env_id: int) -> None:
        self.samples[env_id].clear()
        self.distances[env_id] = 0.0
        self.previous_positions[env_id] = np.nan

    def update(self, depth_m: np.ndarray, positions_xy: np.ndarray) -> np.ndarray:
        sequences = []
        for env_id, (raw_frame, position) in enumerate(zip(depth_m, positions_xy)):
            previous = self.previous_positions[env_id]
            if np.isfinite(previous).all():
                self.distances[env_id] += float(np.linalg.norm(position - previous))
            self.previous_positions[env_id] = position
            frame = preprocess_metric_depth(
                raw_frame[..., 0],
                height=self.height,
                width=self.width,
                maximum_m=self.maximum_m,
            )
            history = self.samples[env_id]
            history.append((self.distances[env_id], frame))
            oldest = self.distances[env_id] - self.offsets_m[0] - RAW_FRAME_SPACING_M
            while len(history) > 1 and history[1][0] < oldest:
                history.popleft()
            distances = np.fromiter((item[0] for item in history), dtype=np.float64)
            frames = tuple(item[1] for item in history)
            targets = self.distances[env_id] - self.offsets_m
            reverse_indices = np.abs(
                distances[::-1, None] - targets[None]
            ).argmin(axis=0)
            indices = len(distances) - 1 - reverse_indices
            sequences.append(np.stack([frames[index] for index in indices])[:, None])
        return np.stack(sequences)


class DepthSafetySelector:
    """Rank free-endpoint paths by safety, task progress, then efficiency."""

    def __init__(self, maximum_depth_m: float) -> None:
        self.maximum_depth_m = maximum_depth_m
        self.forward_cells = int(np.ceil(maximum_depth_m / SELECTOR_GRID_M))
        self.lateral_cells = 2 * self.forward_cells

    def _clearance_grid(self, depth_m: np.ndarray) -> np.ndarray:
        height, width = depth_m.shape
        rows = np.arange(0, height, 2)
        columns = np.arange(0, width, 2)
        vertical, horizontal = np.meshgrid(rows, columns, indexing="ij")
        depth = depth_m[vertical, horizontal]
        focal = BENCHMARK_FOCAL_PX * width / 640.0
        forward = depth
        lateral = -(horizontal - (width - 1) / 2.0) * depth / focal
        obstacle_height = (
            BENCHMARK_CAMERA_HEIGHT_M
            - (vertical - (height - 1) / 2.0) * depth / focal
        )
        valid = (
            np.isfinite(depth)
            & (depth > 0.05)
            & (depth < self.maximum_depth_m)
            & (obstacle_height > 0.08)
            & (obstacle_height < 0.70)
        )
        x_index = np.floor(forward[valid] / SELECTOR_GRID_M).astype(np.int32)
        y_index = np.floor(
            (lateral[valid] + self.maximum_depth_m) / SELECTOR_GRID_M
        ).astype(np.int32)
        in_grid = (
            (x_index >= 0)
            & (x_index < self.forward_cells)
            & (y_index >= 0)
            & (y_index < self.lateral_cells)
        )
        occupied = np.zeros(
            (self.forward_cells, self.lateral_cells), dtype=np.uint8
        )
        occupied[x_index[in_grid], y_index[in_grid]] = 1
        return cv2.distanceTransform(
            1 - occupied, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        ) * SELECTOR_GRID_M

    def _path_clearance(self, paths: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        grid = self._clearance_grid(depth_m)
        x_index = np.floor(paths[..., 0] / SELECTOR_GRID_M).astype(np.int32)
        y_index = np.floor(
            (paths[..., 1] + self.maximum_depth_m) / SELECTOR_GRID_M
        ).astype(np.int32)
        valid = (
            (x_index >= 0)
            & (x_index < self.forward_cells)
            & (y_index >= 0)
            & (y_index < self.lateral_cells)
        )
        clearance = np.zeros(paths.shape[:-1], dtype=np.float32)
        clearance[valid] = grid[x_index[valid], y_index[valid]]
        return clearance.min(axis=-1)

    def select_indices(
        self,
        candidates: np.ndarray,
        depth_m: np.ndarray,
        task_goals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidates, dtype=np.float32)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        task_goals = np.asarray(task_goals, dtype=np.float32)
        if candidates.ndim != 4 or candidates.shape[-1] != 2:
            raise ValueError("candidates must have shape [B,K,P,2]")
        batch_size, candidate_count = candidates.shape[:2]
        if depth_m.ndim != 4 or depth_m.shape[0] != batch_size or depth_m.shape[-1] != 1:
            raise ValueError("depth_m must have shape [B,H,W,1]")
        if task_goals.shape != (batch_size, 2):
            raise ValueError("task_goals must have shape [B,2]")

        selected_indices = []
        all_values = []
        for paths, frame, task_goal in zip(candidates, depth_m, task_goals):
            minimum_clearance = self._path_clearance(paths, frame[..., 0])
            length = np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(axis=1)
            bend = np.linalg.norm(np.diff(paths, n=2, axis=1), axis=2).sum(axis=1)
            efficiency = length + 0.1 * bend
            progress = np.linalg.norm(task_goal) - np.linalg.norm(
                task_goal[None] - paths[:, -1], axis=1
            )
            safe = minimum_clearance >= ROBOT_CLEARANCE_M
            if safe.any():
                safe_indices = np.flatnonzero(safe)
                unsafe_indices = np.flatnonzero(~safe)
                safe_order = safe_indices[
                    np.lexsort((efficiency[safe_indices], -progress[safe_indices]))
                ]
                unsafe_order = unsafe_indices[
                    np.lexsort(
                        (
                            efficiency[unsafe_indices],
                            -progress[unsafe_indices],
                            -minimum_clearance[unsafe_indices],
                        )
                    )
                ]
                order = np.concatenate((safe_order, unsafe_order))
            else:
                order = np.lexsort(
                    (efficiency, -progress, -minimum_clearance)
                )
            values = np.empty(candidate_count, dtype=np.float32)
            values[order] = np.arange(
                candidate_count, 0, -1, dtype=np.float32
            )
            selected_indices.append(int(order[0]))
            all_values.append(values)
        return (
            np.asarray(selected_indices, dtype=np.int64),
            np.stack(all_values),
        )

    def select(
        self,
        candidates: np.ndarray,
        depth_m: np.ndarray,
        task_goals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidates, dtype=np.float32)
        indices, values = self.select_indices(candidates, depth_m, task_goals)
        return candidates[np.arange(len(candidates)), indices], values


class CurveNavRuntime:
    """Batched policy state with one spatial observation timeline per environment."""

    def __init__(
        self,
        config: CurveNavConfig,
        policy,
        device: str,
        precision: str,
        num_samples: int = 8,
    ) -> None:
        self.config = config
        self.policy = policy
        self.device = torch.device(device)
        self.precision = precision
        self.num_samples = num_samples
        self.history = SpatialDepthHistory(config)
        self.selector = DepthSafetySelector(config.data.max_depth_m)
        self.previous_positions = np.empty((0, 3), dtype=np.float32)
        self.batch_size = 0
        self.requests = 0
        self.request_seconds = 0.0

    def reset(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.history.reset(batch_size)
        self.previous_positions = np.full((batch_size, 3), np.nan, dtype=np.float32)

    def reset_env(self, env_id: int) -> None:
        self.history.reset_env(env_id)
        self.previous_positions[env_id] = np.nan

    def _motion_condition(
        self, positions: np.ndarray, quaternions: np.ndarray
    ) -> np.ndarray:
        yaw = _yaw_from_xyzw(quaternions)
        motion = np.zeros((self.batch_size, 3), dtype=np.float32)
        for env_id in range(self.batch_size):
            previous = self.previous_positions[env_id]
            if np.isfinite(previous).all():
                delta = positions[env_id, :2] - previous[:2]
                cosine, sine = np.cos(yaw[env_id]), np.sin(yaw[env_id])
                local = np.array(
                    [
                        cosine * delta[0] + sine * delta[1],
                        -sine * delta[0] + cosine * delta[1],
                    ],
                    dtype=np.float32,
                )
                magnitude = float(np.linalg.norm(local))
                if magnitude > 1e-6:
                    motion[env_id, :2] = local / magnitude
                    motion[env_id, 2] = 1.0
            self.previous_positions[env_id] = positions[env_id]
        return motion

    def step(
        self,
        goals_xy: np.ndarray,
        depth_m: np.ndarray,
        positions: np.ndarray,
        quaternions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        goals_xy = np.asarray(goals_xy, dtype=np.float32)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        positions = np.asarray(positions, dtype=np.float32)
        quaternions = np.asarray(quaternions, dtype=np.float32)
        if goals_xy.shape != (self.batch_size, 2):
            raise ValueError(f"point goal must have shape [{self.batch_size},2]")
        if depth_m.ndim != 4 or depth_m.shape[0] != self.batch_size or depth_m.shape[-1] != 1:
            raise ValueError(f"depth must have shape [{self.batch_size},H,W,1]")
        if positions.shape != (self.batch_size, 3):
            raise ValueError(f"robot_pos must have shape [{self.batch_size},3]")
        if quaternions.shape != (self.batch_size, 4):
            raise ValueError(f"robot_quat must have shape [{self.batch_size},4]")

        condition = PolicyCondition(
            depth=torch.from_numpy(
                self.history.update(depth_m, positions[:, :2])
            ).to(self.device),
            task_goal=torch.from_numpy(goals_xy).to(self.device),
            motion_context=torch.from_numpy(
                self._motion_condition(positions, quaternions)
            ).to(self.device),
        )
        started = time.perf_counter()
        amp = self.precision == "amp" and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=amp
        ):
            prediction = self.policy.sample(condition, num_samples=self.num_samples)
        self.requests += 1
        self.request_seconds += time.perf_counter() - started
        planar = prediction.dense_path.reshape(
            self.batch_size, self.num_samples, -1, 2
        ).float().cpu().numpy()
        trajectory, values = self.selector.select(planar, depth_m, goals_xy)
        candidates = np.concatenate(
            [planar, np.zeros((*planar.shape[:-1], 1), dtype=np.float32)], axis=-1
        )
        selected = np.concatenate(
            [trajectory, np.zeros((*trajectory.shape[:-1], 1), dtype=np.float32)],
            axis=-1,
        )
        return selected, candidates, values
