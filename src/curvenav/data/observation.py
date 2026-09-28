"""Calibrated sensor-clock history; snapshots are independent of inference cadence."""

import numpy as np
from curvenav.config import DataConfig
from curvenav.data.obstacle_memory import ObstacleMemory
from curvenav.data.depth import PinholeIntrinsics, preprocess_depth
from curvenav.data.history import (
    OBSERVATION_PERIOD_S,
    ObservationHistory,
    validate_transform,
)

DEPTH_CONTEXT_FIELDS = frozenset(
    {
        "depth",
        "camera_intrinsics",
        "camera_to_body",
        "observation_to_current",
        "observation_age_s",
        "observation_valid",
        "obstacle_memory",
    }
)


class DepthContextBuffer:
    def __init__(self, data: DataConfig, obstacle_memory_factory=ObstacleMemory):
        self.data = data
        self.obstacle_memory_factory = obstacle_memory_factory
        self.histories = []
        self.frames = []
        self.sequence = 0

    def reset(self, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.histories = [ObservationHistory() for _ in range(batch_size)]
        self.frames = [{} for _ in range(batch_size)]
        horizon_m = self.data.future_steps * self.data.expert_waypoint_spacing_m
        self.obstacle_memories = [
            self.obstacle_memory_factory(horizon_m, self.data.max_depth_m, self.data.robot_geometry)
            for _ in range(batch_size)
        ]
        self.sequence = 0

    def reset_env(self, env_id: int):
        self.histories[env_id] = ObservationHistory()
        self.frames[env_id].clear()
        self.obstacle_memories[env_id] = self.obstacle_memory_factory(
            self.data.future_steps * self.data.expert_waypoint_spacing_m,
            self.data.max_depth_m, self.data.robot_geometry,
        )

    def update(
        self, depth_m, body_to_world, camera_intrinsics, camera_to_body, timestamps
    ):
        batch = len(self.histories)
        if depth_m.ndim != 4 or depth_m.shape[0] != batch or depth_m.shape[-1] != 1:
            raise ValueError("depth_m must be [B,H,W,1]")
        for name, value, shape in (
            ("body_to_world", body_to_world, (batch, 4, 4)),
            ("camera_to_body", camera_to_body, (batch, 4, 4)),
            ("camera_intrinsics", camera_intrinsics, (batch, 3, 3)),
            ("timestamps", timestamps, (batch,)),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        validate_transform(body_to_world)
        validate_transform(camera_to_body)
        # The planner's body frame is gravity-aligned (x forward, y left, z up).
        if not np.allclose(body_to_world[:, :3, 2], [0, 0, 1], atol=1e-4, rtol=0):
            raise ValueError("body_to_world must use a gravity-aligned planning frame")
        if not np.isfinite(timestamps).all() or any(
            timestamp <= history.last_time
            for timestamp, history in zip(timestamps, self.histories, strict=True)
        ):
            raise ValueError(
                "observation timestamps must increase strictly within an episode"
            )
        # Isaac timestamps accumulate in FP32; 0.1 ms covers that clock rounding.
        if any(
            np.isfinite(history.last_time)
            and not np.isclose(
                timestamp - history.last_time, OBSERVATION_PERIOD_S, rtol=0, atol=1e-4
            )
            for timestamp, history in zip(timestamps, self.histories, strict=True)
        ):
            raise ValueError(
                "depth history must capture every 10 Hz sensor observation"
            )
        intrinsics = [
            PinholeIntrinsics.from_matrix(
                k, width=depth_m.shape[2], height=depth_m.shape[1]
            )
            for k in camera_intrinsics
        ]
        collected = {name: [] for name in DEPTH_CONTEXT_FIELDS}
        for env in range(batch):
            depth, intrinsic = preprocess_depth(
                depth_m[env, ..., 0],
                source_intrinsics=intrinsics[env],
                maximum_m=self.data.max_depth_m,
                height=self.data.image_height,
                width=self.data.image_width,
            )
            # Match the immutable training bank's depth quantization exactly.
            self.frames[env][self.sequence] = (
                depth.astype(np.float16),
                intrinsic,
                camera_to_body[env].copy(),
            )
            indices, transform, age, valid = self.histories[env].update(
                self.sequence, body_to_world[env], float(timestamps[env])
            )
            selected = [self.frames[env][int(index)] for index in indices]
            collected["depth"].append(np.stack([x[0] for x in selected])[:, None])
            collected["camera_intrinsics"].append(np.stack([x[1] for x in selected]))
            collected["camera_to_body"].append(
                np.stack([x[2] for x in selected]).astype(np.float32)
            )
            collected["observation_to_current"].append(transform.astype(np.float32))
            collected["observation_age_s"].append(age)
            collected["observation_valid"].append(valid)
            collected["obstacle_memory"].append(self.obstacle_memories[env].update(
                depth.astype(np.float16), intrinsic, camera_to_body[env], body_to_world[env],
            ))
            retained = {x[0] for x in self.histories[env].frames}
            self.frames[env] = {
                index: value
                for index, value in self.frames[env].items()
                if index in retained
            }
        self.sequence += 1
        return {name: np.stack(value) for name, value in collected.items()}
