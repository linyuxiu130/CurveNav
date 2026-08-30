"""Stateful CurveNav inference with a metric-spaced depth context."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import numpy as np
import torch

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data.depth import BENCHMARK_INTRINSICS, preprocess_metric_depth
from curvenav.factory import build_policy
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.types import PolicyCondition


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
    return config, policy


def _yaw_from_xyzw(quaternions: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class DepthContextBuffer:
    """Select four observations by traveled distance, including the current frame."""

    def __init__(self, config: CurveNavConfig) -> None:
        data = config.data
        self.offsets_m = (
            np.arange(data.observation_frames - 1, -1, -1, dtype=np.float64)
            * data.frame_spacing_m
        )
        self.height = data.image_height
        self.width = data.image_width
        self.maximum_m = data.max_depth_m
        self.prune_margin_m = data.expert_waypoint_spacing_m
        self.samples: list[deque[tuple[float, np.ndarray, np.ndarray, float]]] = []
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

    def update(
        self,
        depth_m: np.ndarray,
        positions_xy: np.ndarray,
        yaw_rad: np.ndarray,
    ) -> "DepthContext":
        observations = []
        observation_transforms = []
        validities = []
        for env_id, (raw_frame, position, current_yaw) in enumerate(
            zip(depth_m, positions_xy, yaw_rad, strict=True)
        ):
            previous = self.previous_positions[env_id]
            if np.isfinite(previous).all():
                self.distances[env_id] += float(np.linalg.norm(position - previous))
            self.previous_positions[env_id] = position
            frame = preprocess_metric_depth(
                raw_frame[..., 0],
                source_intrinsics=BENCHMARK_INTRINSICS,
                maximum_m=self.maximum_m,
            )
            samples = self.samples[env_id]
            samples.append(
                (
                    self.distances[env_id],
                    frame,
                    np.asarray(position, dtype=np.float32).copy(),
                    float(current_yaw),
                )
            )
            oldest = self.distances[env_id] - self.offsets_m[0] - self.prune_margin_m
            while len(samples) > 1 and samples[1][0] < oldest:
                samples.popleft()
            distances = np.fromiter((item[0] for item in samples), dtype=np.float64)
            frames = tuple(item[1] for item in samples)
            positions = tuple(item[2] for item in samples)
            yaws = tuple(item[3] for item in samples)
            targets = self.distances[env_id] - self.offsets_m
            reverse_indices = np.abs(distances[::-1, None] - targets[None]).argmin(
                axis=0
            )
            indices = len(distances) - 1 - reverse_indices
            valid = np.ones(len(indices), dtype=np.bool_)
            valid[:-1] = indices[:-1] != indices[1:]
            valid[-1] = True
            observations.append(np.stack([frames[index] for index in indices])[:, None])
            validities.append(valid)
            selected_positions = np.stack([positions[index] for index in indices])
            delta_world = selected_positions - position[None]
            cosine, sine = np.cos(current_yaw), np.sin(current_yaw)
            local_xy = np.stack(
                (
                    cosine * delta_world[:, 0] + sine * delta_world[:, 1],
                    -sine * delta_world[:, 0] + cosine * delta_world[:, 1],
                ),
                axis=-1,
            )
            delta_yaw = np.asarray([yaws[index] for index in indices]) - current_yaw
            observation_transforms.append(
                np.column_stack(
                    (local_xy, np.sin(delta_yaw), np.cos(delta_yaw))
                ).astype(np.float32)
            )
        return DepthContext(
            depth=np.stack(observations),
            observation_to_current=np.stack(observation_transforms),
            observation_valid=np.stack(validities),
        )


@dataclass(frozen=True)
class DepthContext:
    depth: np.ndarray
    observation_to_current: np.ndarray
    observation_valid: np.ndarray


@dataclass(frozen=True)
class RuntimePrediction:
    path: np.ndarray


class CurveNavRuntime:
    """Batched policy state with one spatial observation timeline per environment."""

    def __init__(
        self,
        config: CurveNavConfig,
        policy,
        device: str,
    ) -> None:
        self.config = config
        self.policy = policy
        self.device = torch.device(device)
        self.context_buffer = DepthContextBuffer(config)
        self.batch_size = 0
        self.requests = 0
        self.request_seconds = 0.0

    def _warmup_policy(self, batch_size: int) -> None:
        """Initialize the actual batch execution before the episode starts."""
        observation_to_current = torch.zeros(
            batch_size,
            self.config.data.observation_frames,
            4,
            device=self.device,
        )
        observation_to_current[..., 3] = 1.0
        condition = PolicyCondition(
            depth=torch.zeros(
                batch_size,
                self.config.data.observation_frames,
                1,
                self.config.data.image_height,
                self.config.data.image_width,
                device=self.device,
            ),
            point_goal=torch.zeros(batch_size, 2, device=self.device),
            observation_to_current=observation_to_current,
            observation_valid=torch.ones(
                batch_size,
                self.config.data.observation_frames,
                dtype=torch.bool,
                device=self.device,
            ),
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ),
        ):
            self.policy.sample(condition)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.context_buffer.reset(batch_size)
        self._warmup_policy(batch_size)

    def reset_env(self, env_id: int) -> None:
        self.context_buffer.reset_env(env_id)

    def step(
        self,
        point_goals: np.ndarray,
        depth_m: np.ndarray,
        positions: np.ndarray,
        quaternions: np.ndarray,
    ) -> RuntimePrediction:
        point_goals = np.asarray(point_goals, dtype=np.float32)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        positions = np.asarray(positions, dtype=np.float32)
        quaternions = np.asarray(quaternions, dtype=np.float32)
        if point_goals.shape != (self.batch_size, 2):
            raise ValueError(f"point_goal must have shape [{self.batch_size},2]")
        if (
            depth_m.ndim != 4
            or depth_m.shape[0] != self.batch_size
            or depth_m.shape[-1] != 1
        ):
            raise ValueError(f"depth must have shape [{self.batch_size},H,W,1]")
        if positions.shape != (self.batch_size, 3):
            raise ValueError(f"robot_pos must have shape [{self.batch_size},3]")
        if quaternions.shape != (self.batch_size, 4):
            raise ValueError(f"robot_quat must have shape [{self.batch_size},4]")

        context = self.context_buffer.update(
            depth_m,
            positions[:, :2],
            _yaw_from_xyzw(quaternions),
        )
        condition = PolicyCondition(
            depth=torch.from_numpy(context.depth).to(self.device),
            point_goal=torch.from_numpy(point_goals).to(self.device),
            observation_to_current=torch.from_numpy(context.observation_to_current).to(
                self.device
            ),
            observation_valid=torch.from_numpy(context.observation_valid).to(
                self.device
            ),
        )
        started = time.perf_counter()
        amp = self.device.type == "cuda"
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=amp
            ),
        ):
            prediction = self.policy.sample(condition)
        self.requests += 1
        self.request_seconds += time.perf_counter() - started
        future_path_xy = prediction.path[:, 1:].float().cpu().numpy()
        future_path = np.concatenate(
            [
                future_path_xy,
                np.zeros((*future_path_xy.shape[:-1], 1), dtype=np.float32),
            ],
            axis=-1,
        )
        return RuntimePrediction(path=future_path)
