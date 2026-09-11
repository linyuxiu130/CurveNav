"""Compact per-episode traces for closed-loop navigation diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np


TRACE_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class PlanRecord:
    """One policy request and the MPC solution produced from it."""

    env_id: int
    version: int
    generation: int
    point_goal: np.ndarray
    robot_position: np.ndarray
    robot_quaternion: np.ndarray
    trajectory: np.ndarray
    controls: np.ndarray
    predicted_states: np.ndarray
    desired_speed: np.ndarray
    maximum_curvature: np.ndarray
    policy_seconds: float
    mpc_seconds: float


@dataclass
class _EpisodeBuffer:
    episode_idx: int
    initial_goal_distance: float
    step_time: list[float] = field(default_factory=list)
    robot_position: list[np.ndarray] = field(default_factory=list)
    robot_quaternion: list[np.ndarray] = field(default_factory=list)
    point_goal: list[np.ndarray] = field(default_factory=list)
    planar_speed: list[float] = field(default_factory=list)
    mpc_command: list[np.ndarray] = field(default_factory=list)
    low_level_action: list[np.ndarray] = field(default_factory=list)
    plan_version: list[int] = field(default_factory=list)
    plan_action_index: list[int] = field(default_factory=list)
    plan_step: list[int] = field(default_factory=list)
    plan_records: list[PlanRecord] = field(default_factory=list)


class TrajectoryTraceWriter:
    """Write one durable NPZ file when each episode terminates."""

    def __init__(self, root: Path, num_envs: int) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._episodes: list[_EpisodeBuffer | None] = [None] * num_envs

    def start_episode(
        self,
        env_id: int,
        episode_idx: int,
        initial_goal_distance: float,
    ) -> None:
        self._episodes[env_id] = _EpisodeBuffer(
            episode_idx=episode_idx,
            initial_goal_distance=float(initial_goal_distance),
        )

    def record_step(
        self,
        env_id: int,
        time_seconds: float,
        robot_position: np.ndarray,
        robot_quaternion: np.ndarray,
        point_goal: np.ndarray,
        planar_speed: float,
        mpc_command: np.ndarray,
        low_level_action: np.ndarray,
        plan_version: int,
        plan_action_index: int,
    ) -> None:
        episode = self._episodes[env_id]
        if episode is None:
            return
        episode.step_time.append(float(time_seconds))
        episode.robot_position.append(np.asarray(robot_position, dtype=np.float32))
        episode.robot_quaternion.append(np.asarray(robot_quaternion, dtype=np.float32))
        episode.point_goal.append(np.asarray(point_goal, dtype=np.float32))
        episode.planar_speed.append(float(planar_speed))
        episode.mpc_command.append(np.asarray(mpc_command, dtype=np.float32))
        episode.low_level_action.append(np.asarray(low_level_action, dtype=np.float32))
        episode.plan_version.append(int(plan_version))
        episode.plan_action_index.append(int(plan_action_index))

    def record_plan(self, env_id: int, step: int, record: PlanRecord) -> None:
        episode = self._episodes[env_id]
        if episode is None:
            return
        episode.plan_step.append(int(step))
        episode.plan_records.append(record)

    def finish_episode(
        self,
        env_id: int,
        success: float,
        path_length: float,
        elapsed_seconds: float,
        final_robot_position: np.ndarray,
        final_robot_quaternion: np.ndarray,
        final_point_goal: np.ndarray,
    ) -> Path:
        episode = self._episodes[env_id]
        if episode is None:
            raise RuntimeError("cannot finish an inactive trace episode")
        self._episodes[env_id] = None

        plans = episode.plan_records
        trajectories, trajectory_lengths = self._pad_first_axis(
            plans, "trajectory"
        )
        arrays = {
            "schema_version": np.asarray(TRACE_SCHEMA_VERSION, dtype=np.int16),
            "episode_idx": np.asarray(episode.episode_idx, dtype=np.int32),
            "success": np.asarray(success, dtype=np.float32),
            "termination": np.asarray("success" if success else "timeout"),
            "initial_goal_distance_m": np.asarray(
                episode.initial_goal_distance, dtype=np.float32
            ),
            "executed_path_length_m": np.asarray(path_length, dtype=np.float32),
            "elapsed_simulation_time_s": np.asarray(
                elapsed_seconds, dtype=np.float32
            ),
            "terminal_pre_step_robot_position_world_m": np.asarray(
                final_robot_position, dtype=np.float32
            ),
            "terminal_pre_step_robot_quaternion_xyzw": np.asarray(
                final_robot_quaternion, dtype=np.float32
            ),
            "terminal_pre_step_point_goal_robot_m": np.asarray(
                final_point_goal, dtype=np.float32
            ),
            "step_time_s": np.asarray(episode.step_time, dtype=np.float32),
            "step_robot_position_world_m": np.asarray(
                episode.robot_position, dtype=np.float32
            ),
            "step_robot_quaternion_xyzw": np.asarray(
                episode.robot_quaternion, dtype=np.float32
            ),
            "step_point_goal_robot_m": np.asarray(
                episode.point_goal, dtype=np.float32
            ),
            "step_planar_speed_mps": np.asarray(
                episode.planar_speed, dtype=np.float32
            ),
            "step_mpc_command": np.asarray(episode.mpc_command, dtype=np.float32),
            "step_low_level_action": np.asarray(
                episode.low_level_action, dtype=np.float32
            ),
            "step_plan_version": np.asarray(episode.plan_version, dtype=np.int32),
            "step_plan_action_index": np.asarray(
                episode.plan_action_index, dtype=np.int16
            ),
            "plan_step": np.asarray(episode.plan_step, dtype=np.int32),
            "plan_version": np.asarray(
                [plan.version for plan in plans], dtype=np.int32
            ),
            "plan_point_goal_robot_m": self._stack(plans, "point_goal"),
            "plan_robot_position_world_m": self._stack(plans, "robot_position"),
            "plan_robot_quaternion_xyzw": self._stack(plans, "robot_quaternion"),
            "plan_local_trajectory": trajectories,
            "plan_local_trajectory_length": trajectory_lengths,
            "plan_mpc_controls": self._stack(plans, "controls"),
            "plan_mpc_predicted_states": self._stack(plans, "predicted_states"),
            "plan_mpc_desired_speed_mps": self._stack(plans, "desired_speed"),
            "plan_maximum_curvature_inv_m": self._stack(
                plans, "maximum_curvature"
            ),
            "plan_policy_seconds": np.asarray(
                [plan.policy_seconds for plan in plans], dtype=np.float32
            ),
            "plan_mpc_seconds": np.asarray(
                [plan.mpc_seconds for plan in plans], dtype=np.float32
            ),
        }
        path = self.root / f"episode-{episode.episode_idx:03d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
        return path

    @staticmethod
    def _stack(records: list[PlanRecord], field_name: str) -> np.ndarray:
        if not records:
            return np.empty((0,), dtype=np.float32)
        return np.stack([
            np.asarray(getattr(record, field_name), dtype=np.float32)
            for record in records
        ])

    @staticmethod
    def _pad_first_axis(
        records: list[PlanRecord], field_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a numeric variable-length sequence as padding plus exact lengths."""
        if not records:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        sequences = [
            np.asarray(getattr(record, field_name), dtype=np.float32)
            for record in records
        ]
        if any(sequence.ndim < 1 for sequence in sequences):
            raise ValueError(f"{field_name} must have a sequence axis")
        trailing_shape = sequences[0].shape[1:]
        if any(sequence.shape[1:] != trailing_shape for sequence in sequences):
            raise ValueError(f"{field_name} has inconsistent element shapes")
        lengths = np.asarray(
            [sequence.shape[0] for sequence in sequences], dtype=np.int32
        )
        padded = np.zeros(
            (len(sequences), int(lengths.max(initial=0)), *trailing_shape),
            dtype=np.float32,
        )
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.shape[0]] = sequence
        return padded, lengths
