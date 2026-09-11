"""SanD PointGoal inference adapter for the benchmark's raw tensor path."""

from __future__ import annotations

import numpy as np
import torch

from sand_planner.agent.depth_processor import ArrayDepthProcessor as DepthProcessor
from sand_planner.config import InferenceConfig
from sand_planner.core.orchestrator import SandPlannerInference


class SandPlannerAgent:
    def __init__(
        self,
        image_intrinsic: np.ndarray,
        config: InferenceConfig | None = None,
        verbose: bool = False,
    ) -> None:
        self.image_intrinsic = np.asarray(image_intrinsic)
        self.config = config if config is not None else InferenceConfig()
        self.config.camera_fx = float(self.image_intrinsic[0, 0])
        self.config.camera_fy = float(self.image_intrinsic[1, 1])
        self.config.camera_ppx = float(self.image_intrinsic[0, 2])
        self.config.camera_ppy = float(self.image_intrinsic[1, 2])
        self.planner = SandPlannerInference(self.config, verbose=verbose)

    def reset(self, batch_size: int, _threshold: float) -> None:
        self.batch_size = int(batch_size)
        self.frame_counters = [0] * self.batch_size
        self.depth_caches = [DepthProcessor(self.config) for _ in range(self.batch_size)]
        self.temporal_states = [None] * self.batch_size
        self.planner.reset_environment()

    def reset_env(self, env_id: int) -> None:
        self.frame_counters[env_id] = 0
        self.depth_caches[env_id].clear_cache()
        self.temporal_states[env_id] = None

    def update_camera_config(self, image_intrinsic: np.ndarray) -> None:
        self.image_intrinsic = np.asarray(image_intrinsic)
        self.config.camera_fx = float(self.image_intrinsic[0, 0])
        self.config.camera_fy = float(self.image_intrinsic[1, 1])
        self.config.camera_ppx = float(self.image_intrinsic[0, 2])
        self.config.camera_ppy = float(self.image_intrinsic[1, 2])

    @staticmethod
    def process_pointgoal(goals: np.ndarray) -> np.ndarray:
        clipped = np.asarray(goals).copy()
        clipped[:, 0] = np.clip(clipped[:, 0], -3, 10)
        clipped[:, 1] = np.clip(clipped[:, 1], -10, 10)
        return clipped

    def _process_depth(
        self, depth: np.ndarray,
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        sequences = []
        originals = []
        for env_id, frame in enumerate(depth[:, :, :, 0]):
            cache = self.depth_caches[env_id]
            cache.add_frame_to_cache(frame, should_save=True)
            self.frame_counters[env_id] += 1
            sequences.append(cache.get_sequence_from_cache()[0])
            originals.append(
                np.clip(frame, 0, self.config.max_depth) / self.config.max_depth
            )
        return torch.stack(sequences, dim=0), originals

    def _restore_temporal_state(self, env_id: int) -> None:
        engine = self.planner.inference_engine
        state = self.temporal_states[env_id]
        if state is None:
            engine.reset_warm_start_cache()
            return
        (
            engine._prev_control_points,
            engine._warm_start_counter,
            engine._prev_initial_turn,
            engine._prev_best_trajectory,
            engine._prev_best_control_points,
            engine._executed_distance,
        ) = state

    def _save_temporal_state(self, env_id: int) -> None:
        engine = self.planner.inference_engine
        self.temporal_states[env_id] = (
            engine._prev_control_points,
            engine._warm_start_counter,
            engine._prev_initial_turn,
            engine._prev_best_trajectory,
            engine._prev_best_control_points,
            engine._executed_distance,
        )

    def step_pointgoal(
        self,
        goals: np.ndarray,
        images: np.ndarray,
        depths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        del images
        processed_goals = self.process_pointgoal(goals)
        depth_sequences, original_depths = self._process_depth(depths)
        outputs = []
        for env_id in range(self.batch_size):
            self._restore_temporal_state(env_id)
            self.config.target_position = processed_goals[env_id].tolist()
            self.planner.agent_original_depth = original_depths[env_id]
            results = self.planner.process_depth_arrays(
                depth_sequences[env_id:env_id + 1]
            )
            self._save_temporal_state(env_id)
            outputs.append(self._format_results(results))
        return self._stack_results(outputs)

    @staticmethod
    def _pad_trajectory(value: np.ndarray, length: int) -> np.ndarray:
        tail = np.repeat(value[..., -1:, :], length - value.shape[-2], axis=-2)
        return np.concatenate((value, tail), axis=-2)

    @classmethod
    def _stack_results(
        cls,
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, None]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        max_length = max(output[0].shape[1] for output in outputs)
        best = np.concatenate([
            cls._pad_trajectory(output[0], max_length) for output in outputs
        ], axis=0)
        candidates = np.concatenate([
            cls._pad_trajectory(output[1], max_length) for output in outputs
        ], axis=0)
        costs = np.concatenate([output[2] for output in outputs], axis=0)
        return best, candidates, costs, None

    def _format_results(
        self,
        results: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        sampled = results["sampled_trajectories"]
        ranked = results["evaluation_results"]["results"]
        order = [row["trajectory_id"] for row in ranked]
        values = np.asarray([row["total_cost"] for row in ranked])

        max_length = max(len(trajectory) for trajectory in sampled)
        padded = []
        for trajectory in sampled:
            tail = np.repeat(trajectory[-1:], max_length - len(trajectory), axis=0)
            padded.append(np.concatenate((trajectory, tail), axis=0))
        all_trajectories = np.stack([padded[index] for index in order], axis=0)
        best_trajectory = all_trajectories[0].copy()

        return (
            best_trajectory[np.newaxis],
            all_trajectories[np.newaxis],
            values[np.newaxis],
            None,
        )
