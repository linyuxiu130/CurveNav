"""Contracts for deterministic trajectory-diagnostic aggregation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from navbench.trajectory_metrics import (
    _load_metric_rows,
    analyze_roots,
    write_analysis,
)
from navbench.trajectory_trace import PlanRecord, TrajectoryTraceWriter


class TrajectoryMetricsTests(unittest.TestCase):
    def test_parallel_completion_order_is_sorted_by_episode_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("success", "spl", "distance", "episode_idx"),
                )
                writer.writeheader()
                for episode_idx in (2, 0, 3, 1):
                    writer.writerow({
                        "success": 0.0,
                        "spl": 0.0,
                        "distance": 2.0,
                        "episode_idx": episode_idx,
                    })
            rows = _load_metric_rows(path)
            self.assertEqual(
                [int(row["episode_idx"]) for row in rows],
                [0, 1, 2, 3],
            )

    def test_trace_metrics_reconcile_with_official_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "00-curvenav-test"
            scene_root = root / "scenes" / "home" / "scene"
            trace_root = scene_root / "trajectory_traces"
            trace_root.mkdir(parents=True)
            (root / "run.json").write_text(json.dumps({"model": "curvenav"}))
            with (scene_root / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("success", "spl", "distance", "episode_idx"),
                )
                writer.writeheader()
                writer.writerow({
                    "success": 0.0,
                    "spl": 0.0,
                    "distance": 2.0,
                    "episode_idx": 0,
                })

            traces = TrajectoryTraceWriter(trace_root, num_envs=1)
            traces.start_episode(0, episode_idx=0, initial_goal_distance=2.0)
            for step, x_position in enumerate((0.0, 0.5)):
                traces.record_step(
                    env_id=0,
                    time_seconds=float(step),
                    robot_position=np.array([x_position, 0.0, 0.0]),
                    robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                    point_goal=np.array([2.0 - x_position, 0.0, 0.0]),
                    planar_speed=0.25,
                    mpc_command=np.array([0.25, 0.0]),
                    low_level_action=np.array([1.0, 1.0]),
                    plan_version=1,
                    plan_action_index=step,
                )
            traces.record_plan(0, 0, PlanRecord(
                env_id=0,
                version=1,
                generation=0,
                point_goal=np.array([2.0, 0.0, 0.0]),
                robot_position=np.array([0.0, 0.0, 0.0]),
                robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                trajectory=np.array([
                    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                ]),
                controls=np.array([[0.25, 0.0]]),
                predicted_states=np.zeros((2, 3)),
                desired_speed=np.asarray(0.25),
                maximum_curvature=np.asarray(0.0),
                policy_seconds=0.01,
                mpc_seconds=0.02,
            ))
            traces.record_plan(0, 1, PlanRecord(
                env_id=0,
                version=2,
                generation=1,
                point_goal=np.array([1.5, 0.0, 0.0]),
                robot_position=np.array([0.5, 0.0, 0.0]),
                robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                trajectory=np.array([[0.0, 0.0], [0.25, 0.0]]),
                controls=np.array([[0.25, 0.0]]),
                predicted_states=np.zeros((2, 3)),
                desired_speed=np.asarray(0.25),
                maximum_curvature=np.asarray(0.0),
                policy_seconds=0.01,
                mpc_seconds=0.02,
            ))
            traces.finish_episode(
                env_id=0,
                success=0.0,
                path_length=0.5,
                elapsed_seconds=2.0,
                final_robot_position=np.array([0.5, 0.0, 0.0]),
                final_robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                final_point_goal=np.array([1.5, 0.0, 0.0]),
            )

            episodes, summaries = analyze_roots([root])
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(summaries), 1)
            self.assertAlmostEqual(episodes[0]["goal_progress_fraction"], 0.25)
            self.assertAlmostEqual(episodes[0]["progress_efficiency"], 1.0)
            self.assertAlmostEqual(episodes[0]["path_tortuosity"], 1.0)
            self.assertAlmostEqual(episodes[0]["mean_plan_arc_length_m"], 0.625)
            self.assertAlmostEqual(episodes[0]["mean_total_plan_ms"], 30.0, places=5)
            output = Path(tmp) / "analysis"
            episode_path, summary_path = write_analysis([root], output)
            self.assertTrue(episode_path.is_file())
            self.assertTrue(summary_path.is_file())


if __name__ == "__main__":
    unittest.main()
