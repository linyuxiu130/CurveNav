"""Minimal end-to-end contracts for the X-NavDP paper evaluation."""

import csv
import inspect
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flask import Flask
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def cli_test_environment(monkeypatch):
    """Plan tests use this interpreter, without a host-local simulator install."""
    monkeypatch.setenv("NAVBENCH_EVAL_PYTHON", sys.executable)
    monkeypatch.setenv("NAVBENCH_SERVER_PYTHON", sys.executable)
    monkeypatch.setenv("NAVBENCH_NUM_ENVS", "16")


def test_server_setup_preserves_existing_conda_environment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'conda-meta').mkdir()
    marker = tmp_path / 'keep'
    marker.write_text('existing environment')
    result = subprocess.run(
        ['bash', str(root / 'scripts/setup_envs.sh'), 'server'],
        env={**os.environ, 'NAVBENCH_SERVER_ENV': str(tmp_path)},
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert 'refusing to create a venv' in result.stderr
    assert marker.read_text() == 'existing environment'
    assert not (tmp_path / 'pyvenv.cfg').exists()


def test_evaluator_version_lock_defines_every_installer_version():
    root = Path(__file__).resolve().parents[1]
    installer = (root / 'scripts/setup_evaluator_env.sh').read_text()
    versions = dict(line.split('=', 1) for line in
                    (root / 'config/evaluator-versions.env').read_text().splitlines()
                    if line and not line.startswith('#'))
    required = set(re.findall(r'\$\{([A-Z_]+_VERSION)\}', installer))
    assert required <= versions.keys(), required - versions.keys()

from baselines.curvenav.curvenav_server import create_app as create_curvenav_app
from navbench.adapters import ModelArtifact
from navbench.client import _prepare_depth, pointgoal_step
from navbench.cli import (
    PolicyGpuPool,
    build_initial_targets,
    build_checkpoint_target,
    evaluator_command,
    evaluator_environment,
    materialize_scene_overlay,
    prepare_upstream_inputs,
    queued_artifact,
)
from navbench.evaluator_process import EVENT_PREFIX, EvaluatorProcess
from navbench.episodes import EPISODE_COLUMNS, sha256_file
from navbench.metrics import success_weighted_path_length
from navbench.policy_pool import PolicyPool
from navbench.protocol import policy_response, read_depth
from navbench.scene_evaluator import (
    TRACE_SAMPLE_INTERVAL_STEPS,
    trace_sample_env_ids,
    write_metrics,
)
from navbench.suite import invalid_episode_assets, invalid_navigation_assets, load_suite
from navbench.trajectory_trace import PlanRecord, TrajectoryTraceWriter
from scripts.check_xnavdp_runtime import REVISION, VERSIONS


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "suites/pointgoal-v2.json"
SCENE_ROOT = Path(os.environ.get(
    "NAVBENCH_SCENE_ROOT", str(ROOT / "assets/scenes")
))


class PaperContractTests(unittest.TestCase):
    def test_policy_gpu_pool_enforces_weighted_memory_capacity(self):
        pool = PolicyGpuPool([5, 6], slots_per_gpu=2)
        with pool.reserve([5], slots_per_server=2):
            self.assertEqual(pool.available, {5: 0, 6: 2})
        with pool.reserve([5, 5], slots_per_server=1):
            self.assertEqual(pool.available, {5: 0, 6: 2})
        self.assertEqual(pool.available, {5: 2, 6: 2})

    def test_curvenav_reset_returns_the_sensor_observation_contract(self):
        from curvenav.config import CurveNavConfig
        class Runtime:
            config = CurveNavConfig()

            def reset(self, batch_size):
                self.batch_size = batch_size

        runtime = Runtime()
        intrinsic = np.array(
            ((112.0, 0.0, 111.5), (0.0, 112.0, 62.5), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        response = create_curvenav_app(runtime).test_client().post(
            "/navigator_reset",
            json={"batch_size": 16, "intrinsic": intrinsic.tolist(), "seed": 1234},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.batch_size, 16)

    def test_evaluator_command_forwards_explicit_kit_settings(self):
        args = SimpleNamespace(
            eval_python="/runtime/python",
            suite={"splits": {"home": {"scenes": ["scene"]}}},
            cpu_threads_per_worker=8,
        )
        job = SimpleNamespace(split="home", name="scene")
        with patch.dict(os.environ, {
            "NAVBENCH_EVAL_KIT_ARGS":
                "--/rtx/verifyDriverVersion/enabled=false",
        }):
            command = evaluator_command(
                args, 12121, job, Path("/tmp/eval.yaml"), gpu=3,
            )
        self.assertEqual(
            command[-1], "--/rtx/verifyDriverVersion/enabled=false",
        )
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertIn("--/renderer/activeGpu=3", command)
        self.assertIn(
            "--/plugins/carb.tasking.plugin/threadCount=8", command,
        )
        portable_index = command.index("--portable-root")
        self.assertTrue(command[portable_index + 1].endswith("/kit/gpu_3"))

        args.xnavdp_root = Path("/runtime/xnavdp")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "ACADOS_SOURCE_DIR": tmp, "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }), patch("navbench.cli.cache_root", return_value=Path(tmp)):
            env = evaluator_environment(args, 3, Path(tmp) / "optix")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")

    def test_ready_bundle_binds_checkpoint_config_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "candidate.ready"
            (bundle / "configs").mkdir(parents=True)
            (bundle / "src" / "curvenav").mkdir(parents=True)
            (bundle / "checkpoint.pt").write_bytes(b"weights")
            (bundle / "artifact.json").write_text(json.dumps({
                "schema": "navbench-model-artifact-v1",
                "model": "curvenav",
            }))
            (bundle / "configs" / "base.yaml").write_text("model: {}\n")
            source = bundle / "src" / "curvenav" / "policy.py"
            source.write_text("VALUE = 1\n")
            model, artifact = queued_artifact(bundle)
            first = build_checkpoint_target(model, artifact, root / "run", 0)
            source.write_text("VALUE = 2\n")
            second = build_checkpoint_target(model, artifact, root / "run", 1)
        self.assertEqual(first.checkpoint_sha256, second.checkpoint_sha256)
        self.assertNotEqual(first.source_sha256, second.source_sha256)
        self.assertNotEqual(first.artifact_sha256, second.artifact_sha256)

    def test_ready_bundle_selects_model_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "navdp.ready"
            bundle.mkdir()
            (bundle / "checkpoint.pt").write_bytes(b"weights")
            (bundle / "artifact.json").write_text(json.dumps({
                "schema": "navbench-model-artifact-v1",
                "model": "navdp",
            }))
            model, artifact = queued_artifact(bundle)
        self.assertEqual(model, "navdp")
        self.assertIsNone(artifact.model_config)

    def test_model_adapter_is_part_of_artifact_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"shared weights")
            artifact = ModelArtifact(checkpoint)
            navdp = build_checkpoint_target("navdp", artifact, root / "run", 0)
            sand = build_checkpoint_target(
                "sandplanner", artifact, root / "run", 1
            )
        self.assertNotEqual(navdp.artifact_sha256, sand.artifact_sha256)
        self.assertEqual(navdp.run_root.parent.name, "navdp")
        self.assertEqual(sand.run_root.parent.name, "sandplanner")

    def test_fixed_model_set_preserves_declared_adapter_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sand = root / "sand.pt"
            navdp = root / "navdp.pt"
            xnavdp = root / "xnavdp.pt"
            for path in (sand, navdp, xnavdp):
                path.write_bytes(path.name.encode())
            args = SimpleNamespace(
                model="sandplanner",
                checkpoints=[sand],
                model_config=None,
                additional_artifacts=[
                    ("navdp", ModelArtifact(navdp)),
                    ("x-navdp", ModelArtifact(xnavdp)),
                ],
            )
            targets = build_initial_targets(args, root / "run")
        self.assertEqual(
            [target.model for target in targets],
            ["sandplanner", "navdp", "x-navdp"],
        )
        self.assertEqual(len({target.artifact_sha256 for target in targets}), 3)

    def test_trace_sampling_is_fixed_and_excludes_inactive_slots(self):
        self.assertEqual(TRACE_SAMPLE_INTERVAL_STEPS, 5)
        steps = np.asarray([0, 4, 5, 10], dtype=np.int32)
        self.assertEqual(
            trace_sample_env_ids([0, 1, None, 3], steps), [0, 3]
        )

    def test_policy_pool_preserves_batch_order_and_local_reset_ids(self):
        observation = {
            "pointgoal": np.arange(10, dtype=np.float32).reshape(5, 2),
            "rgb": np.zeros((5, 2, 2, 3), dtype=np.uint8),
            "depth": np.ones((5, 2, 2), dtype=np.float32),
            "robot_pos": np.zeros((5, 3), dtype=np.float32),
            "robot_quat": np.zeros((5, 4), dtype=np.float32),
        }

        def fake_step(*, pointgoal, port, **_unused):
            return pointgoal[:, None, :], None, None

        with (
            patch("navbench.policy_pool.pointgoal_step", side_effect=fake_step),
            patch("navbench.policy_pool.navigator_reset") as reset,
            patch("navbench.policy_pool.navigator_shutdown"),
        ):
            pool = PolicyPool([9000, 9001], batch_size=5)
            trajectory = pool.step(observation)
            pool.reset_env(4, sample_idx=9, scene_name="home")
            pool.close()

        np.testing.assert_array_equal(
            trajectory[:, 0], observation["pointgoal"]
        )
        reset.assert_called_once_with(
            env_id=2, port=9001, sample_idx=9, scene_name="home"
        )

    def test_policy_pool_reset_preserves_global_batch_coordinates(self):
        with (
            patch("navbench.policy_pool.navigator_reset") as reset,
            patch("navbench.policy_pool.navigator_shutdown"),
        ):
            pool = PolicyPool([9000, 9001], batch_size=5)
            pool.reset_all(np.eye(3), list(range(5)), "home")
            pool.close()

        self.assertEqual(reset.call_count, 2)
        first, second = reset.call_args_list
        self.assertEqual(first.kwargs["batch_size"], 2)
        self.assertEqual(first.kwargs["global_batch_size"], 5)
        self.assertEqual(first.kwargs["batch_start"], 0)
        self.assertEqual(second.kwargs["batch_size"], 3)
        self.assertEqual(second.kwargs["global_batch_size"], 5)
        self.assertEqual(second.kwargs["batch_start"], 2)

    def test_server_gpu_mapping_can_share_policy_only_gpus(self):
        from navbench.cli import parse_gpu_mapping

        self.assertEqual(parse_gpu_mapping("5,6,7,5,6"), [5, 6, 7, 5, 6])

    def test_trajectory_trace_is_compact_numeric_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = TrajectoryTraceWriter(Path(tmp), num_envs=1)
            writer.start_episode(0, episode_idx=3, initial_goal_distance=2.0)
            writer.record_step(
                env_id=0,
                time_seconds=0.0,
                robot_position=np.array([1.0, 2.0, 0.0]),
                robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                point_goal=np.array([2.0, 0.0, 0.0]),
                planar_speed=0.1,
                mpc_command=np.array([0.1, 0.0]),
                low_level_action=np.array([1.0, 1.0]),
                plan_version=1,
                plan_action_index=0,
            )
            writer.record_plan(0, 0, PlanRecord(
                env_id=0,
                version=1,
                generation=0,
                point_goal=np.array([2.0, 0.0, 0.0]),
                robot_position=np.array([1.0, 2.0, 0.0]),
                robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                trajectory=np.zeros((5, 3)),
                controls=np.zeros((3, 2)),
                predicted_states=np.zeros((4, 3)),
                desired_speed=np.asarray(0.2),
                maximum_curvature=np.asarray(0.5),
                policy_seconds=0.01,
                mpc_seconds=0.02,
            ))
            writer.record_plan(0, 1, PlanRecord(
                env_id=0,
                version=2,
                generation=1,
                point_goal=np.array([1.9, 0.0, 0.0]),
                robot_position=np.array([1.1, 2.0, 0.0]),
                robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                trajectory=np.zeros((3, 3)),
                controls=np.zeros((3, 2)),
                predicted_states=np.zeros((4, 3)),
                desired_speed=np.asarray(0.2),
                maximum_curvature=np.asarray(0.5),
                policy_seconds=0.01,
                mpc_seconds=0.02,
            ))
            path = writer.finish_episode(
                env_id=0,
                success=0.0,
                path_length=0.1,
                elapsed_seconds=1.25,
                final_robot_position=np.array([1.1, 2.0, 0.0]),
                final_robot_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                final_point_goal=np.array([1.9, 0.0, 0.0]),
            )
            with np.load(path, allow_pickle=False) as trace:
                self.assertEqual(trace["termination"].item(), "timeout")
                self.assertEqual(trace["step_robot_position_world_m"].shape, (1, 3))
                self.assertEqual(trace["plan_local_trajectory"].shape, (2, 5, 3))
                np.testing.assert_array_equal(
                    trace["plan_local_trajectory_length"], [5, 3]
                )
                self.assertEqual(trace["elapsed_simulation_time_s"].item(), 1.25)
                self.assertEqual(trace["plan_mpc_controls"].shape, (2, 3, 2))

    def test_evaluator_environment_lock(self):
        self.assertEqual(VERSIONS["PYTHON_VERSION"], "3.11.15")
        self.assertEqual(VERSIONS["SETUPTOOLS_VERSION"], "80.9.0")
        self.assertEqual(VERSIONS["CMAKE_VERSION"], "3.30.5")
        self.assertEqual(VERSIONS["ISAACSIM_CLICK_VERSION"], "8.1.7")
        self.assertEqual(VERSIONS["ISAACSIM_TYPING_EXTENSIONS_VERSION"], "4.12.2")
        self.assertEqual(VERSIONS["PYTORCH_CUDA_TAG"], "cu126")
        self.assertEqual(VERSIONS["ISAACSIM_DIST_VERSION"], "5.0.0.0")
        self.assertEqual(VERSIONS["ISAACLAB_VERSION"], "0.46.2")
        self.assertEqual(VERSIONS["ISAACLAB_RL_VERSION"], "0.4.0")
        self.assertEqual(VERSIONS["RSL_RL_VERSION"], "3.0.1")
        self.assertEqual(VERSIONS["TENSORDICT_VERSION"], "0.7.2")
        self.assertEqual(VERSIONS["ACADOS_TEMPLATE_VERSION"], "0.5.1")
        self.assertEqual(VERSIONS["ACADOS_TERA_RENDERER_VERSION"], "0.2.0")
        self.assertEqual(REVISION, VERSIONS["XNAVDP_REVISION"])

    def test_runtime_patch_uses_tiled_camera_on_modern_isaaclab(self):
        runtime_patch = (ROOT / "config/xnavdp-rootless-runtime.patch").read_text()
        self.assertIn(
            "from isaaclab.sensors import ContactSensorCfg, CameraCfg, TiledCameraCfg",
            runtime_patch,
        )
        self.assertIn("DINGO_CameraCfg = TiledCameraCfg(", runtime_patch)
        self.assertNotIn("omni.isaac.lab", runtime_patch)

    def test_scene_overlay_closes_relative_mdl_imports_without_asset_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            material = source / "room/material"
            material.mkdir(parents=True)
            (material / "chair.mdl").write_text("import .::OmniUe4Base;\n")
            (material / "OmniUe4Base.mdl").write_text("stale scene copy\n")
            (source / "room/mesh.usd").write_text("#usda 1.0\n")
            modules = root / "modules"
            modules.mkdir()
            for name in ("OmniUe4Base", "OmniUe4Function", "OmniUe4Translucent"):
                (modules / f"{name}.mdl").write_text(f"mdl 1.6; // {name}\n")
            (source / "models").mkdir()
            (source / "models/geometry.usd").write_text("#usda 1.0\n")
            target = root / "overlay"
            linked = materialize_scene_overlay(source, target, modules)
            self.assertTrue((target / "models").is_symlink())
            self.assertEqual((target / "models").resolve(), source / "models")
            self.assertEqual(linked, 1)
            self.assertFalse((target / "room/mesh.usd").is_symlink())
            self.assertFalse((target / "room/material/chair.mdl").is_symlink())
            self.assertEqual((target / "room/material/chair.mdl").read_text(),
                             (material / "chair.mdl").read_text())
            self.assertTrue((target / "room/material/OmniUe4Base.mdl").is_symlink())
            self.assertTrue((material / "OmniUe4Base.mdl").is_file())

    def test_single_split_input_cache_does_not_require_other_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = root / "scenes"
            for name in ("Materials", "SkyTexture"):
                (scenes / name).mkdir(parents=True)
            material = scenes / "internscenes_home/scene/material"
            material.mkdir(parents=True)
            (material / "chair.mdl").write_text("import .::OmniUe4Base;\n")
            (scenes / "navigation_metadata/internscenes_home/esdf").mkdir(
                parents=True
            )
            eval_python = root / "env/bin/python"
            eval_python.parent.mkdir(parents=True)
            eval_python.touch()
            modules = (
                root / "env/lib/python3.10/site-packages/omni/mdl/core/Ue4"
            )
            modules.mkdir(parents=True)
            (root / "env/lib/python3.1").symlink_to(
                "python3.10", target_is_directory=True
            )
            for name in ("OmniUe4Base", "OmniUe4Function", "OmniUe4Translucent"):
                (modules / f"{name}.mdl").write_text("mdl 1.6;\n")
            run_root = root / "run"
            run_root.mkdir()
            repo = root / "repo"
            (repo / "assets/robots").mkdir(parents=True)
            (repo / "assets/robots/dingo.usd").write_text("robot")
            episode = repo / "assets/scenes/internscenes_home/episode-scene"
            episode.mkdir(parents=True)
            (episode / "pointgoal_start_goal_pairs.npy").write_bytes(b"episodes")
            (episode.parent / "models").mkdir()
            (episode.parent / "models/large.usd").write_text("external asset")
            (repo / "assets/scenes/scene_split.json").write_text("{}")
            args = SimpleNamespace(
                suite={"splits": {"home": {"scenes": ["episode-scene"]}, "commercial": {}}},
                jobs=[SimpleNamespace(split="home")],
                scene_root=scenes,
                eval_python=str(eval_python),
            )
            with (
                patch("navbench.cli.ROOT", repo),
                patch("navbench.cli.suite_definition_sha256", return_value="a" * 64),
                patch("navbench.cli.cache_root", return_value=root / "cache"),
            ):
                inputs = prepare_upstream_inputs(args, run_root)
            self.assertTrue((inputs / "internscenes_home").is_dir())
            self.assertFalse((inputs / "internscenes_commercial").exists())
            pairs = inputs / "navigation_metadata/internscenes_home/pointgoal_start_pair"
            self.assertEqual(sorted(p.name for p in pairs.iterdir()), ["episode-scene"])
            self.assertEqual((pairs / "episode-scene/pointgoal_start_goal_pairs.npy").read_bytes(), b"episodes")

    def test_frozen_official_inputs(self):
        suite, jobs = load_suite(MANIFEST, ROOT / "assets/scenes", ROOT)
        self.assertEqual((len(jobs), sum(job.episodes for job in jobs)), (40, 4000))
        self.assertEqual(invalid_episode_assets(jobs), [])
        self.assertEqual(suite["episode_contract"]["columns"], list(EPISODE_COLUMNS))
        self.assertEqual(
            sha256_file(ROOT / "assets/robots/dingo.usd"),
            suite["simulator_contract"]["robot_asset_sha256"],
        )
        camera = suite["simulator_contract"]["camera"]
        self.assertEqual(camera["offset_xyz_m"], [0.28618, 0.0, 0.62532])
        self.assertEqual((camera["focal_length"], camera["offset_convention"]), (1.93, "usd"))
        split = json.loads((ROOT / "assets/scenes/scene_split.json").read_text())
        for domain in ("home", "commercial"):
            self.assertEqual(split[f"{domain}_eval"], suite["splits"][domain]["scenes"])
        if SCENE_ROOT.is_dir():
            external_suite, external_jobs = load_suite(MANIFEST, SCENE_ROOT, ROOT)
            self.assertEqual(invalid_navigation_assets(external_suite, external_jobs), [])

    def test_raw_depth_round_trip_and_rejection(self):
        self.assertEqual(
            list(inspect.signature(pointgoal_step).parameters)[:3],
            ["pointgoal", "rgb", "depth"],
        )
        app = Flask(__name__)
        app.logger.disabled = True

        @app.post("/pointgoal_step")
        def step():
            depth = read_depth(1)
            return policy_response(depth.reshape(1, -1, 1))

        expected = np.asarray(
            [0.0, np.nan, 6.0, 7.0, 8.0], dtype=np.float32
        ).reshape(1, 1, 5, 1)
        response = self._post_depth(app, _prepare_depth(expected))
        with np.load(io.BytesIO(response.data), allow_pickle=False) as payload:
            self.assertEqual(set(payload.files), {"trajectory"})
            actual = payload["trajectory"].reshape(expected.shape)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(actual[~np.isnan(actual)], expected[~np.isnan(expected)])
        for invalid in (
            np.zeros((1, 2, 2, 1), dtype=np.uint16),
            np.zeros((1, 2, 2), dtype=np.float32),
        ):
            self.assertEqual(self._post_depth(app, invalid).status_code, 500)

    @staticmethod
    def _post_depth(app, depth):
        return app.test_client().post(
            "/pointgoal_step",
            data={
                "depth_shape": json.dumps(list(depth.shape)),
                "depth_dtype": depth.dtype.str,
                "depth": (io.BytesIO(depth.tobytes()), "depth.raw"),
            },
            content_type="multipart/form-data",
        )

    def test_official_metrics(self):
        self.assertEqual(success_weighted_path_length(0, 4, 8), 0.0)
        self.assertEqual(success_weighted_path_length(1, 4, 2), 1.0)
        self.assertEqual(success_weighted_path_length(1, 4, 8), 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metric = root / "scenes/home/scene/metric.csv"
            metric.parent.mkdir(parents=True)
            with metric.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("success", "spl", "distance", "episode_idx")
                )
                writer.writeheader()
                writer.writerows((
                    {"success": 1, "spl": 1, "distance": 4, "episode_idx": 0},
                    {"success": 0, "spl": 0, "distance": 6, "episode_idx": 1},
                ))
            subprocess.run((
                sys.executable, "-m", "navbench.metrics", "--root", str(root),
                "--output", str(root / "episodes.csv"),
                "--summary", str(root / "summary.csv"), "--expected", "2",
            ), cwd=ROOT, check=True, capture_output=True)
            with (root / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["domain"], "home")
        self.assertEqual((float(rows[0]["success_rate"]), float(rows[0]["mean_spl"])), (0.5, 0.5))

    def test_full_dry_run_uses_official_b16_jobs(self):
        result = subprocess.run((
            sys.executable, "-m", "navbench", "--model", "navdp",
            "--scene-root", str(SCENE_ROOT), "--gpus", "0", "--dry-run",
        ), cwd=ROOT, check=True, text=True, capture_output=True)
        plan = json.loads(result.stdout)
        self.assertEqual((plan["scene_count"], plan["episode_count"]), (40, 4000))
        self.assertEqual(plan["num_envs"], 16)
        self.assertEqual(plan["profile"], "pointgoal-v2-official-b16")
        self.assertEqual(plan["workers"][0]["gpu"], 0)
        self.assertIn("navbench.scene_evaluator", plan["workers"][0]["evaluator_example"])

    def test_repeated_checkpoints_share_one_evaluator_plan(self):
        result = subprocess.run((
            sys.executable, "-m", "navbench", "--model", "navdp",
            "--scene-root", str(SCENE_ROOT), "--gpus", "0",
            "--checkpoint", "/tmp/checkpoint-a.pt",
            "--checkpoint", "/tmp/checkpoint-b.pt", "--dry-run",
        ), cwd=ROOT, check=True, text=True, capture_output=True)
        plan = json.loads(result.stdout)
        worker = plan["workers"][0]
        self.assertEqual(len(worker["servers"]), 2)
        self.assertEqual(
            worker["evaluator_example"].count("navbench.scene_evaluator"), 1
        )

    def test_checkpoint_queue_pins_one_scene_per_gpu(self):
        suite = json.loads(MANIFEST.read_text())
        scene = suite["splits"]["home"]["scenes"][0]
        with tempfile.TemporaryDirectory() as tmp:
            queue_root = Path(tmp) / "incoming"
            result = subprocess.run((
                sys.executable, "-m", "navbench", "--model", "navdp",
                "--scene-root", str(SCENE_ROOT), "--gpus", "0",
                "--scenes", f"home/{scene}",
                "--checkpoint", "/tmp/checkpoint.pt",
                "--checkpoint-queue", str(queue_root), "--dry-run",
            ), cwd=ROOT, check=True, text=True, capture_output=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["checkpoint_queue"], str(queue_root.resolve()))
        self.assertEqual((plan["scene_count"], len(plan["workers"])), (1, 1))

    def test_checkpoint_queue_rejects_more_scenes_than_gpus(self):
        result = subprocess.run((
            sys.executable, "-m", "navbench", "--model", "navdp",
            "--scene-root", str(SCENE_ROOT), "--gpus", "0",
            "--checkpoint", "/tmp/checkpoint.pt",
            "--checkpoint-queue", "/tmp/checkpoints", "--dry-run",
        ), cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one GPU per selected resident scene", result.stderr)

    def test_persistent_evaluator_control_channel(self):
        program = (
            "import json,os,socket; "
            f"prefix={EVENT_PREFIX!r}; "
            "path=os.environ['NAVBENCH_CONTROL_SOCKET']; "
            "server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); "
            "server.bind(path); server.listen(1); "
            "print(prefix+json.dumps({'event':'ready'}),flush=True); "
            "connection,_=server.accept(); control=connection.makefile('r'); "
            "run=json.loads(control.readline()); "
            "print(prefix+json.dumps({'event':'done','run_id':run['run_id']}),flush=True); "
            "close=json.loads(control.readline()); "
            "assert close['command']=='close'"
        )
        with tempfile.TemporaryDirectory() as tmp:
            process = EvaluatorProcess(
                [sys.executable, "-u", "-c", program],
                cwd=ROOT,
                env=dict(os.environ),
                log_path=Path(tmp) / "eval.log",
                cancelled=lambda: False,
            )
            self.assertEqual(process.wait_for("ready")["event"], "ready")
            process.send({"command": "run", "run_id": "checkpoint:scene"})
            self.assertEqual(
                process.wait_for("done")["run_id"], "checkpoint:scene"
            )
            process.close()
            self.assertEqual(process.process.returncode, 0)

    def test_resident_scene_metrics_are_sorted_by_episode_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            metric = Path(tmp) / "metric.csv"
            write_metrics([
                {"success": 0, "spl": 0, "distance": 2, "episode_idx": 1},
                {"success": 1, "spl": 1, "distance": 1, "episode_idx": 0},
            ], metric)
            with metric.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["episode_idx"]) for row in rows], [0, 1])

    def test_dry_run_scene_sharding_is_deterministic(self):
        result = subprocess.run((
            sys.executable, "-m", "navbench", "--model", "navdp",
            "--scene-root", str(SCENE_ROOT), "--gpus", "0,1",
            "--shard-index", "1", "--shard-count", "2", "--dry-run",
        ), cwd=ROOT, check=True, text=True, capture_output=True)
        plan = json.loads(result.stdout)
        self.assertEqual(
            (plan["scene_count"], plan["episode_count"],
             plan["shard_index"], plan["shard_count"]),
            (20, 2000, 1, 2),
        )
        suite = json.loads(MANIFEST.read_text())
        self.assertEqual(
            plan["scenes"][0]["key"],
            f"home/{suite['splits']['home']['scenes'][1]}",
        )

        weighted = []
        for shard_index in (0, 1):
            result = subprocess.run((
                sys.executable, "-m", "navbench", "--model", "navdp",
                "--scene-root", str(SCENE_ROOT), "--gpus", "0,1",
                "--shard-index", str(shard_index), "--shard-count", "2",
                "--shard-weights", "2,1", "--dry-run",
            ), cwd=ROOT, check=True, text=True, capture_output=True)
            weighted.append(json.loads(result.stdout))
        self.assertEqual([item["scene_count"] for item in weighted], [27, 13])
        self.assertEqual(
            set(scene["key"] for item in weighted for scene in item["scenes"]),
            set(job.key for job in load_suite(MANIFEST, SCENE_ROOT, ROOT)[1]),
        )

    def test_metrics_merges_disjoint_shards_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = []
            for domain, scene in (("home", "a"), ("commercial", "b")):
                shard = root / f"shard-{len(roots)}"
                metric = shard / f"scenes/{domain}/{scene}/metric.csv"
                metric.parent.mkdir(parents=True)
                with metric.open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=("success", "spl", "distance", "episode_idx")
                    )
                    writer.writeheader()
                    writer.writerow({
                        "success": 1, "spl": 1, "distance": 4, "episode_idx": 0,
                    })
                roots.append(shard)
            metadata = {
                "suite": "test", "suite_definition_sha256": "suite",
                "model": "navdp", "precision": "fp32", "num_envs": 1,
                "runtime_revision": "runtime", "checkpoint_sha256": None,
                "model_config_sha256": None, "seed": 1234,
                "episodes_per_scene": 1, "full_suite_scene_count": 2,
                "shard_count": 2, "shard_weights": [1, 1],
                "episode_count": 1,
            }
            for index, (shard, scene_key) in enumerate(zip(
                roots, ("home/a", "commercial/b"),
            )):
                (shard / "run.json").write_text(json.dumps({
                    **metadata, "shard_index": index, "scene_keys": [scene_key],
                }))
            subprocess.run((
                sys.executable, "-m", "navbench.metrics",
                "--root", str(roots[0]), "--root", str(roots[1]),
                "--output", str(root / "episodes.csv"),
                "--summary", str(root / "summary.csv"), "--expected", "2",
            ), cwd=ROOT, check=True, capture_output=True)
            with (root / "episodes.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

            overlap = root / "overlap"
            metric = overlap / "scenes/home/a/metric.csv"
            metric.parent.mkdir(parents=True)
            metric.write_text(
                "success,spl,distance,episode_idx\n1,1,4,0\n",
                encoding="utf-8",
            )
            (overlap / "run.json").write_text(json.dumps({
                **metadata, "shard_index": 1, "scene_keys": ["home/a"],
            }))
            failed = subprocess.run((
                sys.executable, "-m", "navbench.metrics",
                "--root", str(roots[0]), "--root", str(overlap),
                "--output", str(root / "bad.csv"),
                "--summary", str(root / "bad-summary.csv"),
            ), cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("overlapping scenes", failed.stderr)


if __name__ == "__main__":
    unittest.main()
