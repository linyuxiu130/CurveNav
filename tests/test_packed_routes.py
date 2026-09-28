import json
import math

import numpy as np
import pytest

from curvenav.config import CurveNavConfig
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data.depth import BENCHMARK_INTRINSICS, depth_camera_contract
from curvenav.data_generation.generate import render_depth
from curvenav.data.prepare import _route_examples


@pytest.mark.parametrize("stationary_tail", [False, True])
def test_packed_routes_and_local_slicing_share_training_frames(tmp_path, monkeypatch, stationary_tail) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    config = CurveNavConfig()
    data = config.data
    pitch = math.radians(data.camera_downward_pitch_degrees)
    route_id = "train/dataset_hssd_scene/run_0001"
    route_directory = tmp_path / route_id
    route_directory.mkdir(parents=True)
    from types import SimpleNamespace
    monkeypatch.setattr("curvenav.data_generation.generate.set_pose", lambda *a: None)
    simulator = SimpleNamespace(get_sensor_observations=lambda: {
        "depth": np.full((360,640),4.,np.float32),
    })
    render_depth(simulator,np.zeros((5,3)),np.zeros(5),route_directory/"depth.npy",data)
    positions = np.array([0., .15, .3, .6, .6]) if stationary_tail else np.arange(5) * .15
    np.save(
        route_directory / "traj_xy.npy",
        np.column_stack((positions, np.zeros(5))).astype(np.float32),
    )
    np.save(
        route_directory / "traj_yaw.npy",
        np.zeros(5, dtype=np.float32),
    )
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (5, 4, 4)).copy()
    poses[:, 0, 3] = positions
    np.save(route_directory / "body_to_world.npy", poses)
    np.save(route_directory / "timestamps.npy", np.arange(5, dtype=np.float64) * 0.1)
    record = {
        "route_id": route_id,
        "route_directory": route_id,
        "split": "train",
        "scene_id": "scene",
        "source": "hssd",
        "source_family": "scene",
        "frames": 5,
    }
    (tmp_path / "routes.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema": "curvenav_policy_depth_routes_v5",
                "observation": depth_camera_contract(data),
                "routes": 1,
                "route_contract": {
                    "navigation_geometry": (expert_navigation_geometry_contract())
                },
                "camera": {
                    "image": {
                        "K": [
                            [BENCHMARK_INTRINSICS.fx, 0.0, BENCHMARK_INTRINSICS.width / 2],
                            [0.0, BENCHMARK_INTRINSICS.fy, BENCHMARK_INTRINSICS.height / 2],
                            [0.0, 0.0, 1.0],
                        ]
                    },
                    "body_from_camera_optical": [
                        [
                            0.0,
                            -math.sin(pitch),
                            math.cos(pitch),
                            data.camera_forward_offset_m,
                        ],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, -math.cos(pitch), -math.sin(pitch), data.camera_height_m],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            }
        )
    )

    packed = np.load(route_directory / "depth.npy")
    assert packed.shape == (5, 126, 224)
    np.testing.assert_allclose(packed, 0.8, atol=3e-4)

    scene_root = tmp_path / "train/dataset_hssd_scene"
    np.savez_compressed(
        scene_root / "navigation_grid.npz",
        free=np.ones((32, 32), dtype=np.bool_),
        clearance_m=np.ones((32, 32), dtype=np.float32),
        origin_xy=np.array([-0.5, -0.5], dtype=np.float64),
        cell_size_m=np.array(0.1, dtype=np.float64),
    )

    examples = _route_examples((tmp_path, tmp_path), config)
    assert len(examples["train"]) == (6 if stationary_tail else 8)
    np.testing.assert_array_equal(examples["train"][0].depth_indices, [0] * 4)
    np.testing.assert_array_equal(
        examples["train"][0].observation_valid, [False] * 3 + [True]
    )
    np.testing.assert_allclose(examples["train"][0].point_goal, [0.6, 0.0])

    # Compile -> serialize -> strict load -> gather uses the same depth frames.
    from curvenav.data.prepare import _compile_split
    from curvenav.data.prepared import PreparedPolicyDataset, policy_dataset_contract
    from curvenav.data.depth_bank import load_packed_depth_bank
    from curvenav.data.batch import unpack_policy_batch
    from curvenav.data.observation import DepthContextBuffer
    from torch.utils.data import DataLoader, default_convert
    import torch

    destination = tmp_path / "prepared"
    destination.mkdir()
    (destination / "manifest.json").write_text(
        json.dumps({"contract": policy_dataset_contract(data, config.trajectory)})
    )
    _compile_split(
        destination / "train",
        examples["train"],
        42,
        config,
    )
    dataset = PreparedPolicyDataset(destination, "train", data, config.trajectory)
    packed_depth = load_packed_depth_bank(dataset.depth_bank)
    sample = next(iter(DataLoader(dataset, batch_size=1, collate_fn=default_convert)))
    from curvenav.data.depth_bank import gather_depth_observations

    sample["depth"] = gather_depth_observations(packed_depth, sample["depth_indices"])
    prepared = unpack_policy_batch(sample)
    # Compare a selected compiled anchor to the online history built from the same route.
    current_index = int(sample["depth_indices"][0, -1])
    runtime = DepthContextBuffer(config.data)
    runtime.reset(1)
    intrinsic = np.asarray(
        json.loads((tmp_path / "dataset_manifest.json").read_text())["camera"]["image"][
            "K"
        ],
        np.float32,
    )
    extrinsic = np.asarray(
        json.loads((tmp_path / "dataset_manifest.json").read_text())["camera"][
            "body_from_camera_optical"
        ],
        np.float32,
    )
    for index in range(current_index + 1):
        online = runtime.update(
            np.full((1, 360, 640, 1), 4.0, np.float32),
            poses[index : index + 1],
            intrinsic[None],
            extrinsic[None],
            np.array([index * 0.1]),
        )
    online["depth"] = online["depth"].astype(np.float32)
    for name, value in online.items():
        np.testing.assert_allclose(
            getattr(prepared.condition, name).numpy(), value, atol=1e-6
        )


def test_future_horizon_keeps_sub_waypoint_turns():
    from curvenav.data.prepare import _fixed_future
    path = np.array([[0., 0.], [.03, 0.], [.05, .02], [.05, .08], [.02, .11]], np.float32)
    clipped, terminal = _fixed_future(path, 24, .15)
    np.testing.assert_array_equal(clipped, path)
    assert terminal


def test_route_failure_after_planning_is_not_retried_or_erased(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from curvenav.data_generation import generate as generator
    from curvenav.data_generation.geometry import PlanningError

    xy = np.array([[0., 0.], [4., 0.]], np.float32)
    monkeypatch.setattr(generator, "candidate_pairs", lambda *args: [(xy[0], xy[1])] * 2)
    monkeypatch.setattr(generator, "source_route", lambda *args: None)
    monkeypatch.setattr(generator, "sampled_route", lambda *args: (
        xy, np.zeros(2), np.zeros((2, 3)), np.zeros(2), np.zeros((2, 2)),
    ))

    def fail_render(*args):
        raise PlanningError("render failure must propagate")

    monkeypatch.setattr(generator, "render_depth", fail_render)
    with pytest.raises(PlanningError, match="render failure must propagate"):
        generator.generate_route(
            None, SimpleNamespace(safe=lambda path: True), tmp_path, tmp_path,
            {"split": "train", "scene_id": "scene"}, 0, "near", [3., 6.],
            {"candidate_limit": 2, "observation_period_s": .1,
             "expert_speed_m_s": .3, "expert_angular_speed_rad_s": .5,
             "endpoint_sampling": {"bands": {"near": {"bearing_degrees": [-180., 180.]}}}},
            0., 42, CurveNavConfig().data,
        )
    assert (tmp_path / "near_0001.partial/traj_xy.npy").is_file()


def test_label_safety_checks_fitted_tail_beyond_nominal_horizon(tmp_path):
    import torch
    from curvenav.data.prepare import _SourceMetadata, _source_minimum_clearance
    from curvenav.data.privileged import SourceConfigurationSpaceQuery
    p = tmp_path / 'navigation_grid.npz'
    free = np.ones((50, 5), dtype=bool)
    free[37:] = False
    clearance = np.where(free, 1., 0.).astype(np.float32)
    np.savez(p, free=free, clearance_m=clearance,
             origin_xy=np.array([-.05, -.25]), cell_size_m=np.array(.1))
    source = _SourceMetadata((p,), np.array([0]), np.zeros((1, 2),np.float32),
                             np.zeros(1,np.float32), SourceConfigurationSpaceQuery.from_paths((p,)))
    path = torch.tensor([[[0., 0.], [3.8, 0.]]])
    prefix = source.query.query(path, torch.tensor([0]), torch.zeros(1,2), torch.zeros(1), 3.6)
    assert prefix.minimum_clearance_m.item() > .1
    assert _source_minimum_clearance(path, source, 3.6)[0] < .1
