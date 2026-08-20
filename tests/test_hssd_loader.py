import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from curvenav.config import CurveNavConfig
from curvenav.data import build_hssd_v2_loader, v2a_motion_context
from curvenav.data.depth_bank import gather_depth_sequences, load_packed_depth_bank
from curvenav.data.depth_cache import prepare_hssd_v2_depth_cache
from curvenav.data.loader import CurveNavSandDataset
from curvenav.deployment.runtime import CurveNavRuntime


def _write_hssd_fixture(root: Path) -> None:
    manifest = {
        "schema_version": "curvenav_hssd_v2.0",
        "history_contract": {"frames": 4, "index_offsets": [-9, -6, -3, 0]},
        "camera": {"depth": {"dtype": "float32", "unit": "m"}},
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest))
    record = {
        "sample_id": "hssd_v2/train/unit/run_0000:0012",
        "episode_id": "hssd_v2/train/unit/run_0000",
        "split": "train",
        "scene_id": "unit",
        "anchor_index": 12,
        "history_indices": [3, 6, 9, 12],
        "task_goal_local_xy": [0.3, 0.0],
        "target_end_index": 14,
    }
    (root / "samples.jsonl").write_text(json.dumps(record) + "\n")
    episode = root / "train/dataset_hssd_unit/run_0000"
    episode.mkdir(parents=True)
    world_xy = np.column_stack(
        (np.zeros(15, dtype=np.float32), np.arange(15, dtype=np.float32) * 0.15)
    )
    yaw = np.full(15, np.pi / 2.0, dtype=np.float32)
    np.savez(episode / "route.npz", world_xy=world_xy, yaw_rad=yaw)
    depth = np.stack(
        [np.full((6, 8), frame + 1.0, dtype=np.float32) for frame in range(15)]
    )
    depth_path = episode / "depth_m.npy"
    np.save(depth_path, depth)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "frames": len(depth),
                "depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
            }
        )
    )


def _append_validation_hssd_fixture(root: Path) -> None:
    record = {
        "sample_id": "hssd_v2/validation/unit/run_0001:0012",
        "episode_id": "hssd_v2/validation/unit/run_0001",
        "split": "validation",
        "scene_id": "unit",
        "anchor_index": 12,
        "history_indices": [3, 6, 9, 12],
        "task_goal_local_xy": [0.6, 0.0],
        "target_end_index": 14,
    }
    with (root / "samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    episode = root / "validation/dataset_hssd_unit/run_0001"
    episode.mkdir(parents=True)
    world_xy = np.column_stack(
        (np.zeros(15, dtype=np.float32), np.arange(15, dtype=np.float32) * 0.15)
    )
    yaw = np.full(15, np.pi / 2.0, dtype=np.float32)
    np.savez(episode / "route.npz", world_xy=world_xy, yaw_rad=yaw)
    depth = np.stack(
        [np.full((6, 8), frame + 1.0, dtype=np.float32) for frame in range(15)]
    )
    depth_path = episode / "depth_m.npy"
    np.save(depth_path, depth)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "frames": len(depth),
                "depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
            }
        )
    )


def test_hssd_v2_loader_reconstructs_the_v2a_condition(tmp_path) -> None:
    _write_hssd_fixture(tmp_path)
    config = CurveNavConfig()
    prepare_hssd_v2_depth_cache(
        tmp_path,
        height=config.data.image_height,
        width=config.data.image_width,
        max_depth_m=config.data.max_depth_m,
        workers=1,
    )
    (tmp_path / "train/dataset_hssd_unit/run_0000/depth_m.npy").unlink()
    bundle = build_hssd_v2_loader(
        tmp_path,
        config.data,
        config.trajectory,
        batch_size=1,
        num_workers=0,
        split="train",
    )
    batch = next(iter(bundle.loader))
    bank = load_packed_depth_bank(bundle.depth_bank, torch.device("cpu"))
    depth = gather_depth_sequences(bank, batch["depth_indices"])

    assert batch["sample_index"].tolist() == [0]
    assert bundle.dataset.records[0]["sample_id"] == "hssd_v2/train/unit/run_0000:0012"
    assert depth.shape == (1, 4, 1, 168, 224)
    expected_depth = torch.tensor([0.5, 0.875, 1.0, 1.0])
    torch.testing.assert_close(
        depth[0, :, 0].mean(dim=(1, 2)).float(), expected_depth
    )
    torch.testing.assert_close(
        batch["motion_context"], torch.tensor([[1.0, 0.0, 1.0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(batch["canonical_path"][0, -1], torch.tensor([0.3, 0.0]))
    assert batch["control_points"].shape == (1, 12, 2)


def test_hssd_split_preserves_global_sample_index(tmp_path) -> None:
    _write_hssd_fixture(tmp_path)
    _append_validation_hssd_fixture(tmp_path)
    config = CurveNavConfig()
    prepare_hssd_v2_depth_cache(
        tmp_path,
        height=config.data.image_height,
        width=config.data.image_width,
        max_depth_m=config.data.max_depth_m,
        workers=1,
    )
    bundle = build_hssd_v2_loader(
        tmp_path,
        config.data,
        config.trajectory,
        batch_size=1,
        num_workers=0,
        split="validation",
    )

    condition = bundle.dataset.condition_item(0)

    assert condition["sample_index"].item() == 1
    assert "metric_path" not in condition
    assert next(iter(bundle.loader))["sample_index"].tolist() == [1]


def test_hssd_motion_matches_sand_and_runtime_contract() -> None:
    world_xy = np.array([[1.0, 2.0], [1.0, 2.15]], dtype=np.float32)
    yaw = np.array([np.pi / 2.0, np.pi / 2.0], dtype=np.float32)
    hssd = v2a_motion_context(world_xy, yaw, anchor_index=1)

    sand = object.__new__(CurveNavSandDataset)
    sand.upstream = SimpleNamespace(
        run_info={
            "run": {
                "traj_xyz": np.column_stack((world_xy, np.zeros(2, dtype=np.float32))),
                "traj_yaw": yaw,
                "traj_pitch": np.zeros(2, dtype=np.float32),
            }
        },
        sequence_length=4,
        frame_skip=0,
    )
    sand._depth_offsets = {"run": 0}
    sand_context = sand._prepare_sample("run", 1, 1)["motion_context"].numpy()

    runtime = object.__new__(CurveNavRuntime)
    runtime.batch_size = 1
    runtime.previous_positions = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
    half = np.pi / 4.0
    quaternion = np.array([[0.0, 0.0, np.sin(half), np.cos(half)]], dtype=np.float32)
    runtime_context = runtime._motion_condition(
        np.array([[1.0, 2.15, 0.0]], dtype=np.float32), quaternion
    )[0]

    np.testing.assert_allclose(hssd, sand_context, atol=1e-6)
    np.testing.assert_allclose(hssd, runtime_context, atol=1e-6)
