import json

import cv2
import numpy as np

from curvenav.config import CurveNavConfig
from curvenav.data.depth_cache import (
    depth_cache_root,
    hssd_depth_cache_root,
    prepare_depth_cache,
    prepare_hssd_depth_cache,
)
from curvenav.data.prepare import _hssd_examples


def test_sand_depth_cache_records_empty_upstream_runs(tmp_path) -> None:
    valid_depth = tmp_path / "dataset_avoid/run_0000/depth"
    empty_depth = tmp_path / "dataset_avoid/run_0001/depth"
    valid_depth.mkdir(parents=True)
    empty_depth.mkdir(parents=True)
    assert cv2.imwrite(
        str(valid_depth / "depth_0000.png"),
        np.full((480, 640), 1000, dtype=np.uint16),
    )

    manifest = prepare_depth_cache(
        tmp_path,
        height=126,
        width=224,
        depth_units_per_m=1000.0,
        max_depth_m=5.0,
        workers=1,
    )

    assert manifest["runs"] == {"dataset_avoid/run_0000": 1}
    assert manifest["excluded_runs"] == {"dataset_avoid/run_0001": "no_depth_frames"}
    assert (depth_cache_root(tmp_path, 126, 224) / "manifest.json").is_file()


def test_hssd_route_cache_and_local_slicing_share_depth_frames(tmp_path) -> None:
    route_id = "train/dataset_hssd_scene/run_0001"
    route_directory = tmp_path / route_id
    route_directory.mkdir(parents=True)
    np.save(
        route_directory / "depth_m.npy",
        np.full((5, 126, 224), 4.0, dtype=np.float32),
    )
    np.save(
        route_directory / "traj_xy.npy",
        np.column_stack((np.arange(5) * 0.15, np.zeros(5))).astype(np.float32),
    )
    np.save(
        route_directory / "traj_yaw.npy",
        np.zeros(5, dtype=np.float32),
    )
    record = {
        "route_id": route_id,
        "route_directory": route_id,
        "split": "train",
        "scene_id": "scene",
        "frames": 5,
    }
    (tmp_path / "routes.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema": "curvenav_hssd_expert_routes",
                "routes": 1,
            }
        )
    )

    manifest = prepare_hssd_depth_cache(
        tmp_path,
        height=126,
        width=224,
        max_depth_m=5.0,
        workers=1,
    )

    cached = manifest["runs"][route_id]
    packed = np.load(hssd_depth_cache_root(tmp_path, 126, 224) / cached["file"])
    assert manifest["max_depth_m"] == 5.0
    assert packed.shape == (5, 126, 224)
    np.testing.assert_allclose(packed, 0.8, atol=3e-4)

    examples = _hssd_examples(tmp_path, CurveNavConfig())
    assert len(examples["train"]) == 4
    np.testing.assert_array_equal(examples["train"][0].depth_indices, [0, 0, 0, 0])
    np.testing.assert_array_equal(
        examples["train"][0].observation_valid, [False, False, False, True]
    )
    np.testing.assert_allclose(examples["train"][0].point_goal, [0.6, 0.0])
