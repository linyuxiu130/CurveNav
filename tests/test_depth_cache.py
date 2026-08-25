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
    assert manifest["excluded_runs"] == {
        "dataset_avoid/run_0001": "no_depth_frames"
    }
    assert (depth_cache_root(tmp_path, 126, 224) / "manifest.json").is_file()


def test_hssd_sample_cache_matches_the_generated_dataset_contract(tmp_path) -> None:
    sample_id = "hssd/train/scene/source_route_00/near_00/standard"
    sample_directory = (
        tmp_path
        / "train/dataset_hssd_scene/source_routes/source_route_00/anchors/near_00/standard"
    )
    sample_directory.mkdir(parents=True)
    np.save(
        sample_directory / "depth_m.npy",
        np.full((4, 360, 640), 4.0, dtype=np.float32),
    )
    observation_to_current = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1)
    )
    np.savez(
        sample_directory / "geometry.npz",
        task_goal_local_xy=np.array([4.0, 0.0], dtype=np.float32),
        observation_to_current=observation_to_current,
        target_path_local_xy=np.array(
            [[0.0, 0.0], [3.0, 0.0]], dtype=np.float32
        ),
    )
    record = {
        "sample_id": sample_id,
        "sample_directory": str(sample_directory.relative_to(tmp_path)),
        "split": "train",
        "scene_id": "scene",
    }
    (tmp_path / "samples.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema": "curvenav_hssd_policy_dataset",
                "samples": 1,
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

    packed = np.load(
        hssd_depth_cache_root(tmp_path, 126, 224)
        / "train/00000.npy"
    )
    assert manifest["max_depth_m"] == 5.0
    assert packed.shape == (4, 126, 224)
    np.testing.assert_allclose(packed, 0.8, atol=3e-4)

    examples = _hssd_examples(tmp_path, CurveNavConfig())
    assert len(examples["train"]) == 1
    np.testing.assert_array_equal(examples["train"][0].depth_indices, range(4))
    np.testing.assert_array_equal(
        examples["train"][0].observation_to_current,
        observation_to_current,
    )
