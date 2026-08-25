import json
import math

import numpy as np

from curvenav.config import CurveNavConfig
from curvenav.data.depth import depth_camera_contract
from curvenav.data.depth_cache import (
    hssd_depth_cache_root,
    prepare_hssd_depth_cache,
)
from curvenav.data.prepare import _hssd_examples


def test_hssd_route_cache_and_local_slicing_share_depth_frames(tmp_path) -> None:
    config = CurveNavConfig()
    data = config.data
    pitch = math.radians(data.camera_downward_pitch_degrees)
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
                "camera": {
                    "image": {
                        "K": [
                            [data.canonical_focal_x_px, 0.0, data.image_width / 2],
                            [0.0, data.canonical_focal_y_px, data.image_height / 2],
                            [0.0, 0.0, 1.0],
                        ]
                    },
                    "body_from_camera_optical": [
                        [0.0, -math.sin(pitch), math.cos(pitch), data.camera_forward_offset_m],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, -math.cos(pitch), -math.sin(pitch), data.camera_height_m],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            }
        )
    )

    manifest = prepare_hssd_depth_cache(
        tmp_path,
        data=data,
        workers=1,
    )

    cached = manifest["runs"][route_id]
    packed = np.load(hssd_depth_cache_root(tmp_path, 126, 224) / cached["file"])
    assert manifest["max_depth_m"] == 5.0
    assert manifest["target_camera"] == depth_camera_contract(data)
    assert packed.shape == (5, 126, 224)
    np.testing.assert_allclose(packed, 0.8, atol=3e-4)

    examples = _hssd_examples(tmp_path, config)
    assert len(examples["train"]) == 4
    np.testing.assert_array_equal(examples["train"][0].depth_indices, [0, 0, 0, 0])
    np.testing.assert_array_equal(
        examples["train"][0].observation_valid, [False, False, False, True]
    )
    np.testing.assert_allclose(examples["train"][0].point_goal, [0.6, 0.0])
