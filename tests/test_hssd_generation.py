from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from curvenav.data_generation import assets
from curvenav.data_generation.assets import selected_asset_paths
from curvenav.data_generation.generate import anchor_candidates, validate_config
from curvenav.data_generation.geometry import (
    Grid,
    observation_to_current,
    path_length,
    plan_route,
    prefix,
    route_turn,
    source_family,
    variant_history,
)


def test_observation_transform_is_expressed_in_current_body_frame() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    yaw = np.array([0.0, np.pi / 4, np.pi / 2])
    pose = observation_to_current(xy, yaw)

    assert pose.shape == (3, 4)
    assert np.allclose(pose[-1], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(pose[0, :2], [-1.0, 1.0])


def test_prefix_caps_arc_and_naturally_shortens() -> None:
    long_path = np.array([[0.0, 0.0], [5.0, 0.0]])
    short_path = np.array([[0.0, 0.0], [1.2, 0.0]])

    assert path_length(prefix(long_path, 3.0)) == pytest.approx(3.0)
    assert path_length(prefix(short_path, 3.0)) == pytest.approx(1.2)


def test_anchor_candidates_include_near_middle_and_far() -> None:
    route = np.column_stack((np.arange(0.0, 12.05, 0.05), np.zeros(241)))
    bands = {"near": [0.5, 3.0], "middle": [3.0, 6.0], "far": [6.0, 10.0]}
    values = anchor_candidates(route, np.array([12.0, 0.0]), bands, 0.25, 17)

    assert len(values["near"]) >= 1
    assert len(values["middle"]) >= 2
    assert len(values["far"]) >= 2
    assert values["near"].max() > path_length(route) - 3.5


def test_variant_history_uses_physical_four_frame_profile() -> None:
    route = np.column_stack((np.arange(0.0, 8.05, 0.05), np.zeros(161)))
    history = variant_history(route, 4.0, 0.45, 25.0)

    assert history["world_xy"].shape == (4, 2)
    assert history["world_xy"][0, 1] == pytest.approx(0.0)
    assert history["world_xy"][-1, 1] == pytest.approx(0.45)
    assert np.rad2deg(history["yaw"][-1]) == pytest.approx(25.0)


def test_clearance_aware_planner_is_safe_and_deterministic() -> None:
    free = np.ones((80, 80), dtype=bool)
    free[25:55, 36:44] = False
    from scipy.ndimage import distance_transform_edt

    clearance = distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1] * 0.05
    clearance[~free] = 0.0
    grid = Grid(free, clearance.astype(np.float32), np.zeros(2), 0.05)

    first = plan_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]))
    second = plan_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]))

    assert grid.safe(first.path_xy)
    assert np.allclose(first.path_xy, second.path_xy)
    assert path_length(first.path_xy) < 4.5


def test_route_turn_and_source_family_are_stable() -> None:
    left = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    right = left * np.array([1.0, -1.0])

    assert route_turn(left, 0.0)[1] == "left"
    assert route_turn(right, 0.0)[1] == "right"
    assert source_family("106366323_174226647") == "106366"


def test_config_rejects_family_leakage(base_config: dict) -> None:
    config = dict(base_config)
    config["selected_scenes"] = [
        {"scene_id": f"{index + 100000}_a", "split": "train"} for index in range(16)
    ] + [{"scene_id": f"{index + 200000}_b", "split": "validation"} for index in range(4)]
    validate_config(config)
    config["selected_scenes"][-1]["scene_id"] = "100000_b"
    with pytest.raises(ValueError, match="source-family"):
        validate_config(config)


def test_hssd_asset_selection_is_derived_from_scene_instances(tmp_path: Path) -> None:
    scene_id = "scene"
    scene_path = tmp_path / "scenes" / f"{scene_id}.scene_instance.json"
    scene_path.parent.mkdir()
    scene_path.write_text(
        json.dumps({"object_instances": [{"template_name": "object"}]}),
        encoding="utf-8",
    )
    repository_paths = [
        "hssd-hab.scene_dataset_config.json",
        "semantics/hssd-hab_semantic_lexicon.json",
        f"scenes/{scene_id}.scene_instance.json",
        f"stages/{scene_id}.glb",
        f"stages/{scene_id}.stage_config.json",
        f"semantics/scenes/{scene_id}.semantic_config.json",
        "objects/o/object.object_config.json",
        "objects/o/object.glb",
        "objects/o/object.collider.glb",
    ]

    selected = selected_asset_paths(tmp_path, repository_paths, [scene_id])

    assert set(repository_paths) == selected


def test_hssd_asset_download_is_atomic_and_commit_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    config_path = project / "configs" / "hssd_dataset.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "asset_root": "data/hssd",
                "selected_scenes": [{"scene_id": "scene", "split": "train"}],
            }
        ),
        encoding="utf-8",
    )
    repository_paths = [
        "hssd-hab.scene_dataset_config.json",
        "semantics/hssd-hab_semantic_lexicon.json",
        "scenes/scene.scene_instance.json",
        "stages/scene.glb",
        "stages/scene.stage_config.json",
        "semantics/scenes/scene.semantic_config.json",
        "objects/o/object.object_config.json",
        "objects/o/object.glb",
    ]

    def download(root: Path, path: str) -> None:
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path == "scenes/scene.scene_instance.json":
            value = {"object_instances": [{"template_name": "object"}]}
            destination.write_text(json.dumps(value), encoding="utf-8")
        elif path.endswith(".json"):
            destination.write_text("{}", encoding="utf-8")
        else:
            destination.write_bytes(struct.pack("<4sII", b"glTF", 2, 12))

    monkeypatch.setattr(assets, "_repository_paths", lambda: repository_paths)
    monkeypatch.setattr(assets, "_download", download)

    manifest = assets.download_assets(config_path)

    assert manifest["commit"] == assets.HSSD_COMMIT
    assert manifest["files"] == len(repository_paths)
    assert (project / "data/hssd/download_manifest.json").is_file()
    assert not (project / "data/hssd.building").exists()


@pytest.fixture
def base_config() -> dict:
    return {
        "selected_scenes": [],
        "trajectory_variants": [
            {"name": name}
            for name in (
                "standard",
                "left_mild",
                "right_mild",
                "left_hard",
                "right_hard",
            )
        ],
        "goal_distance_bands": {
            "near": [0.5, 3.0],
            "middle": [3.0, 6.0],
            "far": [6.0, 10.0],
        },
        "anchors_per_source_route": {"near": 1, "middle": 2, "far": 2},
        "source_routes_per_scene": 2,
        "expected_samples": 1000,
        "workers": 4,
        "gpu_device": 0,
    }
