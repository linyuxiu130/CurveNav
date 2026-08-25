from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from curvenav.data_generation import assets
from curvenav.data_generation.assets import selected_asset_paths
from curvenav.data_generation.generate import camera_contract, route_bands, validate_config
from curvenav.config_io import load_config
from curvenav.data_generation.geometry import (
    Grid,
    path_length,
    plan_route,
    source_family,
)


def test_route_distance_schedule_is_exact_and_unperturbed(base_config: dict) -> None:
    bands = route_bands(base_config)

    assert len(bands) == 25
    assert bands.count("near") == 5
    assert bands.count("middle") == 10
    assert bands.count("far") == 10


def test_clearance_aware_planner_is_safe_and_deterministic() -> None:
    free = np.ones((80, 80), dtype=bool)
    free[25:55, 36:44] = False
    from scipy.ndimage import distance_transform_edt

    clearance = (
        distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
        * 0.05
    )
    clearance[~free] = 0.0
    grid = Grid(free, clearance.astype(np.float32), np.zeros(2), 0.05)

    first = plan_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]))
    second = plan_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]))

    assert grid.safe(first.path_xy)
    assert np.allclose(first.path_xy, second.path_xy)
    assert path_length(first.path_xy) < 4.5


def test_source_family_is_stable() -> None:
    assert source_family("106366323_174226647") == "106366"


def test_hssd_generator_uses_the_model_camera_contract() -> None:
    project = Path(__file__).resolve().parents[1]
    generation = json.loads(
        (project / "configs/hssd_dataset.json").read_text(encoding="utf-8")
    )
    data = load_config(project / "configs/base.yaml").data
    camera = camera_contract(generation["camera"])

    assert camera["image"]["K"] == [
        [data.canonical_focal_x_px, 0.0, data.image_width / 2],
        [0.0, data.canonical_focal_y_px, data.image_height / 2],
        [0.0, 0.0, 1.0],
    ]
    assert camera["body_from_camera_optical"] == [
        [0.0, 0.0, 1.0, data.camera_forward_offset_m],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, data.camera_height_m],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_config_rejects_family_leakage(base_config: dict) -> None:
    config = dict(base_config)
    config["selected_scenes"] = [
        {"scene_id": f"{index + 100000}_a", "split": "train"} for index in range(16)
    ] + [
        {"scene_id": f"{index + 200000}_b", "split": "validation"} for index in range(4)
    ]
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
    assert (project / "data/hssd/repository_files.json").is_file()
    assert not (project / "data/hssd.building").exists()


@pytest.fixture
def base_config() -> dict:
    return {
        "selected_scenes": [],
        "endpoint_distance_bands_m": {
            "near": [3.0, 7.0],
            "middle": [7.0, 11.0],
            "far": [11.0, 15.0],
        },
        "routes_per_scene_by_distance": {"near": 5, "middle": 10, "far": 10},
        "routes_per_scene": 25,
        "expected_routes": 500,
        "route_sample_spacing_m": 0.15,
        "workers": 4,
        "gpu_device": 0,
        "camera": {
            "image_width": 224,
            "image_height": 126,
            "focal_x_px": 166.80851063829786,
            "focal_y_px": 166.80851063829786,
            "forward_offset_m": 0.0,
            "height_m": 0.40,
            "downward_pitch_degrees": 0.0,
        },
    }
