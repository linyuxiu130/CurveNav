import json
from pathlib import Path

import numpy as np
import pytest

from curvenav.config_io import load_config
from curvenav.data_generation.generate import validate_config
from curvenav.data_generation.grscenes import (
    export_geometry, habitat_from_usd, validate_scene_selection,
)


def test_grscenes_split_is_scene_held_out_and_excludes_official_evaluation():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/grscenes_dataset.json").read_text())
    validate_config(config, load_config(root / "configs/base.yaml").data)
    scenes = config["selected_scenes"]
    assert len(scenes) == 59
    assert sum(s["split"] == "train" for s in scenes) == 49
    split = json.loads((root / "online_evaluation/assets/scenes/scene_split.json").read_text())
    validate_scene_selection(scenes, split)
    bad = {**scenes[0], "scene_id": split["home_eval"][0]}
    with pytest.raises(ValueError, match="official training"):
        validate_scene_selection([bad], split)


def test_usd_basis_preserves_distance_and_handedness():
    basis = habitat_from_usd("Z", .01)
    np.testing.assert_allclose(basis @ [100, 200, 300, 1], [1, 3, -2, 1])
    np.testing.assert_allclose(basis[:3, :3].T @ basis[:3, :3], np.eye(3) * .0001)
    assert np.linalg.det(basis[:3, :3]) > 0


def test_usd_export_composes_instances_units_and_ignores_lighting_geometry(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, Gf
    import trimesh

    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    mesh = UsdGeom.Mesh.Define(stage, "/Prototype")
    mesh.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 100, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Xform.Define(stage, "/Root/Meshes")
    for i in range(3):
        instance = stage.DefinePrim(f"/Root/Meshes/instance{i}")
        instance.GetReferences().AddInternalReference("/Prototype")
        instance.SetInstanceable(True)
        UsdGeom.Xformable(instance).AddTranslateOp().Set(Gf.Vec3d(200*i, 0, 300))
        if i == 2:
            UsdGeom.Xformable(instance).AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
    stage.GetRootLayer().Save()
    result = export_geometry(source, tmp_path, tmp_path / "scene.glb")
    loaded = trimesh.load(tmp_path / "scene.glb")
    np.testing.assert_allclose(loaded.bounds, [[0, 3, -1], [4, 3, 0]])
    assert result["triangles"] == 3
    assert result["meshes"] == 2
    assert all(np.linalg.det(loaded.graph[node][0][:3, :3]) > 0 for node in loaded.graph.nodes_geometry)


def test_distance_strata_fit_small_scenes_and_cache_reuses_completed_routes(tmp_path, monkeypatch):
    from curvenav.config import DataConfig
    from curvenav.data_generation import generate as generation
    from curvenav.data_generation.geometry import Grid

    grid = Grid(np.ones((30, 40), bool), np.ones((30, 40), np.float32), np.zeros(2), .05)
    sampling = {"distance_unit": "quantiles", "bands": {
        name: {"range": interval, "weight": 1, "bearing_degrees": [-180, 180]}
        for name, interval in {"near": [0, 1/3], "middle": [1/3, 2/3], "far": [2/3, 1]}.items()}}
    ranges = generation.endpoint_distance_ranges(grid, sampling, 42)
    assert 0 == ranges["near"][0] < ranges["near"][1] == ranges["middle"][0]
    assert ranges["middle"][1] == ranges["far"][0] < ranges["far"][1] < 3
    assert ranges == generation.endpoint_distance_ranges(grid, sampling, 42)
    scene = {"scene_id": "small", "source_family": "family", "split": "train"}
    config = {"source": "grscenes", "selected_scenes": [scene],
              "routes_per_split": {"train": 3}, "endpoint_sampling": sampling}
    records = []
    for band, count in generation.route_quota(config, scene).items():
        for i in range(count):
            directory = tmp_path / "train/grscenes_small" / f"{band}_{i+1:04d}"
            directory.mkdir(parents=True)
            record = {"route_id": str(directory.relative_to(tmp_path)), "source_family": "family"}
            (directory / "metadata.json").write_text(json.dumps(record))
            records.append(record)
    monkeypatch.setattr(generation, "create_simulator", lambda *a: pytest.fail("cache hit must not load a simulator"))
    assert generation.generate_scene(scene, config, str(tmp_path), 1, DataConfig()) == records


def test_short_expert_uses_the_same_cubic_without_empty_optimization():
    from curvenav.data_generation.geometry import Grid, plan_route, timed_route
    grid = Grid(np.ones((40, 40), bool), np.ones((40, 40), np.float32), np.zeros(2), .05)
    route = plan_route(grid, np.array([.5, .5]), np.array([.6, .5]), 0.)
    assert route.curve.k == 3
    xy, yaw, control = timed_route(route.curve, .1, .3, .5)
    np.testing.assert_allclose(xy[:, 1], .5, atol=1e-8)
    np.testing.assert_allclose(yaw, 0, atol=1e-6)
    assert np.all(control[:, 0] >= 0) and control[:, 0].max() <= .3


def test_near_goal_extension_uses_metric_ranges_and_preserves_held_out_scenes():
    from curvenav.data_generation.generate import endpoint_distance_ranges, route_quota
    from curvenav.data_generation.geometry import Grid
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/grscenes_dataset.json").read_text())
    extension = json.loads((root / "configs/grscenes_near_goal_dataset.json").read_text())
    validate_config(extension, load_config(root / "configs/base.yaml").data)
    assert extension["selected_scenes"] == [s for s in original["selected_scenes"] if s["split"] == "train"]
    for scene in extension["selected_scenes"]:
        assert set(route_quota(extension, scene).values()) == {5}
    grid = Grid(np.ones((30, 40), bool), np.ones((30, 40), np.float32), np.zeros(2), .05)
    ranges = endpoint_distance_ranges(grid, extension["endpoint_sampling"], 42)
    assert all(bounds == [.5, 1.5] for bounds in ranges.values())


def test_merged_routes_check_actual_family_labels_across_roots(tmp_path):
    from curvenav.data.prepare import _route_examples
    roots = tuple(tmp_path / split for split in ("train", "validation"))
    for root in roots:
        root.mkdir()
        (root / "routes.jsonl").write_text(json.dumps({
            "source": "grscenes", "source_family": "same_layout", "split": root.name,
        }) + "\n")
    with pytest.raises(ValueError, match="source-family split leakage across"):
        _route_examples(roots, None)


def test_exact_split_totals_are_balanced_order_independent_and_extend_existing_routes():
    from collections import Counter
    from curvenav.data_generation.generate import route_quota
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/grscenes_dataset.json').read_text())
    quotas = {s['scene_id']: route_quota(config, s) for s in config['selected_scenes']}
    counts = Counter()
    for scene in config['selected_scenes']:
        counts[scene['split']] += sum(quotas[scene['scene_id']].values())
    assert counts == {'train': 5000, 'validation': 1000}
    reordered = json.loads(json.dumps(config, sort_keys=True))
    reordered['selected_scenes'].reverse()
    validate_config(reordered, load_config(root / 'configs/base.yaml').data)
    for scene in config['selected_scenes']:
        assert route_quota(reordered, scene) == quotas[scene['scene_id']]
        assert all(quotas[scene['scene_id']][band] >= count
                   for band, count in {'near': 5, 'middle': 10, 'far': 10}.items())
    for split in counts:
        sizes = [sum(quotas[s['scene_id']].values()) for s in config['selected_scenes'] if s['split'] == split]
        assert max(sizes) - min(sizes) <= 1
