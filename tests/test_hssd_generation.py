from __future__ import annotations

import json
import io
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from curvenav.data_generation import assets
from curvenav.data_generation import generate as generation
from curvenav.data_generation.audit import validate_route_spacing
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data_generation.assets import selected_asset_paths
from curvenav.data_generation.generate import (
    base_position_from_navmesh,
    camera_contract,
    configure_navmesh_settings,
    route_bands,
    validate_config,
)


from curvenav.config_io import load_config
from curvenav.config import DataConfig
from curvenav.data.depth import BENCHMARK_INTRINSICS
from curvenav.data_generation.geometry import (
    Grid,
    path_length,
    plan_route,
    source_family,
)
from curvenav.physical import (
    MAXIMUM_TRAVERSABLE_HEIGHT_M,
    MAXIMUM_TRAVERSABLE_SLOPE_DEGREES,
    ROBOT_COLLISION_HEIGHT_M,
    ROBOT_BASE_HEIGHT_ABOVE_GROUND_M,
    ROBOT_FOOTPRINT_RADIUS_M,
)


def test_route_speed_accounts_for_storage_rounding_without_allowing_overspeed():
    for origin in (0., 30., -30., 1000.):
        xy = np.array([[origin, origin], [origin + .03, origin]], np.float32)
        validate_route_spacing(xy, .03)
        xy[1, 0] += .001
        with pytest.raises(ValueError, match="speed"):
            validate_route_spacing(xy, .03)


def test_depth_stream_matches_policy_numpy_files(tmp_path, monkeypatch):
    from curvenav.config import DataConfig
    depth = np.empty((2, 360, 640), np.float32)
    depth[0] = 1
    depth[0, :180, :320] = np.nan
    depth[1] = 3
    frames = iter({"depth": d} for d in depth)
    simulator = SimpleNamespace(get_sensor_observations=lambda: next(frames))
    monkeypatch.setattr(generation, "set_pose", lambda *args: None)
    path = tmp_path / "depth.npy"
    _, invalid = generation.render_depth(simulator, np.zeros((2, 3)), np.zeros(2), path, DataConfig())
    expected_depth = np.empty((2, 126, 224), np.float16)
    expected_depth[0] = .2
    expected_depth[0, :63, :112] = 0
    expected_depth[1] = .6
    for name, expected in (("depth.npy", expected_depth),):
        reference = io.BytesIO()
        np.save(reference, expected, allow_pickle=False)
        assert (tmp_path / name).read_bytes() == reference.getvalue()
    assert invalid == .125
    assert not (tmp_path / "depth_m.npy").exists()


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
        [BENCHMARK_INTRINSICS.fx, 0.0, BENCHMARK_INTRINSICS.width / 2],
        [0.0, BENCHMARK_INTRINSICS.fy, BENCHMARK_INTRINSICS.height / 2],
        [0.0, 0.0, 1.0],
    ]
    pitch = math.radians(data.camera_downward_pitch_degrees)
    assert camera["body_from_camera_optical"] == [
        [0.0, -math.sin(pitch), math.cos(pitch), data.camera_forward_offset_m],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -math.cos(pitch), -math.sin(pitch), data.camera_height_m],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_render_pose_uses_the_dingo_base_link_height() -> None:
    floor_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    base_position = base_position_from_navmesh(floor_position)

    assert np.array_equal(floor_position, np.array([1.0, 2.0, 3.0]))
    assert base_position == pytest.approx(
        [1.0, 2.0 + ROBOT_BASE_HEIGHT_ABOVE_GROUND_M, 3.0]
    )


def test_hssd_generator_rejects_nonbenchmark_camera() -> None:
    project = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project / "configs/hssd_dataset.json").read_text(encoding="utf-8")
    )
    config["camera"]["height_m"] += 0.01

    with pytest.raises(ValueError, match="benchmark Dingo"):
        validate_config(config, DataConfig())


def test_hssd_expert_navmesh_includes_static_scene_objects() -> None:
    class Settings:
        def set_defaults(self) -> None:
            self.include_static_objects = False

    settings = Settings()
    configure_navmesh_settings(settings)

    assert settings.include_static_objects is True
    assert settings.agent_radius == ROBOT_FOOTPRINT_RADIUS_M
    assert settings.agent_height == ROBOT_COLLISION_HEIGHT_M
    assert settings.agent_max_climb == MAXIMUM_TRAVERSABLE_HEIGHT_M
    assert settings.agent_max_slope == MAXIMUM_TRAVERSABLE_SLOPE_DEGREES
    assert settings.cell_size == settings.cell_height == 0.05


def test_dingo_geometry_contract_is_derived_from_the_benchmark_asset() -> None:
    geometry = expert_navigation_geometry_contract()

    assert geometry["embodiment"] == "dingo"
    assert geometry["embodiment_asset_sha256"] == (
        "43db9c54066d833e0bfc91d8d33eb1ff3345d4c8a3cd29f914649bafffbcce20"
    )
    assert geometry["wheel_radius_m"] == pytest.approx(0.06125)
    assert geometry["wheel_base_m"] == pytest.approx(0.22616)
    assert geometry["footprint_radius_m"] == pytest.approx(0.167584539)
    assert geometry["collision_height_m"] == pytest.approx(0.161981500)
    assert geometry["base_height_above_ground_m"] == pytest.approx(0.044000001)


def test_config_rejects_family_leakage(base_config: dict) -> None:
    config = dict(base_config)
    config["selected_scenes"] = [
        {"scene_id": f"{index + 100000}_a", "split": "train"} for index in range(16)
    ] + [
        {"scene_id": f"{index + 200000}_b", "split": "validation"} for index in range(4)
    ]
    validate_config(config, DataConfig())
    config["selected_scenes"][-1]["scene_id"] = "100000_b"
    with pytest.raises(ValueError, match="source-family"):
        validate_config(config, DataConfig())


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


def test_hssd_download_requests_use_the_selected_endpoint(
    tmp_path, monkeypatch
) -> None:
    requests = []

    def open_request(request, timeout):
        assert timeout == 60
        requests.append(request)
        if "/api/" in request.full_url:
            return io.BytesIO(b'{"siblings": [{"rfilename": "asset.glb"}]}')
        return io.BytesIO(b"asset")

    monkeypatch.setattr(assets, "HF_ENDPOINT", "https://mirror.example")
    monkeypatch.setattr(assets, "urlopen", open_request)
    assert assets._repository_paths() == ["asset.glb"]
    assets._download(tmp_path, "asset.glb")
    assert (tmp_path / "asset.glb").read_bytes() == b"asset"
    assert not (tmp_path / "asset.glb.partial").exists()
    assert [request.full_url for request in requests] == [
        f"https://mirror.example/api/datasets/{assets.HSSD_REPOSITORY}/revision/{assets.HSSD_COMMIT}",
        f"https://mirror.example/datasets/{assets.HSSD_REPOSITORY}/resolve/{assets.HSSD_COMMIT}/asset.glb",
    ]
    assert all(
        request.get_header("User-agent") == "CurveNav/1.0" for request in requests
    )


def test_hssd_download_retries_timeout(tmp_path, monkeypatch):
    attempts = []

    def open_request(request, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise TimeoutError("stalled connection")
        return io.BytesIO(b"complete")

    monkeypatch.setattr(assets, "urlopen", open_request)
    monkeypatch.setattr(assets.time, "sleep", lambda _: None)
    assets._download(tmp_path, "asset.glb")
    assert attempts == [60, 60]
    assert (tmp_path / "asset.glb").read_bytes() == b"complete"
    assert not (tmp_path / "asset.glb.partial").exists()


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

    building = project / "data/hssd.building"
    building.mkdir(parents=True)
    (building / "repository_files.json").write_text(
        json.dumps({"revision": assets.HSSD_COMMIT, "files": repository_paths})
    )
    download(building, "scenes/scene.scene_instance.json")
    scene_mtime = (building / "scenes/scene.scene_instance.json").stat().st_mtime_ns
    manifest = assets.download_assets(config_path)
    assert (
        project / "data/hssd/scenes/scene.scene_instance.json"
    ).stat().st_mtime_ns == scene_mtime

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
            "near": [3.0, 6.0],
            "middle": [6.0, 8.5],
            "far": [8.5, 10.5],
        },
        "routes_per_scene_by_distance": {"near": 5, "middle": 10, "far": 10},
        "observation_period_s": 0.1,
        "expert_speed_m_s": 0.3,
        "expert_angular_speed_rad_s": 0.5,
        "workers": 4,
        "gpu_device": 0,
        "camera": {
            "image_width": 640,
            "image_height": 360,
            "focal_x_px": 326.398559570312,
            "focal_y_px": 326.398559570312,
            "forward_offset_m": 0.28618,
            "height_m": 0.62532,
            "downward_pitch_degrees": 10.0,
        },
    }


def test_curve_clock_preserves_forward_turning_and_sensor_rate():
    from scipy.interpolate import BSpline
    from curvenav.data_generation.geometry import timed_route
    # Convex-hull cubic turns to a rear endpoint without reversing the body.
    curve = BSpline([0, 0, 0, 0, 1, 1, 1, 1],
                    [[0, 0], [2, 0], [2, 3], [-1, 3]], 3)
    xy, yaw, controls = timed_route(curve, .1, .3, .5)
    np.testing.assert_allclose(xy[[0, -1]], [[0, 0], [-1, 3]], atol=1e-10)
    assert np.all(controls[:, 0] > 0)
    assert np.max(controls[:, 0]) <= .3 + 1e-8
    assert np.max(abs(controls[:, 1])) <= .5 + 1e-8
    assert np.max(abs(np.diff(np.unwrap(yaw)))) <= .05 + 1e-6
    assert np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1)) <= .03 + 1e-6
    # Central finite differences converge to the analytic forward-only velocity.
    velocity = (xy[2:] - xy[:-2]) / .2
    lateral = -np.sin(yaw[1:-1])*velocity[:, 0] + np.cos(yaw[1:-1])*velocity[:, 1]
    assert np.max(abs(lateral)) < 1e-4


def test_clearance_curve_energy_gradient_matches_finite_difference():
    from scipy.interpolate import BSpline
    from scipy.ndimage import distance_transform_edt
    from scipy.optimize._numdiff import approx_derivative
    from curvenav.data_generation.geometry import _curve_objective
    free = np.ones((80,80),bool)
    free[30:40,30:40] = False
    distance = distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*.05
    grid = Grid(free,distance,np.zeros(2),.05)
    controls = np.array([[.617,.793],[1.037,.719],[1.843,1.017],[2.413,.613]])
    u = np.linspace(0,1,37)
    identity = BSpline([0,0,0,0,1,1,1,1],np.eye(4),3)
    args = (identity(u),identity.derivative()(u),identity.derivative(2)(u),np.ones(37)/37,grid,2.)
    value, gradient = _curve_objective(controls,*args)
    numerical = approx_derivative(lambda x:_curve_objective(x.reshape(4,2),*args)[0],controls.ravel(),method='3-point',abs_step=1e-6)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient.ravel(),numerical.ravel(),rtol=2e-4,atol=2e-5)


def test_clearance_optimization_keeps_a_narrow_feasible_corridor():
    from scipy.ndimage import distance_transform_edt
    free = np.zeros((80,80),bool)
    free[:,28:34] = True
    distance = distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*.05
    grid = Grid(free,distance,np.zeros(2),.05)
    plan = plan_route(grid,np.array([.5,1.55]),np.array([3.5,1.55]))
    assert grid.safe(plan.path_xy)
    assert grid.clearance(plan.path_xy).min() <= .15+1e-6


def test_route_audit_rejects_valid_but_misaligned_depth_pose():
    from curvenav.data_generation.audit import validate_route_pose
    xy = np.array([[1., 2.], [1.1, 2.1]], dtype=np.float32)
    yaw = np.array([.3, .4], dtype=np.float32)
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[:, :2, 3] = xy * [1, -1]
    c, s = np.cos(yaw), np.sin(yaw)
    poses[:, 0, 0] = poses[:, 1, 1] = c
    poses[:, 0, 1], poses[:, 1, 0] = s, -s
    validate_route_pose(xy, yaw, poses)
    poses[1, 0, 3] += .1
    with pytest.raises(ValueError, match='disagree'):
        validate_route_pose(xy, yaw, poses)
    poses[1, 0, 3] -= .1
    with pytest.raises(ValueError, match='disagree'):
        validate_route_pose(xy, yaw + .1, poses)
