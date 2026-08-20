import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from curvenav.data_generation.manifest import (
    balanced_scene_type_pair_caps,
    select_balanced_episodes,
)
from curvenav.data_generation.occupancy import NavigationGrid, read_ply_xyz
from curvenav.data_generation.planner import (
    PlannerConfig,
    PlanningError,
    RouteMetrics,
    SafeEfficientPlanner,
    _safety_efficiency_frontier,
)
from curvenav.data_generation.scan_refinement import refine_scan_style_trajectory
from curvenav.data_generation.scene_catalog import (
    SceneRecord,
    assign_group_disjoint_validation,
    scene_family_id,
)


def _grid(free: np.ndarray, cell_size: float = 0.1) -> NavigationGrid:
    padded = np.pad(free, 1, constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * cell_size
    clearance[~free] = 0.0
    return NavigationGrid(
        free=free,
        clearance_m=clearance.astype(np.float32),
        origin_xy=np.zeros(2),
        cell_size_m=cell_size,
    )


def _center(x: int, y: int) -> np.ndarray:
    return np.array([(x + 0.5) * 0.1, (y + 0.5) * 0.1])


def test_safe_planner_respects_clearance_and_detour_budget():
    free = np.ones((34, 24), dtype=bool)
    free[[0, -1], :] = False
    free[:, [0, -1]] = False
    free[17, 1:-1] = False
    free[17, 7] = True  # short, narrow opening
    free[17, 14:20] = True  # longer, safer opening
    planner = SafeEfficientPlanner(
        _grid(free),
        PlannerConfig(maximum_safe_detour_ratio=1.25),
    )

    route = planner.plan(_center(3, 7), _center(30, 7))

    assert route.metrics.reference_length_ratio <= 1.25 + 1e-6
    assert route.metrics.minimum_clearance_m >= 0.1 - 1e-6
    assert planner.grid.path_is_safe(route.path_xy, minimum_clearance_m=0.1)
    assert route.path_xy.shape[1] == 2
    assert route.refinement["method"] == "scan_style_lbfgs"


def test_safety_efficiency_frontier_removes_only_strictly_dominated_routes():
    def candidate(
        weight: float,
        *,
        length: float,
        clearance_p05: float,
        risk: float,
        curvature_p95: float,
    ):
        metrics = RouteMetrics(
            length_m=length,
            euclidean_distance_m=10.0,
            geodesic_ratio=length / 10.0,
            reference_length_ratio=length / 10.0,
            minimum_clearance_m=0.1,
            clearance_p05_m=clearance_p05,
            mean_clearance_m=0.3,
            risk_density=risk,
            preferred_clearance_exposure=0.2,
            total_turn_radians=0.0,
            curvature_p95=curvature_p95,
            maximum_curvature=curvature_p95,
        )
        return weight, np.zeros((2, 2)), metrics, {}

    dominated = candidate(
        0.0, length=10.1, clearance_p05=0.1, risk=0.3, curvature_p95=0.1
    )
    safer_and_shorter = candidate(
        2.0, length=10.0, clearance_p05=0.2, risk=0.29, curvature_p95=4.0
    )
    safer_but_longer = candidate(
        4.0, length=11.0, clearance_p05=0.3, risk=0.2, curvature_p95=1.0
    )
    same_clearance_but_more_efficient = candidate(
        6.0, length=10.9, clearance_p05=0.3, risk=0.19, curvature_p95=1.0
    )

    frontier = _safety_efficiency_frontier(
        [
            dominated,
            safer_and_shorter,
            safer_but_longer,
            same_clearance_but_more_efficient,
        ]
    )

    assert [item[0] for item in frontier] == [2.0, 4.0, 6.0]


def test_scan_style_refinement_smooths_a_safe_right_angle():
    free = np.ones((50, 50), dtype=bool)
    free[[0, -1], :] = False
    free[:, [0, -1]] = False
    grid = _grid(free)
    path = np.vstack(
        [
            _center(5, 5),
            _center(25, 5),
            _center(25, 30),
            _center(42, 30),
        ]
    )

    result = refine_scan_style_trajectory(
        path,
        grid,
        minimum_clearance_m=0.1,
        validation_step_m=0.025,
        output_spacing_m=0.05,
    )

    assert result.diagnostics["accepted"]
    assert result.diagnostics["refined_spatial_jerk_rms"] < result.diagnostics[
        "baseline_spatial_jerk_rms"
    ]
    assert grid.path_is_safe(
        result.path_xy,
        minimum_clearance_m=0.1,
        sample_step_m=0.025,
    )
    np.testing.assert_allclose(result.path_xy[[0, -1]], path[[0, -1]])


def test_astar_does_not_cut_diagonal_obstacle_corners():
    free = np.zeros((5, 5), dtype=bool)
    free[1, 1] = True
    free[2, 2] = True
    planner = SafeEfficientPlanner(_grid(free))

    with pytest.raises(PlanningError, match="reference path"):
        planner.plan(_center(1, 1), _center(2, 2))


def test_safety_check_never_resamples_away_an_unsafe_vertex():
    free = np.ones((6, 6), dtype=bool)
    free[2, 2] = False
    grid = _grid(free)
    path = np.vstack([_center(1, 1), _center(2, 2), _center(4, 4)])

    assert not grid.path_is_safe(
        path,
        minimum_clearance_m=0.0,
        sample_step_m=10.0,
    )


def test_ascii_ply_reader_keeps_xyz_and_ignores_color(tmp_path):
    ply = tmp_path / "map.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property double x",
                "property double y",
                "property double z",
                "property uchar red",
                "end_header",
                "1 2 3 4",
                "5 6 7 8",
            ]
        ),
        encoding="ascii",
    )

    np.testing.assert_allclose(read_ply_xyz(ply), [[1, 2, 3], [5, 6, 7]])


def _scene(scene_id: str, scene_type: str) -> SceneRecord:
    return SceneRecord(
        scene_id=scene_id,
        scene_type=scene_type,
        official_split="train",
        internal_split="train",
        family_id=scene_family_id(scene_id),
        navigable_ply="map.ply",
        pointgoal_npy="pairs.npy",
    )


def test_internal_validation_is_disjoint_by_family_across_scene_types():
    shared_home = _scene("MV7J6NIKTKJZ2AABAAAAADA8_usd", "home")
    shared_commercial = _scene("MV7J6NIKTKJZ2AABAAAAAAA8_usd", "commercial")
    other = _scene("MWBGLKQKTKJZ2AABAAAAAAA8_usd", "home")

    split = assign_group_disjoint_validation(
        [shared_home, shared_commercial, other], validation_fraction=0.5, seed=7
    )

    by_id = {(scene.scene_type, scene.scene_id): scene.internal_split for scene in split}
    assert by_id[("home", shared_home.scene_id)] == by_id[
        ("commercial", shared_commercial.scene_id)
    ]
    assert {scene.internal_split for scene in split} == {"train", "validation"}


def test_balanced_selection_equalizes_scene_types_before_scene_count():
    candidates = []
    for index in range(10):
        candidates.append(
            {
                "episode_id": f"home-{index}",
                "scene_type": "home",
                "difficulty": "open",
                "scene_id": f"h-{index % 5}",
                "pair_index": index,
            }
        )
    for index in range(4):
        candidates.append(
            {
                "episode_id": f"commercial-{index}",
                "scene_type": "commercial",
                "difficulty": "narrow",
                "scene_id": "c-0",
                "pair_index": index,
            }
        )

    selected = select_balanced_episodes(candidates, target_episodes=8, seed=42)

    assert sum(item["scene_type"] == "home" for item in selected) == 4
    assert sum(item["scene_type"] == "commercial" for item in selected) == 4


def test_balanced_selection_does_not_depend_on_planner_version_in_episode_id():
    candidates = [
        {
            "episode_id": f"v2-{index}",
            "scene_type": "home",
            "difficulty": "open",
            "scene_id": f"scene-{index % 2}",
            "pair_index": index,
        }
        for index in range(10)
    ]
    renamed = [
        {**item, "episode_id": str(item["episode_id"]).replace("v2", "v3")}
        for item in candidates
    ]

    selected_v2 = select_balanced_episodes(candidates, target_episodes=5, seed=42)
    selected_v3 = select_balanced_episodes(renamed, target_episodes=5, seed=42)

    assert [item["pair_index"] for item in selected_v2] == [
        item["pair_index"] for item in selected_v3
    ]


def test_pair_caps_oversample_scarce_scene_types():
    scenes = [
        *[_scene(f"home-{index}", "home") for index in range(4)],
        _scene("commercial-0", "commercial"),
    ]

    caps = dict(
        balanced_scene_type_pair_caps(scenes, base_pairs_per_scene=20)
    )

    assert caps == {"commercial": 80, "home": 20}
