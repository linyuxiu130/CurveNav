"""Offline expert-route generation for CurveNav datasets.

This package deliberately stops before IsaacLab rendering.  It turns the
privileged navigation metadata into a deterministic, quality-audited episode
manifest that can be consumed efficiently by one or more simulator workers.
"""

__all__ = [
    "NavigationGrid",
    "PlannerConfig",
    "PlannedRoute",
    "SafeEfficientPlanner",
    "SceneRecord",
    "build_navigation_grid",
    "discover_training_scenes",
]


def __getattr__(name: str):
    if name in {"NavigationGrid", "build_navigation_grid"}:
        from curvenav.data_generation.occupancy import NavigationGrid, build_navigation_grid

        return {"NavigationGrid": NavigationGrid, "build_navigation_grid": build_navigation_grid}[name]
    if name in {"PlannerConfig", "PlannedRoute", "SafeEfficientPlanner"}:
        from curvenav.data_generation.planner import (
            PlannedRoute,
            PlannerConfig,
            SafeEfficientPlanner,
        )

        return {
            "PlannerConfig": PlannerConfig,
            "PlannedRoute": PlannedRoute,
            "SafeEfficientPlanner": SafeEfficientPlanner,
        }[name]
    if name in {"SceneRecord", "discover_training_scenes"}:
        from curvenav.data_generation.scene_catalog import SceneRecord, discover_training_scenes

        return {"SceneRecord": SceneRecord, "discover_training_scenes": discover_training_scenes}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
