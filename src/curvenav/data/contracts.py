"""Stable source geometry identifiers shared by generation and training data."""

from __future__ import annotations

from curvenav.physical import (
    BODY_OBSTACLE_MIN_Z_M,
    DINGO_USD_SHA256,
    DINGO_WHEEL_BASE_M,
    DINGO_WHEEL_RADIUS_M,
    MAXIMUM_TRAVERSABLE_HEIGHT_M,
    ROBOT_COLLISION_BOTTOM_Z_M,
    ROBOT_COLLISION_HEIGHT_M,
    ROBOT_COLLISION_TOP_Z_M,
    ROBOT_FOOTPRINT_RADIUS_M,
)


EXPERT_NAVIGATION_GEOMETRY_TYPE = (
    "stage_and_static_collision_meshes_robot_configuration_space"
)


def expert_navigation_geometry_contract() -> dict[str, float | str]:
    return {
        "type": EXPERT_NAVIGATION_GEOMETRY_TYPE,
        "embodiment": "dingo",
        "embodiment_asset_sha256": DINGO_USD_SHA256,
        "wheel_radius_m": DINGO_WHEEL_RADIUS_M,
        "wheel_base_m": DINGO_WHEEL_BASE_M,
        "footprint_radius_m": ROBOT_FOOTPRINT_RADIUS_M,
        "collision_bottom_z_m": ROBOT_COLLISION_BOTTOM_Z_M,
        "collision_top_z_m": ROBOT_COLLISION_TOP_Z_M,
        "collision_height_m": ROBOT_COLLISION_HEIGHT_M,
        "body_obstacle_min_z_m": BODY_OBSTACLE_MIN_Z_M,
        "maximum_traversable_height_m": MAXIMUM_TRAVERSABLE_HEIGHT_M,
    }
