# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

OBSTACLE_LOSS = 2.0
TRAVERSABLE_INTENDED_LOSS = 0
TRAVERSABLE_UNINTENDED_LOSS = 0.5
ROAD_LOSS = 1.5
TERRAIN_LOSS = 1.0
# NOTE: only obstacle loss should be over obscale_loss defined in costmap_cfg.py

# original coco meta
VIPLANNER_SEM_META = [
    # TRAVERSABLE SPACE ###
    # traversable intended
    {
        "name": "sidewalk",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 255, 0],
        "ground": True,
    },
    {
        "name": "crosswalk",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 102, 0],
        "ground": True,
    },
    {
        "name": "floor",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 204, 0],
        "ground": True,
    },
    {
        "name": "stairs",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 153, 0],
        "ground": True,
    },
    # traversable not intended
    {
        "name": "gravel",
        "loss": TRAVERSABLE_UNINTENDED_LOSS,
        "color": [204, 255, 0],
        "ground": True,
    },
    {
        "name": "sand",
        "loss": TRAVERSABLE_UNINTENDED_LOSS,
        "color": [153, 204, 0],
        "ground": True,
    },
    {
        "name": "snow",
        "loss": TRAVERSABLE_UNINTENDED_LOSS,
        "color": [204, 102, 0],
        "ground": True,
    },
    {
        "name": "indoor_soft",  # human made thing, can be walked on
        "color": [102, 153, 0],
        "loss": TERRAIN_LOSS,
        "ground": False,
    },
    {
        "name": "terrain",
        "color": [255, 255, 0],
        "loss": TERRAIN_LOSS,
        "ground": True,
    },
    {
        "name": "road",
        "loss": ROAD_LOSS,
        "color": [255, 128, 0],
        "ground": True,
    },
    # OBSTACLES ###
    # human
    {
        "name": "person",
        "color": [255, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "anymal",
        "color": [204, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # vehicle
    {
        "name": "vehicle",
        "color": [153, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "on_rails",
        "color": [51, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "motorcycle",
        "color": [102, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "bicycle",
        "color": [102, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # construction
    {
        "name": "building",
        "loss": OBSTACLE_LOSS,
        "color": [127, 0, 255],
        "ground": False,
    },
    {
        "name": "wall",
        "color": [102, 0, 204],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "fence",
        "color": [76, 0, 153],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "bridge",
        "color": [51, 0, 102],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "tunnel",
        "color": [51, 0, 102],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # object
    {
        "name": "pole",
        "color": [0, 0, 255],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "traffic_sign",
        "color": [0, 0, 153],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "traffic_light",
        "color": [0, 0, 204],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "bench",
        "color": [0, 0, 102],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # nature
    {
        "name": "vegetation",
        "color": [153, 0, 153],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "water_surface",
        "color": [204, 0, 204],
        "loss": OBSTACLE_LOSS,
        "ground": True,
    },
    # sky
    {
        "name": "sky",
        "color": [102, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "background",
        "color": [102, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # void outdoor
    {
        "name": "dynamic",
        "color": [32, 0, 32],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "static",  # also everything unknown
        "color": [0, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # indoor
    {
        "name": "furniture",
        "color": [0, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "door",
        "color": [153, 153, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "ceiling",
        "color": [25, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
]


class VIPlannerSemMetaHandler:
    """Useful functions for handling VIPlanner semantic meta data."""

    def __init__(self) -> None:
        # meta config
        self.meta = VIPLANNER_SEM_META

        # class loss dict
        self.class_loss: dict = self._get_class_loss_dict()
        self.class_color: dict = self._get_class_color_dict()
        self.class_ground: dict = self._get_class_ground_dict()
        self.class_id: dict = self._get_class_id_dict()
        return

    def get_colors_for_names(self, name_list: list) -> list:
        """Get list of colors for a list of names."""
        colors = []
        name_to_color = {nc["name"]: nc["color"] for nc in self.meta}
        for name in name_list:
            if name in name_to_color:
                colors.append(name_to_color[name])
        return colors

    def _get_class_loss_dict(self) -> dict:
        """Get class loss dict."""
        return {nc["name"]: nc["loss"] for nc in self.meta}

    def _get_class_color_dict(self) -> dict:
        """Get class color dict."""
        return {nc["name"]: nc["color"] for nc in self.meta}

    def _get_class_ground_dict(self) -> dict:
        """Get class ground dict."""
        return {nc["name"]: nc["ground"] for nc in self.meta}

    def _get_class_id_dict(self) -> dict:
        """Get class id dict."""
        return {nc["name"]: i for i, nc in enumerate(self.meta)}

    @property
    def colors(self) -> list:
        """Get list of colors."""
        return list(self.class_color.values())

    @property
    def losses(self) -> list:
        """Get list of losses."""
        return list(self.class_loss.values())

    @property
    def names(self) -> list:
        """Get list of names."""
        return list(self.class_loss.keys())

    @property
    def ground(self) -> list:
        """Get list of ground."""
        return list(self.class_ground.values())
