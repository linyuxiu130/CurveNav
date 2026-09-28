"""Render one actual OmniGibson scene from above with its ceiling hidden."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from scene import load_scene, training_instance


def render(evaluator, output, width, height):
    import omnigibson as og
    from omnigibson.sensors.vision_sensor import VisionSensor

    scene = evaluator.env.scene
    floors = [obj for obj in scene.objects if obj.category == "floors"]
    if not floors:
        raise RuntimeError("Scene has no floor geometry to frame")
    hidden_categories = {"ceilings", "roof"}
    visible = [obj for obj in scene.objects if obj.category not in hidden_categories]
    bounds = np.array([[lo.cpu().numpy(), hi.cpu().numpy()] for obj in visible for lo, hi in [obj.aabb]])
    floor_top = max(float(obj.aabb[1][2]) for obj in floors)
    lower = bounds[:, 0, :2].min(axis=0)
    upper = bounds[:, 1, :2].max(axis=0)
    center = (lower + upper) / 2
    span = upper - lower

    hidden = [obj for obj in scene.objects if obj.category in hidden_categories]
    for obj in hidden:
        obj.visible = False

    camera = VisionSensor(
        relative_prim_path="/rendered_bev_camera",
        name="rendered_bev_camera",
        modalities="rgb",
        image_width=width,
        image_height=height,
        focal_length=17.0,
    )
    camera.load(None)
    camera.initialize()
    world_width = max(span[0] * 1.12, span[1] * 1.12 * width / height)
    camera_z = float(floor_top + world_width * camera.focal_length / camera.horizontal_aperture)
    camera.set_position_orientation(
        position=torch.tensor([center[0], center[1], camera_z], dtype=torch.float32),
        orientation=torch.tensor([0., 0., 0., 1.]),
    )
    for _ in range(6):
        og.sim.render()
    rgb = camera.get_obs()[0]["rgb"].cpu().numpy()[..., :3].astype(np.uint8)
    if rgb.max() == rgb.min():
        raise RuntimeError("Overhead camera produced a blank image")

    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(output.stem + "_raw.png")
    cv2.imwrite(str(raw), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    position, _ = evaluator.robot.get_position_orientation()
    robot = position.cpu().numpy()
    k = camera.intrinsic_matrix.cpu().numpy()
    depth = camera_z - (float(robot[2]) + 1.0)
    x = int(round(k[0, 0] * (robot[0] - center[0]) / depth + k[0, 2]))
    y = int(round(k[1, 1] * (center[1] - robot[1]) / depth + k[1, 2]))
    marked = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(marked, (x, y), 30, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.circle(marked, (x, y), 25, (45, 67, 255), 5, cv2.LINE_AA)
    cv2.circle(marked, (x, y), 5, (45, 67, 255), -1, cv2.LINE_AA)
    cv2.putText(marked, "R1Pro", (x + 36, y - 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(marked, "R1Pro", (x + 36, y - 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (45, 67, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), marked)
    print({"rendered": str(output), "raw": str(raw), "scene": evaluator.env.task.scene_name,
           "robot_xy": robot[:2].tolist(), "hidden_ceiling_objects": len(hidden),
           "camera_z": camera_z}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--instance", type=training_instance, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    with load_scene(args.task, args.instance) as evaluator:
        render(evaluator, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
