"""Persistent BEHAVIOR scene with browser-controlled CurveNav navigation."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
import torch

from curvenav.data.observation import DepthContextBuffer
from curvenav.deployment.runtime import load_policy
RUN = Path("/shibo_huang/data/curvenav/datasets/behavior_task_goal_224_v1/h800_runs/train10_9303_20260924")
CHECKPOINT = RUN / "training/best.pt"
CONFIG = RUN / "training/config.yaml"
HTML = Path(__file__).with_name("index.html")
PORT = 8765
RESOLUTION = 0.05
SIM_DT = 1 / 30
OBSERVE_EVERY = 3
TOP_WIDTH, TOP_HEIGHT = 960, 600


def _matrix(position, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    result[:3, 3] = position
    return result


def _encode_image(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("ascii")


class LiveViewer:
    def __init__(self, evaluator, task, instance):
        import omnigibson as og
        from omnigibson.utils import transform_utils as T
        from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
        from collect import DATA
        from robot import (
            camera_calibration,
            configure_navigation_camera,
            hold_action,
            navigation_action,
            robot_contract,
        )
        from route import build_map
        from curvenav.data.obstacle_memory_cuda import CudaObstacleMemory
        self.og, self.T = og, T
        self.detect_collision = detect_robot_collision_in_sim
        self.camera_calibration = camera_calibration
        self.navigation_action = navigation_action
        self.hold = hold_action(evaluator.robot)
        self.evaluator = evaluator
        self.robot = evaluator.robot
        self.task, self.instance = task, instance
        plans = Path("/shibo_huang/data/curvenav/datasets/behavior_task_goal_224_v1/plans")
        self.task_index = next(
            (int(path.stem.split("_")[-1]) for path in plans.glob("task_*.json")
             if json.loads(path.read_text()).get("task") == task),
            -1,
        )
        self.camera = configure_navigation_camera(evaluator, DATA.image_width, DATA.image_height)
        if not {"rgb", "depth_linear"}.issubset(self.camera.modalities):
            raise RuntimeError("The head camera must provide aligned RGB and linear depth")

        self.session = Path(os.environ.get("CURVENAV_LIVE_OUTPUT", "/tmp/curvenav-behavior-live"))
        self.session.mkdir(parents=True, exist_ok=True)
        self.robot_info = robot_contract(evaluator, self.session)
        self.grid = build_map(evaluator, self.robot_info, full_scene=True)
        np.savez_compressed(self.session / "scene_map.npz", **self.grid)
        self.free = self.grid["free"]
        self.origin = self.grid["origin_xy"]
        self.height, self.width = self.free.shape
        self.map_image = self._make_map_image()
        self.top_camera, self.render_bounds = self._make_top_camera()
        self.config, self.policy = load_policy(str(CHECKPOINT), str(CONFIG), "cuda:0")
        self.context = DepthContextBuffer(
            self.config.data,
            obstacle_memory_factory=lambda h, m, g: CudaObstacleMemory(h, m, g, device=0),
        )
        self.context.reset(1)
        self.goal = None
        self.selected_path = None
        self.trail = []
        self.depth_jpeg = ""
        self.rgb_jpeg = ""
        self.top_jpeg = ""
        self.top_frame = 0
        self.input_audit = {}
        self.paused = True
        self.stop_reason = "等待目标"
        self.status = "场景已加载；点击俯视图设置目标"
        self.error = None
        self.last_velocity = np.zeros(3)
        self.state_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.pending = None
        self._randomize_robot()
        self._observe_and_plan()
        self._publish("场景就绪，仿真常驻；点击俯视图开始导航")

    def _make_map_image(self):
        support = self.grid["layout_support"] & self.grid["physical_support"].astype(bool)
        image = np.full((self.height, self.width, 3), (22, 27, 36), np.uint8)
        image[support] = (230, 234, 239)
        image[self.grid["occupied"].astype(bool)] = (50, 59, 76)
        image[self.free] = (248, 249, 251)
        image = np.flipud(image)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode the BEV map")
        return encoded.tobytes()

    def _make_top_camera(self):
        from omnigibson.sensors import VisionSensor

        support = self.grid["layout_support"] & self.grid["physical_support"]
        cells = np.argwhere(support)
        if not len(cells):
            raise RuntimeError("No floor support for the rendered scene camera")
        xmin, xmax = self.origin[0] + cells[:, 1].min() * RESOLUTION, self.origin[0] + cells[:, 1].max() * RESOLUTION
        ymin, ymax = self.origin[1] + cells[:, 0].min() * RESOLUTION, self.origin[1] + cells[:, 0].max() * RESOLUTION
        center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        span_x = max(xmax - xmin + 4., (ymax - ymin + 4.) * TOP_WIDTH / TOP_HEIGHT)
        span_y = span_x * TOP_HEIGHT / TOP_WIDTH
        floor_z = float(self.grid["expected_base_z_m"])
        camera_z = floor_z + span_x * 17.0 / 20.995
        for obj in self.evaluator.env.scene.objects:
            if obj.category in {"ceilings", "roof"}:
                obj.visible = False
        camera = VisionSensor(
            relative_prim_path="/live_top_camera", name="live_top_camera",
            modalities=["rgb"], image_width=TOP_WIDTH, image_height=TOP_HEIGHT,
            focal_length=17.0, horizontal_aperture=20.995, viewport_name=None,
        )
        camera.load(None)
        camera.initialize()
        camera.set_position_orientation(
            position=[*center, camera_z], orientation=[0., 0., 0., 1.],
        )
        for _ in range(4):
            self.og.sim.render()
        self.og.sim.update_handles()
        return camera, [center[0] - span_x / 2, center[1] - span_y / 2,
                        center[0] + span_x / 2, center[1] + span_y / 2]

    def _publish(self, status=None):
        pos, quat = self.robot.get_position_orientation()
        yaw = float(self.T.quat2euler(quat)[2])
        with self.state_lock:
            if status is not None:
                self.status = status
            self.state = {
                "ready": True,
                "task": self.task,
                "task_index": self.task_index,
                "instance": self.instance,
                "scene": self.evaluator.env.task.scene_name,
                "origin": self.origin.tolist(),
                "map_shape": [self.height, self.width],
                "resolution": RESOLUTION,
                "robot": [float(pos[0]), float(pos[1]), yaw],
                "goal": self.goal,
                "selected_path": self.selected_path,
                "trail": self.trail[-2400:],
                "depth_jpeg": self.depth_jpeg,
                "rgb_jpeg": self.rgb_jpeg,
                "top_jpeg": self.top_jpeg,
                "top_frame": self.top_frame,
                "render_bounds": self.render_bounds,
                "render_shape": [TOP_HEIGHT, TOP_WIDTH],
                "input_audit": self.input_audit,
                "status": self.status,
                "error": self.error,
                "paused": self.paused,
                "stop_reason": self.stop_reason,
                "inference_ms": getattr(self, "inference_ms", None),
                "safe_fraction": getattr(self, "safe_fraction", None),
                "frame": getattr(self, "frame", 0),
            }

    def _randomize_robot(self):
        cells = np.argwhere(self.free)
        if not len(cells):
            raise RuntimeError("BEHAVIOR scene has no robot-safe spawn cells")
        position, _ = self.robot.get_position_orientation()
        rng = np.random.default_rng()
        for y, x in cells[rng.permutation(len(cells))[:min(len(cells), 128)]]:
            cell = (float(self.origin[0] + x * RESOLUTION),
                    float(self.origin[1] + y * RESOLUTION))
            yaw = float(rng.uniform(-math.pi, math.pi))
            pose = torch.tensor([*cell, float(position[2])], dtype=torch.float32)
            orientation = torch.tensor([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)])
            self.robot.set_position_orientation(pose, orientation)
            self.robot.keep_still()
            for _ in range(8):
                self.og.sim.step()
            if not self.detect_collision(self.robot, ignore_obj_in_hand=False):
                self._finish_placement(cell)
                return
        raise RuntimeError("Could not place R1Pro at a collision-free random spawn")

    def _finish_placement(self, point):
        self.spawn = tuple(point)
        self.goal = None
        self.selected_path = None
        self.trail = [list(point)]
        self.paused = True
        self.stop_reason = "等待目标"
        self.error = None
        self.last_velocity[:] = 0
        self.context.reset_env(0)

    def _place_robot(self, point, yaw):
        x = int(round((point[0] - self.origin[0]) / RESOLUTION))
        y = int(round((point[1] - self.origin[1]) / RESOLUTION))
        support = self.grid["physical_support"]
        if not (0 <= y < self.height and 0 <= x < self.width and support[y, x]):
            raise ValueError("放置点没有实际地板")
        previous_position, previous_orientation = self.robot.get_position_orientation()
        position = torch.tensor([point[0], point[1], float(self.grid["expected_base_z_m"])], dtype=torch.float32)
        orientation = torch.tensor([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)])
        self.paused = True
        self.robot.set_position_orientation(position, orientation)
        self.robot.keep_still()
        for _ in range(8):
            self.og.sim.step()
        if self.detect_collision(self.robot, ignore_obj_in_hand=False):
            self.robot.set_position_orientation(previous_position, previous_orientation)
            self.robot.keep_still()
            for _ in range(8):
                self.og.sim.step()
            self.context.reset_env(0)
            self.goal = None
            self.selected_path = None
            self.stop_reason = "等待目标"
            raise ValueError("放置点与场景物体发生碰撞，已恢复原位")
        self._finish_placement(point)

    def set_goal(self, point):
        self.goal = [float(point[0]), float(point[1])]
        self.error = None
        self.paused = False
        self.stop_reason = None
        self.status = "目标已交给模型；闭环导航中"

    def _observe_and_plan(self):
        self.og.sim.render()
        observation = self.camera.get_obs()[0]
        depth_m = observation["depth_linear"].detach().cpu().numpy()
        intrinsic, camera_to_body = self.camera_calibration(self.evaluator)
        intrinsic = intrinsic.detach().cpu().numpy().astype(np.float32)
        camera_to_body = camera_to_body.detach().cpu().numpy().astype(np.float64)
        pos, quat = self.robot.get_position_orientation()
        yaw = float(self.T.quat2euler(quat)[2])
        body_to_world = _matrix(pos.detach().cpu().numpy(), yaw)
        actual_rotation = self.T.quat2mat(quat).detach().cpu().numpy()
        planning_from_actual = np.eye(4)
        planning_from_actual[:3, :3] = body_to_world[:3, :3].T @ actual_rotation
        camera_to_body = planning_from_actual @ camera_to_body
        optical_flip = np.diag([1., -1., -1., 1.])
        camera_world = self.T.pose2mat(self.camera.get_position_orientation()).detach().cpu().numpy() @ optical_flip
        camera_pose_error = float(np.max(np.abs(body_to_world @ camera_to_body - camera_world)))
        if camera_pose_error > 1e-3:
            raise ValueError(f"Head camera extrinsics disagree with its world pose: {camera_pose_error:.5f}")
        raw_depth = depth_m
        while raw_depth.ndim > 3 and raw_depth.shape[0] == 1:
            raw_depth = raw_depth[0]
        if raw_depth.ndim == 3 and raw_depth.shape[0] == 1 and raw_depth.shape[-1] != 1:
            raw_depth = raw_depth[0]
        if raw_depth.ndim == 2:
            raw_depth = raw_depth[..., None]
        if raw_depth.ndim != 3 or raw_depth.shape[-1] != 1:
            raise ValueError(f"Head depth camera returned unsupported shape {depth_m.shape}")
        context = self.context.update(
            raw_depth[None].astype(np.float32),
            body_to_world[None].astype(np.float64),
            intrinsic.astype(np.float32)[None],
            camera_to_body.astype(np.float32)[None],
            np.array([float(self.og.sim.current_time)], dtype=np.float64),
        )
        ages = context["observation_age_s"][0]
        valid = context["observation_valid"][0]
        if (context["depth"].shape != (1, 4, 1, self.config.data.image_height, self.config.data.image_width)
                or not np.isfinite(context["depth"]).all()
                or context["depth"].min() < 0 or context["depth"].max() > 1
                or not np.allclose(context["camera_intrinsics"][0, -1], intrinsic, atol=1e-3)
                or not np.allclose(context["observation_to_current"][0, -1], np.eye(4), atol=1e-4)
                or not valid[-1] or abs(float(ages[-1])) > 1e-5
                or np.any(np.diff(ages[valid]) >= 0)):
            raise ValueError("Online depth, calibration, or temporal context violates the training contract")
        finite_depth = raw_depth[np.isfinite(raw_depth)]
        self.input_audit = {
            "depth_hw": [int(raw_depth.shape[0]), int(raw_depth.shape[1])],
            "depth_range_m": [round(float(finite_depth.min()), 3), round(float(finite_depth.max()), 3)] if finite_depth.size else [],
            "intrinsics": np.round(intrinsic, 3).tolist(),
            "valid_history": int(valid.sum()),
            "history_ages_s": np.round(ages, 3).tolist(),
            "camera_pose_error_m": round(camera_pose_error, 6),
            "sensor_time_s": round(float(self.og.sim.current_time), 3),
        }
        if self.goal is None and self.stop_reason == "等待目标":
            self.status = "场景就绪；点击俯视图设置目标"
        depth_vis = np.clip(np.nan_to_num(depth_m.squeeze(), nan=5., posinf=5.) / 5 * 255, 0, 255).astype(np.uint8)
        self.depth_jpeg = _encode_image(cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_TURBO))
        rgb = observation["rgb"].detach().cpu().numpy()
        self.rgb_jpeg = _encode_image(cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR))
        self.frame = getattr(self, "frame", 0) + 1
        if self.frame == 1 or self.frame % 2 == 0:
            top = self.top_camera.get_obs()[0]["rgb"].detach().cpu().numpy()
            self.top_jpeg = _encode_image(cv2.cvtColor(top[..., :3], cv2.COLOR_RGB2BGR))
            self.top_frame += 1
        if self.goal is None or self.paused:
            self.selected_path = None
            return
        delta = np.asarray(self.goal, dtype=np.float64) - body_to_world[:2, 3]
        goal_local = body_to_world[:2, :2].T @ delta
        started = time.perf_counter()
        if self.policy is None:
            raise RuntimeError("CurveNav policy is not loaded")
        device = torch.device("cuda:0")
        condition = {
            name: torch.as_tensor(value, device=device)
            for name, value in context.items()
        }
        condition["depth"] = condition["depth"].float()
        from curvenav.types import PolicyCondition
        prepared = PolicyCondition(
            point_goal=torch.as_tensor(goal_local[None], dtype=torch.float32, device=device),
            **condition,
        )
        prepared.validate()
        with torch.inference_mode():
            prediction = self.policy.sample(prepared)
        path = prediction.path[0, 1:].float().cpu().numpy()
        self.inference_ms = (time.perf_counter() - started) * 1000
        self.safe_fraction = float(self._path_safe_fraction(path[:, :2], body_to_world))
        self.selected_path = self._path_to_world(path[:, :2], body_to_world).tolist()

    def _path_safe_fraction(self, path, pose):
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.r_[0., np.cumsum(distances)]
        end = int(np.searchsorted(arc, 1.0, side="right"))
        path = path[:max(2, end)]
        world = self._path_to_world(path, pose)
        cells = np.rint((world - self.origin) / RESOLUTION).astype(int)
        inside = ((cells[:, 0] >= 0) & (cells[:, 0] < self.width)
                  & (cells[:, 1] >= 0) & (cells[:, 1] < self.height))
        safe = np.zeros(len(cells), dtype=bool)
        safe[inside] = self.free[cells[inside, 1], cells[inside, 0]]
        return float(safe.mean())

    @staticmethod
    def _path_to_world(path, pose):
        return np.asarray(path) @ pose[:2, :2].T + pose[:2, 3]

    def _follow_selected_path(self):
        from route import tracking_command

        path = self.selected_path
        if path is None or len(path) < 2:
            velocity = np.zeros(3)
        else:
            path = np.asarray(path, dtype=np.float64)
            segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
            arc = np.r_[0., np.cumsum(segment)]
            target_index = min(int(np.searchsorted(arc, .35)), len(path) - 1)
            target = path[target_index]
            position, quat = self.robot.get_position_orientation()
            xy = position[:2].detach().cpu().numpy()
            yaw = float(self.T.quat2euler(quat)[2])
            heading = math.atan2(target[1] - xy[1], target[0] - xy[0])
            distance = float(np.linalg.norm(target - xy))
            speed = min(.22, .5 * distance)
            feedforward = np.array([speed * math.cos(heading), speed * math.sin(heading), 0.])
            velocity, _ = tracking_command(
                xy, yaw, target, heading, feedforward, self.last_velocity, SIM_DT,
            )
        self.last_velocity = velocity.copy()
        action = self.navigation_action(self.robot, self.hold, velocity)
        self.robot.apply_action(action)
        self.og.sim.step()
        if self.goal is not None and not self.paused:
            pos, _ = self.robot.get_position_orientation()
            point = [float(pos[0]), float(pos[1])]
            if not self.trail or math.dist(point, self.trail[-1]) >= .025:
                self.trail.append(point)

    def _run_command(self, command):
        if command["action"] == "goal":
            self.set_goal(command["point"])
        elif command["action"] == "randomize":
            self._randomize_robot()
            self.status = "机器人已在当前场景随机换位；场景保持常驻"
        elif command["action"] == "place":
            self._place_robot(command["point"], command["yaw_rad"])
            self.status = "机器人已放置；点击俯视图设置目标"
        elif command["action"] == "stop":
            self.paused = True
            self.stop_reason = "手动停车"
            self.last_velocity[:] = 0
            self.error = None
            self.status = "已停车；仿真场景保持运行"

    def loop(self):
        next_tick = time.perf_counter()
        # The constructor captures the first depth frame. Start the regular
        # three-step (10 Hz) capture cadence after one full interval.
        control_step = 0
        while True:
            with self.command_lock:
                command, self.pending = self.pending, None
            if command:
                try:
                    self._run_command(command)
                except Exception as exc:
                    self.error = str(exc)
                    if command["action"] == "goal" and isinstance(exc, ValueError):
                        self.status = "目标未更新；原导航继续运行" if not self.paused else "目标无效"
                    elif command["action"] == "place" and isinstance(exc, ValueError):
                        self.status = "放置失败"
                    else:
                        self.paused = True
                        self.stop_reason = "操作失败"
                        self.status = "操作失败"
            if control_step and control_step % OBSERVE_EVERY == 0:
                try:
                    self._observe_and_plan()
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.paused = True
                    self.stop_reason = "模型或传感器错误"
                    self.status = "模型/传感器错误；已停车"
            if self.goal is not None and not self.paused and self.selected_path is not None:
                try:
                    self._follow_selected_path()
                    if self.detect_collision(self.robot, ignore_obj_in_hand=False):
                        self.paused = True
                        self.stop_reason = "实际碰撞"
                        self.error = "仿真检测到机器人与场景物体发生接触"
                        self.status = "碰撞停车"
                    else:
                        pos, _ = self.robot.get_position_orientation()
                        reached = math.dist(pos[:2].detach().cpu().numpy(), self.goal) < .15
                        if reached:
                            self.goal = None
                            self.selected_path = None
                            self.paused = True
                            self.stop_reason = "已到达目标"
                            self.error = None
                            self.status = "已到达目标"
                            self.last_velocity[:] = 0
                except Exception as exc:
                    self.paused = True
                    self.stop_reason = "控制错误"
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.status = "导航停止"
            else:
                self.robot.apply_action(self.navigation_action(self.robot, self.hold, np.zeros(3)))
                self.og.sim.step()
            control_step += 1
            if control_step % OBSERVE_EVERY == 0:
                self._publish(self.status)
            next_tick += SIM_DT
            time.sleep(max(0., next_tick - time.perf_counter()))


def _serve(viewer, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif path == "/api/state":
                with viewer.state_lock:
                    state = dict(viewer.state)
                state.pop("depth_jpeg", None)
                state.pop("rgb_jpeg", None)
                state.pop("top_jpeg", None)
                body = json.dumps(state, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
            elif path in ("/api/depth.jpg", "/api/rgb.jpg", "/api/top.jpg"):
                key = {"/api/depth.jpg": "depth_jpeg", "/api/rgb.jpg": "rgb_jpeg",
                       "/api/top.jpg": "top_jpeg"}[path]
                with viewer.state_lock:
                    body = base64.b64decode(viewer.state.get(key, ""))
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
            elif path == "/api/map.png":
                body = viewer.map_image
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                command = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if command.get("action") not in ("goal", "randomize", "place", "stop"):
                    raise ValueError("未知操作")
                if command["action"] in ("goal", "place"):
                    point = command.get("point")
                    if not isinstance(point, list) or len(point) != 2 or not np.isfinite(point).all():
                        raise ValueError("目标坐标无效")
                if command["action"] == "place":
                    yaw = command.get("yaw_rad")
                    if not isinstance(yaw, (int, float)) or not math.isfinite(yaw) or abs(yaw) > math.pi:
                        raise ValueError("机器人朝向需在 -180° 到 180° 之间")
                with viewer.command_lock:
                    viewer.pending = command
                body = b'{"queued":true}'
                self.send_response(202)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    from omnigibson.macros import gm
    from scene import load_scene

    gm.HEADLESS = True
    np.random.seed(0)
    torch.manual_seed(0)
    with load_scene(args.task, args.instance) as evaluator:
        viewer = LiveViewer(evaluator, args.task, args.instance)
        server = threading.Thread(
            target=_serve, args=(viewer, args.host, args.port), daemon=True,
        )
        server.start()
        print(json.dumps({
            "event": "viewer_ready",
            "url": f"http://{args.host}:{args.port}",
            "task": args.task,
            "instance": args.instance,
            "scene": evaluator.env.task.scene_name,
            "map_shape": [viewer.height, viewer.width],
            "checkpoint": str(CHECKPOINT),
        }), flush=True)
        viewer.loop()


if __name__ == "__main__":
    main()
