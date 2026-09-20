"""Persistent-scene PointGoal evaluator using the pinned X-NavDP runtime."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import threading
import time
import traceback

import numpy as np
import torch

from navbench.evaluator_process import EVENT_PREFIX
from navbench.policy_pool import PolicyPool
from navbench.trajectory_trace import PlanRecord, TrajectoryTraceWriter


# A 0.5 s state/control cadence is sufficient to diagnose stopped motion,
# oscillation, and MPC tracking while keeping the evaluator's GPU hot path free
# of per-step device-to-host copies.  Every policy/MPC replan is still exact.
TRACE_SAMPLE_INTERVAL_STEPS = 5


def emit(event: str, **values: object) -> None:
    print(
        EVENT_PREFIX + json.dumps({"event": event, **values}, separators=(",", ":")),
        flush=True,
    )


def parse_observations(observation, obs_mapping) -> dict[str, np.ndarray]:
    if hasattr(obs_mapping, "__dict__"):
        obs_mapping = vars(obs_mapping)
    parsed = {
        key: observation[value].cpu().numpy()
        for key, value in obs_mapping.items()
        if value in observation
    }
    for key in (
        "body_to_world",
        "camera_to_body",
        "camera_intrinsics",
        "timestamps",
        "planning_goal",
    ):
        parsed[key] = observation[key].cpu().numpy()
    return parsed


def add_robot_state(observation, env, math_utils, camera_mount):
    observation = dict(observation)
    robot = env.unwrapped.scene["robot"]
    robot_rotation = math_utils.matrix_from_quat(robot.data.root_quat_w)
    goal_position = env.unwrapped._goal_pos_w
    observation["goal_pose"] = torch.bmm(
        robot_rotation.transpose(1, 2),
        (goal_position - robot.data.root_pos_w).unsqueeze(-1),
    ).squeeze(-1)
    observation["robot_pose"] = robot.data.root_pos_w
    observation["robot_rot"] = robot.data.root_quat_w[:, [1, 2, 3, 0]]
    camera = env.unwrapped.scene.sensors["camera_sensor"]
    sensor = camera.data
    batch = robot_rotation.shape[0]
    pose = torch.eye(4, device=robot_rotation.device).repeat(batch, 1, 1)
    yaw = torch.atan2(robot_rotation[:, 1, 0], robot_rotation[:, 0, 0])
    c, sn = yaw.cos(), yaw.sin()
    pose[:, 0, 0], pose[:, 0, 1], pose[:, 1, 0], pose[:, 1, 1] = c, -sn, sn, c
    pose[:, :3, 3] = robot.data.root_pos_w
    # Camera child XForm poses can remain at their authored USD location while
    # PhysX/Fabric moves the parent. Compose the measured rigid mount with the
    # live root-link attitude, including roll/pitch, instead of reading that cache.
    root_to_planning = pose[:, :3, :3].transpose(1, 2) @ robot_rotation
    extrinsic = torch.eye(4, device=pose.device).repeat(batch, 1, 1)
    extrinsic[:, :3, :3] = root_to_planning @ camera_mount[:3, :3]
    extrinsic[:, :3, 3] = root_to_planning @ camera_mount[:3, 3]
    observation["body_to_world"] = pose
    observation["camera_to_body"] = extrinsic
    observation["camera_intrinsics"] = sensor.intrinsic_matrices
    observation["timestamps"] = camera._timestamp_last_update.double().clone()
    observation["planning_goal"] = (
        pose[:, :3, :3].transpose(1, 2) @ (goal_position - pose[:, :3, 3]).unsqueeze(-1)
    ).squeeze(-1)[:, :2]

    return observation


def write_metrics(metrics: list[dict[str, float | int]], path: Path) -> None:
    ordered = sorted(metrics, key=lambda row: int(row["episode_idx"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered[0].keys())
        writer.writeheader()
        writer.writerows(ordered)


def trace_sample_env_ids(
    current_episode: list[int | None],
    episode_steps: np.ndarray,
) -> list[int]:
    """Return active environments at the fixed diagnostic sample cadence."""
    return [
        env_id
        for env_id, episode_idx in enumerate(current_episode)
        if episode_idx is not None
        and episode_steps[env_id] % TRACE_SAMPLE_INTERVAL_STEPS == 0
    ]


class Planner:
    """Infer asynchronously; track the latest path at the current control pose."""

    def __init__(self, policy_pool: PolicyPool, mpc_controller, num_envs: int) -> None:
        self.policy_pool = policy_pool
        self.mpc_controller = mpc_controller
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.input_observation = None
        self.output_plan = None
        self.output_version = 0
        self.episode_generation = np.zeros(num_envs, dtype=np.int64)
        self.active_envs = np.ones(num_envs, dtype=bool)
        self.error: BaseException | None = None
        self.plan_records: list[PlanRecord] = []
        self.thread = threading.Thread(target=self._run, name="pointgoal-planner", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                with self.input_lock:
                    observation = self.input_observation
                    generation = self.episode_generation.copy()
                    self.input_observation = None
                if observation is not None:
                    started = time.perf_counter()
                    trajectory = self.policy_pool.step(observation)
                    trajectory = np.concatenate(
                        (np.zeros_like(trajectory[:, :1]), trajectory), axis=1
                    )
                    with self.output_lock:
                        self.output_plan = (
                            trajectory, observation["body_to_world"], generation,
                            time.perf_counter() - started,
                        )
                self.stop_event.wait(0.01)
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()

    def submit(self, observation: dict[str, np.ndarray]) -> None:
        with self.input_lock:
            self.input_observation = observation

    def pop_action(
        self, observation: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, int, int] | None:
        if self.error is not None:
            raise RuntimeError("policy planning failed") from self.error
        with self.output_lock:
            plan = self.output_plan
        if plan is None:
            return None
        trajectory, source_pose, generation, policy_seconds = plan
        active_ids = np.flatnonzero((generation == self.episode_generation) & self.active_envs)
        if not len(active_ids):
            return None
        current = observation["body_to_world"][active_ids]
        source = source_pose[active_ids]
        # Both poses use gravity-aligned planning axes. Apply T_current^-1 T_capture.
        world = (
            np.einsum("bij,bpj->bpi", source[:, :3, :3], trajectory[active_ids])
            + source[:, None, :3, 3]
        )
        local = np.einsum(
            "bji,bpj->bpi", current[:, :3, :3], world - current[:, None, :3, 3]
        )
        # An all-zero policy path requests a stop, not a return to its capture
        # position. Keep that reference at the current origin across latency.
        stop = ~np.any(trajectory[active_ids] != 0, axis=(1, 2))
        local[stop] = 0
        started = time.perf_counter()
        controls, states, desired_speed, maximum_curvature = self.mpc_controller.solve(local)
        mpc_seconds = time.perf_counter() - started
        action = np.zeros((len(generation), 2), dtype=controls.dtype)
        action[active_ids] = controls[:, 0]
        self.output_version += 1
        for row, env_id in enumerate(active_ids):
            self.plan_records.append(PlanRecord(
                env_id=int(env_id), version=self.output_version, generation=int(generation[env_id]),
                point_goal=observation["pointgoal"][env_id].copy(),
                robot_position=observation["robot_pos"][env_id].copy(),
                robot_quaternion=observation["robot_quat"][env_id].copy(),
                trajectory=local[row].copy(), controls=controls[row].copy(),
                predicted_states=states[row].copy(), desired_speed=np.asarray(desired_speed[row]).copy(),
                maximum_curvature=np.asarray(maximum_curvature[row]).copy(),
                policy_seconds=policy_seconds, mpc_seconds=mpc_seconds,
            ))
        return action, self.output_version, 0

    def drain_plan_records(self) -> list[PlanRecord]:
        records = self.plan_records
        self.plan_records = []
        return records

    def reset_env(self, env_id: int, active: bool) -> int:
        with self.input_lock:
            self.input_observation = None
            self.episode_generation[env_id] += 1
            self.active_envs[env_id] = active
            return int(self.episode_generation[env_id])

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("policy planning failed") from self.error


class SceneSession:
    """Run independent checkpoints against one resident simulator scene."""

    def __init__(self, env, controller, cfg, house_id: str, math_utils) -> None:
        from eval.environment import BatchMPCNEWController, namespace_to_dict
        from pxr import UsdGeom

        self.env = env
        self.controller = controller
        self.cfg = cfg
        self.house_id = house_id
        self.math_utils = math_utils
        camera = env.unwrapped.scene.sensors["camera_sensor"]
        prim = camera._sensor_prims[0].GetPrim()
        transforms = UsdGeom.XformCache()
        # Keep the ancestor-scaled metric translation, but remove scale from
        # both world frames before forming the rigid camera-to-root transform.
        camera_world = transforms.GetLocalToWorldTransform(prim).RemoveScaleShear()
        root_world = transforms.GetLocalToWorldTransform(prim.GetParent()).RemoveScaleShear()
        mount = np.asarray(camera_world * root_world.GetInverse()).T.copy()
        mount[:3, 1:3] *= -1  # OpenGL optical axes -> ROS optical axes.
        self.camera_mount = torch.as_tensor(mount, dtype=torch.float32, device=env.device)
        self.mpc_controller = BatchMPCNEWController(
            batch=cfg.environment.num_envs,
            **namespace_to_dict(cfg.mpc),
        )

    @staticmethod
    def _observations(reset_outputs):
        if isinstance(reset_outputs, tuple) and len(reset_outputs) == 2:
            raw_observations, infos = reset_outputs
            return infos.get("observations", raw_observations)
        return reset_outputs[0] if isinstance(reset_outputs, tuple) else reset_outputs

    def run(self, server_ports: list[int], episodes: int, metric_path: Path) -> int:
        self.env.unwrapped._next_sample_idx = 0
        observations = self._observations(self.env.reset())
        observations = add_robot_state(observations, self.env, self.math_utils, self.camera_mount)

        intrinsic = self.env.unwrapped.scene.sensors[
            "camera_sensor"
        ].data.intrinsic_matrices[0]
        sample_indices = [
            int(self.env.unwrapped._sample_idx[index].item())
            for index in range(self.env.num_envs)
        ]
        policy_pool = PolicyPool(server_ports, self.env.num_envs)
        try:
            policy_contract = policy_pool.reset_all(
                intrinsic.cpu().numpy(),
                sample_indices,
                scene_name=self.house_id,
            )
        except BaseException:
            policy_pool.close()
            raise

        context_buffer = None
        if policy_contract["algo"] == "curvenav":
            from curvenav.config import DataConfig
            from curvenav.data.observation import DepthContextBuffer

            context_buffer = DepthContextBuffer(DataConfig(**policy_contract["observation_config"]))
            context_buffer.reset(self.env.num_envs)

        def capture(observation):
            parsed = parse_observations(observation, self.cfg.obs_mapping)
            if context_buffer is not None:
                context = context_buffer.update(
                    parsed["depth"], parsed["body_to_world"],
                    parsed["camera_intrinsics"], parsed["camera_to_body"], parsed["timestamps"],
                )
                parsed.update({"depth_context_" + name: value for name, value in context.items()})
            return parsed

        total = min(episodes, int(self.env.unwrapped._total_episode_count))
        current_episode = [
            sample if sample < total else None for sample in sample_indices
        ]
        planner = Planner(policy_pool, self.mpc_controller, self.env.num_envs)
        planner.active_envs[:] = np.asarray(
            [sample is not None for sample in current_episode]
        )
        planner.start()
        parsed_observation = capture(observations)
        planner.submit(parsed_observation)

        goal_distance = torch.linalg.vector_norm(
            observations["goal_pose"][:, :2], dim=-1
        )
        trajectory_length = torch.zeros_like(goal_distance)
        episode_steps = np.zeros(self.env.num_envs, dtype=np.int32)
        metrics: list[dict[str, float | int]] = []
        completed: set[int] = set()
        traces = TrajectoryTraceWriter(
            metric_path.parent / "trajectory_traces", self.env.num_envs
        )
        generation_episode: dict[tuple[int, int], int] = {}
        for env_id, episode_idx in enumerate(current_episode):
            if episode_idx is not None:
                traces.start_episode(env_id, episode_idx, goal_distance[env_id])
                generation_episode[(env_id, 0)] = episode_idx

        def record_completed_plans() -> None:
            for record in planner.drain_plan_records():
                generation = int(record.generation)
                if (record.env_id, generation) not in generation_episode:
                    continue
                traces.record_plan(
                    record.env_id,
                    int(episode_steps[record.env_id]),
                    record,
                )

        try:
            while len(completed) < total:
                planned_action = planner.pop_action(parsed_observation)
                if planned_action is None:
                    action = np.zeros((self.env.num_envs, 2))
                    plan_version = -1
                    plan_action_index = -1
                else:
                    action, plan_version, plan_action_index = planned_action
                record_completed_plans()
                robot_action = torch.as_tensor(
                    self.controller.forward_batch(observations["policy"], action),
                    device=observations["policy"].device,
                )
                planar_speed = torch.linalg.vector_norm(
                    observations["policy"][:, :2], dim=-1
                )
                trajectory_length.add_(planar_speed * self.env.unwrapped.step_dt)

                sampled_env_ids = trace_sample_env_ids(current_episode, episode_steps)
                if sampled_env_ids:
                    sample_index = torch.as_tensor(
                        sampled_env_ids, device=planar_speed.device
                    )
                    low_level_action = (
                        robot_action[sample_index]
                        if isinstance(robot_action, torch.Tensor)
                        else torch.as_tensor(
                            np.asarray(robot_action)[sampled_env_ids],
                            device=planar_speed.device,
                        )
                    )
                    trace_state = (
                        torch.cat(
                            (
                                observations["robot_pose"][sample_index],
                                observations["robot_rot"][sample_index],
                                observations["goal_pose"][sample_index],
                                planar_speed[sample_index, None],
                                low_level_action,
                            ),
                            dim=-1,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    for row, env_id in enumerate(sampled_env_ids):
                        traces.record_step(
                            env_id=env_id,
                            time_seconds=(
                                episode_steps[env_id] * self.env.unwrapped.step_dt
                            ),
                            robot_position=trace_state[row, :3],
                            robot_quaternion=trace_state[row, 3:7],
                            point_goal=trace_state[row, 7:10],
                            planar_speed=trace_state[row, 10],
                            mpc_command=action[env_id],
                            low_level_action=trace_state[row, 11:13],
                            plan_version=plan_version,
                            plan_action_index=plan_action_index,
                        )
                for env_id, episode_idx in enumerate(current_episode):
                    if episode_idx is not None:
                        episode_steps[env_id] += 1

                terminal_robot_position = observations["robot_pose"].clone()
                terminal_robot_quaternion = observations["robot_rot"].clone()
                terminal_point_goal = observations["goal_pose"].clone()

                step_outputs = self.env.step(robot_action)
                if len(step_outputs) == 5:
                    observations, _, terminated, truncated, infos = step_outputs
                    dones = terminated | truncated
                else:
                    raw_observations, _, dones, infos = step_outputs
                    observations = infos.get("observations", raw_observations)
                observations = add_robot_state(observations, self.env, self.math_utils, self.camera_mount)
                record_completed_plans()

                done_env_ids = torch.nonzero(dones, as_tuple=False).flatten()
                done_env_ids = done_env_ids.detach().cpu().tolist()
                if done_env_ids:
                    done_index = torch.as_tensor(
                        done_env_ids, device=goal_distance.device
                    )
                    terminal_state = (
                        torch.cat(
                            (
                                terminal_robot_position[done_index],
                                terminal_robot_quaternion[done_index],
                                terminal_point_goal[done_index],
                            ),
                            dim=-1,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    terminal_lengths = trajectory_length[done_index].cpu().numpy()
                    time_outs = infos["time_outs"][done_index].detach().cpu().numpy()
                for row, env_id in enumerate(done_env_ids):
                    finished = current_episode[env_id]
                    next_sample = int(self.env.unwrapped._sample_idx[env_id].item())
                    active = next_sample < total and next_sample not in completed
                    generation = planner.reset_env(env_id, active)
                    policy_pool.reset_env(env_id, next_sample, self.house_id)
                    if context_buffer is not None:
                        context_buffer.reset_env(env_id)

                    if finished is not None and finished not in completed:
                        success = 1.0 - float(time_outs[row])
                        path_length = float(terminal_lengths[row])
                        initial_goal_distance = float(goal_distance[env_id].item())
                        metrics.append(
                            {
                                "success": success,
                                "spl": success
                                * initial_goal_distance
                                / max(path_length, initial_goal_distance, 1e-8),
                                "distance": initial_goal_distance,
                                "episode_idx": finished,
                            }
                        )
                        completed.add(finished)
                        write_metrics(metrics, metric_path)
                        traces.finish_episode(
                            env_id=env_id,
                            success=success,
                            path_length=path_length,
                            elapsed_seconds=(
                                episode_steps[env_id] * self.env.unwrapped.step_dt
                            ),
                            final_robot_position=terminal_state[row, :3],
                            final_robot_quaternion=terminal_state[row, 3:7],
                            final_point_goal=terminal_state[row, 7:10],
                        )

                    current_episode[env_id] = next_sample if active else None
                    goal_distance[env_id] = torch.linalg.vector_norm(
                        observations["goal_pose"][env_id, :2]
                    )
                    trajectory_length[env_id] = 0.0
                    episode_steps[env_id] = 0
                    if active:
                        generation_episode[(env_id, generation)] = next_sample
                        traces.start_episode(env_id, next_sample, goal_distance[env_id])

                if len(completed) < total:
                    parsed_observation = capture(observations)
                    planner.submit(parsed_observation)
        finally:
            try:
                planner.close()
            finally:
                policy_pool.close()
        return total

    def close(self) -> None:
        self.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--portable-root", required=True)
    args, kit_args = parser.parse_known_args()
    invalid = [argument for argument in kit_args if not argument.startswith("--/")]
    if invalid:
        parser.error(f"unknown evaluator arguments: {' '.join(invalid)}")
    return args


def main() -> None:
    startup_started = time.monotonic()
    args = parse_args()
    control_path = Path(os.environ.pop("NAVBENCH_CONTROL_SOCKET"))
    # Use the unified environment's pinned Warp before Kit registers its bundled
    # 1.7.1 module, whose cuDeviceGetUuid query uses the invalid version 110400.
    import warp
    warp.init()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        device=args.device,
        multi_gpu=False,
    )
    simulation_app = launcher.app
    import isaaclab.utils.math as math_utils
    from eval.config_utils import load_default_config
    from eval.environment import create_environment

    cfg = load_default_config(args.config_file)
    env, controller, house_id = create_environment(
        cfg, scene_index=args.scene_index, device=args.device
    )
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    composition_errors = {
        str(error)
        for root in (stage.GetPseudoRoot(), *stage.GetPrototypes())
        for prim in Usd.PrimRange(root)
        for error in prim.GetPrimIndex().localErrors
    }
    if composition_errors:
        raise RuntimeError(
            "Incomplete USD scene composition; refusing evaluation:\n"
            + "\n".join(sorted(composition_errors))
        )
    session = SceneSession(env, controller, cfg, house_id, math_utils)
    control_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    control_server.bind(str(control_path))
    control_server.listen(1)
    emit(
        "ready",
        scene=house_id,
        num_envs=env.num_envs,
        camera_mount_ros=session.camera_mount.cpu().tolist(),
        startup_seconds=time.monotonic() - startup_started,
    )
    try:
        control_socket, _ = control_server.accept()
        with control_socket, control_socket.makefile(
            "r", encoding="utf-8"
        ) as control_stream:
            for line in control_stream:
                command = json.loads(line)
                if command["command"] == "close":
                    break
                if command["command"] != "run":
                    raise ValueError(
                        f"unknown evaluator command: {command['command']!r}"
                    )
                metric_path = Path(command["metric_path"])
                metric_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                completed = session.run(
                    [int(port) for port in command["server_ports"]],
                    int(command["episodes"]),
                    metric_path,
                )
                emit(
                    "done",
                    run_id=command["run_id"],
                    episodes=completed,
                    elapsed_seconds=time.monotonic() - started,
                    metric_path=str(metric_path),
                )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        control_server.close()
        control_path.unlink(missing_ok=True)
        session.close()
        simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
