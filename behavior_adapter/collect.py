"""Collect task-grounded, physically supported BEHAVIOR expert navigation."""
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation

from curvenav.config import DataConfig
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data.depth import depth_camera_contract


DATA = DataConfig(embodiment='r1pro', image_height=224, robot_radius_m=.42, obstacle_min_z_m=.02,
                  obstacle_max_z_m=1.50, camera_forward_offset_m=.239,
                  camera_height_m=1.393, camera_downward_pitch_degrees=22.5)


def export_route(directory, grid):
    """Convert world XY to the existing source-chart X,-Y; keep SE(3) right-handed."""
    r = np.load(directory/'trajectory.npz')
    indices = r['depth_frame_steps']
    positions = r['position_world_m'][indices]
    yaw = -r['yaw_rad'][indices].astype(np.float32)
    poses = np.repeat(np.eye(4)[None], len(indices), axis=0)
    c, s = np.cos(yaw), np.sin(yaw)
    poses[:, 0, 0] = poses[:, 1, 1] = c
    poses[:, 0, 1], poses[:, 1, 0] = s, -s
    poses[:, :3, 3] = positions
    actual_rotation = Rotation.from_quat(r['quaternion_xyzw'][indices]).as_matrix()
    planning_from_actual = np.repeat(np.eye(4)[None], len(indices), axis=0)
    planning_from_actual[:, :3, :3] = poses[:, :3, :3].transpose(0, 2, 1) @ actual_rotation
    extrinsic = planning_from_actual @ r['camera_to_body_optical']
    arrays = {'traj_xy': positions[:, :2]*[1, -1], 'traj_yaw': yaw,
              'body_to_world': poses, 'timestamps': r['time_s'][indices]-r['time_s'][0],
              'camera_intrinsics': np.broadcast_to(r['camera_intrinsics'], (len(indices), 3, 3)),
              'camera_to_body': extrinsic}
    for name, values in arrays.items():
        np.save(directory/f'{name}.npy', values.astype(np.float64 if name == 'timestamps' else np.float32))
    resolution = float(grid['resolution_m'])
    origin = np.array([grid['origin_xy'][0], -(grid['origin_xy'][1]+(grid['free'].shape[0]-1)*resolution)])-resolution/2
    np.savez_compressed(directory.parent/'navigation_grid.npz',
                        free=(grid['clearance_m'] >= 0)[::-1].T,
                        clearance_m=grid['clearance_m'][::-1].T.astype(np.float32),
                        origin_xy=origin, cell_size_m=resolution)
    return len(indices)


def write_manifest(root):
    manifest = {'schema': 'curvenav_policy_depth_routes_v5',
                'goal_contract': 'behavior_task_object_v1',
                'route_contract': {'navigation_geometry': expert_navigation_geometry_contract(DATA)},
                'camera': {'calibration': 'per_frame'}, 'observation': depth_camera_contract(DATA)}
    (root/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def sample_endpoints(grid, rng, band):
    """Sample within a connected component; narrow-band starts expose tight clearance."""
    from route import RESOLUTION, RoutePlanningError
    components, _ = label(grid['free'])
    cells = np.argwhere(grid['free'])
    if band == 'narrow':
        cells = cells[grid['clearance_m'][cells[:, 0], cells[:, 1]] < .30]
    if not len(cells):
        raise RoutePlanningError('No start cells for the requested band')
    lo, hi = {'near': (.5, 1.5), 'medium': (1.5, 5.), 'long': (5., 10.),
              'narrow': (1.5, 5.), 'any': (.5, 10.)}[band]
    for _ in range(32):
        start = cells[rng.integers(len(cells))]
        targets = np.argwhere(components == components[tuple(start)])
        distances = np.linalg.norm(targets-start, axis=1)*RESOLUTION
        targets = targets[(distances >= lo) & (distances <= hi)]
        if len(targets):
            goal = targets[rng.integers(len(targets))]
            return grid['origin_xy']+np.stack([start[::-1], goal[::-1]])*RESOLUTION
    raise RoutePlanningError('No connected goal in the requested distance band')


def near_duplicate(records, scene_name, instance, endpoints, tolerance_m=.25):
    """Reject repeated paths in the same loaded task instance, including reversals."""
    for record in records:
        if (record.get('scene_name') != scene_name or record.get('instance') != instance
                or 'planned_endpoints_xy_m' not in record):
            continue
        prior = np.asarray(record['planned_endpoints_xy_m'])
        forward = np.linalg.norm(endpoints-prior, axis=1).max()
        reverse = np.linalg.norm(endpoints-prior[::-1], axis=1).max()
        if min(forward, reverse) < tolerance_m:
            return True
    return False


def collect_task(evaluator, plan_path, audit_output):
    import omnigibson as og
    import torch
    from robot import robot_contract, hold_action, navigation_action, step_navigation
    from route import build_map, plan_route, generate_route, RoutePlanningError
    from goals import APPROACH_MAX_M, resolve_and_sample_task_goal, TaskGoalUnavailable
    from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim

    plan = json.loads(plan_path.read_text())
    if plan.get('goal_contract') != 'behavior_task_object_v1':
        raise ValueError('Plan lacks the task-object goal contract; generic random-goal plans cannot be collected as expert task routes')
    root = Path(plan['route_root'])
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root)
    journal = root/'routes.jsonl'
    records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    completed = {(r['split'], r.get('plan_route_id', Path(r['route_directory']).name)) for r in records}

    def prepare_instance(instance):
        evaluator.env.env.reset(get_obs=False)
        evaluator.load_task_instance(instance)
        body = robot_contract(evaluator, audit_output)
        bounds = np.asarray(body['collision_bounds_base_m'])
        if body['circular_envelope_radius_m'] > DATA.robot_radius_m or bounds[1, 2] > DATA.obstacle_max_z_m:
            raise ValueError('Loaded posture exceeds the dataset R1Pro envelope')
        body['circular_envelope_radius_m'] = DATA.robot_radius_m
        body['collision_bounds_base_m'] = [[-DATA.robot_radius_m]*2+[bounds[0, 2]],
                                           [DATA.robot_radius_m]*2+[DATA.obstacle_max_z_m]]
        grid = build_map(evaluator, body, full_scene=True)
        initial_z = float(evaluator.robot.get_position_orientation()[0][2])
        return body, grid, initial_z

    with (root/'rejections.jsonl').open('a', buffering=1) as rejected:
        for group in plan['instances']:
            split = group['split']
            requests = [r for r in group['routes'] if (split, r['id']) not in completed]
            if not requests:
                continue
            instance_options = group.get('candidate_instances', [group['instance']])
            preferred_index = 0
            loaded_index = None
            for request in requests:
                route_id = request['id']
                band = request['band']
                goal_type = request['goal_type']
                if goal_type not in ('task_target', 'scene_random'):
                    raise ValueError(f'Unknown navigation goal type: {goal_type}')
                # Prefer the last successful instance but revisit earlier
                # candidates if later ones fail. A single exhausted request
                # does not prove an instance has no other safe, distinct route.
                option_order = list(range(preferred_index, len(instance_options)))
                option_order += list(range(preferred_index))
                for option_index in option_order:
                    instance = instance_options[option_index]
                    if loaded_index != option_index:
                        try:
                            body, grid, initial_z = prepare_instance(instance)
                        except RoutePlanningError as error:
                            rejected.write(json.dumps({'route': route_id, 'instance': instance,
                                                       'reason': f'Instance has no safe navigation map: {error}'})+'\n')
                            continue
                        loaded_index = option_index
                    parent = root/split/f"instance_{instance:03d}"
                    parent.mkdir(parents=True, exist_ok=True)
                    rng = np.random.default_rng([plan['seed'], instance,
                                                 int(route_id.rsplit('_', 1)[1]), int(split == 'validation')])
                    partial = parent/f'{route_id}.partial'
                    if partial.exists():
                        shutil.rmtree(partial)
                    unavailable = False
                    succeeded = False
                    for candidate in range(100):
                        directory = parent/f'{route_id}.partial'
                        try:
                            if goal_type == 'task_target':
                                target, endpoints, final_yaw = resolve_and_sample_task_goal(
                                    evaluator, plan['task'], plan['task_index'], grid, rng, band)
                            else:
                                target, final_yaw = None, None
                                endpoints = sample_endpoints(grid, rng, band)
                            if near_duplicate(records, evaluator.env.task.scene_name, instance, endpoints):
                                raise RoutePlanningError('Near-duplicate start and goal in this task instance')
                            evaluator.env.env.reset(get_obs=False)
                            robot = evaluator.robot
                            heading = rng.uniform(-math.pi, math.pi)
                            robot.set_position_orientation(torch.tensor([*endpoints[0], initial_z]),
                                                           torch.tensor([0., 0., math.sin(heading/2), math.cos(heading/2)]))
                            robot.keep_still()
                            hold = hold_action(robot)
                            for _ in range(5):
                                step_navigation(robot, navigation_action(robot, hold, [0., 0., 0.]))
                            if abs(float(robot.get_position_orientation()[0][2]) - initial_z) > .08:
                                raise RoutePlanningError('Sampled start fell away from the supported floor')
                            if detect_robot_collision_in_sim(robot, ignore_obj_in_hand=False):
                                raise RoutePlanningError('Sampled start has non-ground contact')
                            path = plan_route(evaluator, grid, goal_xy=endpoints[1])
                            directory.mkdir()
                            result = generate_route(evaluator, body, directory, grid=grid, path=path,
                                                    visualize=not records, final_yaw=final_yaw)
                            if not result['success']:
                                raise RoutePlanningError(result['failure'])
                        except TaskGoalUnavailable as error:
                            rejected.write(json.dumps({'route': route_id, 'instance': instance,
                                                       'reason': str(error)})+'\n')
                            unavailable = True
                            break
                        except RoutePlanningError as error:
                            rejected.write(json.dumps({'route': route_id, 'instance': instance, 'band': band,
                                                       'candidate': candidate, 'reason': str(error)})+'\n')
                            if directory.exists():
                                shutil.rmtree(directory)
                            continue
                        frames = export_route(directory, grid)
                        destination = parent/route_id
                        directory.rename(destination)
                        record = {'source': 'behavior2026', 'source_family': f"{plan['task']}/{instance}",
                                  'scene_id': f"{evaluator.env.task.scene_name}/{plan['task']}/{instance}",
                                  'split': split, 'route_id': f"{plan['task']}/{destination.relative_to(root)}",
                                  'route_directory': str(destination.relative_to(root)), 'frames': frames,
                                  'task': plan['task'], 'instance': instance, 'plan_route_id': route_id,
                                  'scene_name': evaluator.env.task.scene_name,
                                  'planned_endpoints_xy_m': endpoints.tolist(), 'band': band,
                                  'requested_band': request['band'], 'candidate': candidate,
                                  'sampling': 'task_object_approach_v3' if target else 'scene_random_v1',
                                  'task_approach_max_m': APPROACH_MAX_M if target else None,
                                  'task_route_min_m': .25 if target else None,
                                  'diversity_contract': 'same_instance_endpoints_025m_v2',
                                  'goal_type': goal_type, 'target': target,
                                  'goal_contract': plan['goal_contract'], 'result': result}
                        records.append(record)
                        with (root/'routes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record)+'\n')
                        print(json.dumps({'event': 'dataset_route_complete', 'task': plan['task'],
                                          'routes': len(records), 'split': split, 'frames': frames}), flush=True)
                        succeeded = True
                        break
                    if not succeeded:
                        if not unavailable:
                            rejected.write(json.dumps({'route': route_id, 'instance': instance,
                                                       'reason': 'Candidate budget exhausted; trying next task instance'})+'\n')
                        continue
                    preferred_index = option_index
                    break
                else:
                    raise RuntimeError(f'All safe instance candidates exhausted for {route_id}')
    (root/'complete.json').write_text(json.dumps({'routes': len(records), 'task': plan['task']})+'\n')
