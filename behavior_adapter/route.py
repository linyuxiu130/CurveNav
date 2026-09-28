"""One R1Pro expert route: current collision geometry, native A*, actual rollout."""
import json
import heapq
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label


RESOLUTION = 0.05
CLEARANCE = 0.10


class RoutePlanningError(ValueError):
    pass


def build_map(evaluator, body, full_scene=False):
    import torch
    from omnigibson.utils.constants import GROUND_CATEGORIES
    from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh

    robot, scene = evaluator.robot, evaluator.env.scene
    start = robot.get_position_orientation()[0].cpu().numpy()
    origin = np.floor(start[:2] / RESOLUTION) * RESOLUTION - 4.0
    yy, xx = np.indices((160, 160))
    world = origin + np.stack([xx, yy], axis=-1) * RESOLUTION
    floor = cv2.imread(str(Path(scene.scene_dir) / 'layout/floor_trav_no_obj_0.png'), cv2.IMREAD_GRAYSCALE)
    if floor is None:
        raise FileNotFoundError('Official floor-support map is missing')
    if full_scene:
        span = np.array(floor.shape[::-1]) * scene.trav_map.map_default_resolution
        origin = -span/2
        yy, xx = np.indices(tuple(np.ceil(span[::-1]/RESOLUTION).astype(int)))
        world = origin + np.stack([xx, yy], axis=-1)*RESOLUTION
    cells = np.rint(world / scene.trav_map.map_default_resolution + np.array(floor.shape[::-1]) / 2).astype(int)
    inside = ((cells >= 0) & (cells < np.array(floor.shape[::-1]))).all(-1)
    layout_support = np.zeros(xx.shape, dtype=bool)
    layout_support[inside] = floor[cells[..., 1][inside], cells[..., 0][inside]] == 255
    # The layout bitmap describes intended travel, not collision geometry. Some
    # BEHAVIOR scenes have white cells outside every physical floor mesh.
    physical_support = np.zeros(xx.shape, dtype=np.uint8)
    occupied = np.zeros(xx.shape, dtype=np.uint8)
    zlo, zhi = np.asarray(body['collision_bounds_base_m'])[:, 2] + start[2]
    floor_z = start[2] + float(np.asarray(body['collision_bounds_base_m'])[0, 2])
    mesh_count = support_mesh_count = low_obstacle_mesh_count = 0
    for obj in scene.objects:
        if obj is robot:
            continue
        ground = obj.category in GROUND_CATEGORIES
        for link in obj.links.values():
            for mesh in link.collision_meshes.values():
                if not mesh.prim.GetAttribute('physics:collisionEnabled').Get():
                    continue
                lo, hi = (a.cpu().numpy() for a in mesh.aabb)
                if (not ground and (hi[2] < zlo or lo[2] > zhi)) or np.any(hi[:2] < origin) or np.any(lo[:2] > origin + np.array(occupied.shape[::-1])*RESOLUTION):
                    continue
                geometry = mesh_prim_to_trimesh_mesh(mesh.prim, include_normals=False, include_texcoord=False)
                vertices = mesh.transform_local_points_to_world(torch.as_tensor(np.asarray(geometry.vertices), dtype=torch.float32)).cpu().numpy()
                triangles = vertices[np.asarray(geometry.faces)]
                if ground:
                    # Rasterize actual, nearly horizontal top faces at the
                    # robot's floor level. Mesh AABBs fill nonexistent areas.
                    span = np.ptp(triangles[..., 2], axis=1)
                    at_level = np.abs(np.mean(triangles[..., 2], axis=1) - floor_z) <= .08
                    top = triangles[(span <= .04) & at_level]
                    pixels = np.rint((top[..., :2] - origin) / RESOLUTION).astype(np.int32)
                    for triangle in pixels:
                        cv2.fillConvexPoly(physical_support, triangle, 1)
                    support_mesh_count += int(len(top) > 0)
                    continue
                triangles = triangles[(triangles[..., 2].max(1) >= zlo) & (triangles[..., 2].min(1) <= zhi)]
                if not len(triangles):
                    continue
                # Project each triangle, not a mesh-wide hull that would fill concave rooms.
                pixels = np.rint((triangles[..., :2] - origin) / RESOLUTION).astype(np.int32)
                for triangle in pixels:
                    cv2.fillConvexPoly(occupied, triangle, 1)
                mesh_count += 1
                low_obstacle_mesh_count += int(hi[2] <= floor_z + .12)
    support = layout_support & physical_support.astype(bool)
    if not support.any():
        raise RoutePlanningError('No physical collision floor overlaps the official traversability map')
    occupied = occupied.astype(bool) | ~support
    # Reserve the raster cell diagonal in addition to the measured robot envelope.
    distance = distance_transform_edt(np.pad(~occupied, 1))[1:-1, 1:-1] * RESOLUTION
    clearance = distance - body['circular_envelope_radius_m'] - math.sqrt(2) * RESOLUTION
    free = clearance >= CLEARANCE
    return dict(origin_xy=origin, occupied=occupied, free=free, clearance_m=clearance,
                layout_support=layout_support, physical_support=physical_support.astype(bool),
                expected_base_z_m=np.array(start[2]),
                resolution_m=np.array(RESOLUTION), collision_mesh_count=np.array(mesh_count),
                support_mesh_count=np.array(support_mesh_count),
                low_obstacle_mesh_count=np.array(low_obstacle_mesh_count))


def astar(free, start, goal):
    """Four-neighbor A*: allocate costs only for visited cells, preserving official tie order."""
    start, goal = tuple(start), tuple(goal)
    queue, cost, parent, visited = [(0., start)], {start: 0.}, {}, set()
    while queue:
        _, current = heapq.heappop(queue)
        visited.add(current)
        if current == goal:
            cells = [current]
            while current in parent:
                current = parent[current]
                cells.append(current)
            return np.asarray(cells[::-1])
        x, y = current
        for neighbor in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            nx, ny = neighbor
            if not (0 <= nx < free.shape[0] and 0 <= ny < free.shape[1]) or not free[neighbor] or neighbor in visited:
                continue
            trial = cost[current]+1
            if trial < cost.get(neighbor, math.inf):
                cost[neighbor], parent[neighbor] = trial, current
                distance = math.sqrt((nx-goal[0])**2+(ny-goal[1])**2)
                heapq.heappush(queue, (trial+distance, neighbor))
    raise RoutePlanningError('No connected grid route')


def plan_route(evaluator, grid, goal_xy=None):
    from omnigibson.utils import transform_utils as T

    position, orientation = evaluator.robot.get_position_orientation()
    start = position[:2].cpu().numpy()
    yaw = float(T.quat2euler(orientation)[2])
    origin, free = grid['origin_xy'], grid['free']
    start_cell = np.rint((start - origin) / RESOLUTION).astype(int)[::-1]
    if not free[tuple(start_cell)]:
        raise RoutePlanningError('Initial robot pose is outside the conservative R1Pro free region')
    if goal_xy is None:
        components, _ = label(free)
        candidates = np.argwhere(components == components[tuple(start_cell)])
        xy = origin + candidates[:, ::-1] * RESOLUTION
        delta = xy - start
        distance = np.linalg.norm(delta, axis=1)
        eligible = (distance >= 1.5) & (distance <= 2.5)
        candidates, delta, distance = candidates[eligible], delta[eligible], distance[eligible]
        if not len(candidates):
            raise RoutePlanningError('No reachable 1.5–2.5 m expert goal in the current free region')
        alignment = delta @ np.array([math.cos(yaw), math.sin(yaw)]) / distance
        goal_cell = candidates[np.argmax(distance + 0.2 * alignment)]
    else:
        goal_cell = np.rint((goal_xy-origin)/RESOLUTION).astype(int)[::-1]
    cells = astar(free, tuple(start_cell), tuple(goal_cell))
    path = origin + cells[:, ::-1] * RESOLUTION
    path[0] = start
    # Remove only segments certified free by the same inflated grid.
    simplified, index = [path[0]], 0
    while index < len(path) - 1:
        end = len(path) - 1
        while end > index + 1:
            count = math.ceil(np.linalg.norm(path[end] - path[index]) / (RESOLUTION / 4)) + 1
            segment = np.linspace(path[index], path[end], count)
            pixels = np.rint((segment - origin) / RESOLUTION).astype(int)
            if free[pixels[:, 1], pixels[:, 0]].all():
                break
            end -= 1
        simplified.append(path[end])
        index = end
    return np.asarray(simplified)


def continuous_reference(path, grid, final_yaw=None):
    """Reuse CurveNav's curve clock; certify the R1Pro curve in its own map."""
    from scipy.interpolate import make_interp_spline
    from curvenav.data_generation.geometry import timed_route

    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    # Dense knots keep cubic smoothing local to corners instead of cutting across long turns.
    fit_arc = np.unique(np.r_[arc, np.arange(0., arc[-1], .20), arc[-1]])
    fit_path = np.column_stack([np.interp(fit_arc, arc, path[:, axis]) for axis in range(2)])
    curve = make_interp_spline(fit_arc / arc[-1], fit_path, k=3,
                              bc_type=([(1, segments[0]/lengths[0]*arc[-1])],
                                       [(1, segments[-1]/lengths[-1]*arc[-1])]))
    bound = np.linalg.norm(curve.derivative().c, axis=1).max()
    sampled = curve(np.linspace(0., 1., math.ceil(bound/(RESOLUTION/4))+1))
    pixels = np.rint((sampled-grid['origin_xy'])/RESOLUTION).astype(int)
    inside = ((pixels >= 0) & (pixels < np.array(grid['free'].shape[::-1]))).all(1)
    if not inside.all() or not grid['free'][pixels[:, 1], pixels[:, 0]].all():
        raise RoutePlanningError('Continuous expert curve leaves the R1Pro free region')
    xy, yaw, controls = timed_route(curve, 1/30, .25, .30)
    # timed_route uses Habitat's opposite yaw sign; OmniGibson is world XY.
    controls[:, 1] *= -1
    # Retiming preserves the curve while respecting start/stop and curve-speed limits.
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = controls[:, 0].copy()
    speed[0] = speed[-1] = 0.
    for i in range(1, len(speed)):
        speed[i] = min(speed[i], math.sqrt(speed[i-1]**2 + 2*.35*distance[i-1]))
    for i in range(len(speed)-2, -1, -1):
        speed[i] = min(speed[i], math.sqrt(speed[i+1]**2 + 2*.35*distance[i]))
    clock = np.r_[0., np.cumsum(2*distance/(speed[:-1]+speed[1:]))]
    count = math.ceil(clock[-1]*30)
    ticks = np.linspace(0., clock[-1], count+1)
    rate = clock[-1]/(count/30)
    speed = np.interp(ticks, clock, speed)*rate
    # Smooth the time law, never XY coordinates: remain on the certified curve.
    kernel = np.exp(-.5*(np.arange(-18, 19)/6)**2)
    speed = np.convolve(speed, kernel/kernel.sum(), mode='full')
    travel = np.r_[0., np.cumsum((speed[:-1]+speed[1:])/60)]
    curve_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(sampled, axis=0), axis=1))]
    scale = curve_arc[-1]/travel[-1]
    speed *= scale
    parameter = np.interp(travel*scale, curve_arc, np.linspace(0., 1., len(sampled)))
    first, second = curve.derivative(), curve.derivative(2)
    tangent, bend = first(parameter), second(parameter)
    curvature = (tangent[:, 0]*bend[:, 1]-tangent[:, 1]*bend[:, 0])/np.linalg.norm(tangent, axis=1)**3
    stretch = max(1., speed.max()/.25, np.max(abs(curvature*speed))/.30)
    duration = (len(speed)-1)/30*stretch
    count = math.ceil(duration*30)
    clock = np.linspace(0., duration, len(speed))
    ticks = np.linspace(0., duration, count+1)
    parameter = np.interp(ticks, clock, parameter)
    speed = np.interp(ticks, clock, speed)/stretch * duration/(count/30)
    tangent, bend = first(parameter), second(parameter)
    curvature = (tangent[:, 0]*bend[:, 1]-tangent[:, 1]*bend[:, 0])/np.linalg.norm(tangent, axis=1)**3
    reference_xy = curve(parameter)
    reference_yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    controls = np.column_stack([speed, curvature*speed])
    if final_yaw is not None:
        delta = math.atan2(math.sin(final_yaw-reference_yaw[-1]), math.cos(final_yaw-reference_yaw[-1]))
        turn_steps = max(1, math.ceil(abs(delta)/(.25/30)))
        turn_yaws = reference_yaw[-1] + delta*np.arange(1, turn_steps+1)/turn_steps
        reference_xy = np.vstack([reference_xy, np.repeat(reference_xy[-1:], turn_steps, axis=0)])
        reference_yaw = np.r_[reference_yaw, turn_yaws]
        controls = np.vstack([controls, np.column_stack([np.zeros(turn_steps),
                                                          np.full(turn_steps, delta/turn_steps*30)])])
    return reference_xy, reference_yaw, controls


def tracking_command(xy, yaw, target, heading, feedforward, previous, dt):
    """Curve velocity feedforward plus actual-pose feedback, with slew limits."""
    yaw_error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
    world_velocity = feedforward[:2] + 2.5 * (target-xy)
    world_velocity *= min(1., .30/max(np.linalg.norm(world_velocity), 1e-12))
    local = np.array([[math.cos(yaw), math.sin(yaw)],
                      [-math.sin(yaw), math.cos(yaw)]]) @ world_velocity
    command = np.r_[local, np.clip(feedforward[2]+2*yaw_error, -.5, .5)]
    change = command-previous
    change[:2] *= min(1., .5*dt/max(np.linalg.norm(change[:2]), 1e-12))
    change[2] = np.clip(change[2], -dt, dt)
    return previous+change, yaw_error


def plot_route(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    with np.load(output / 'map.npz') as grid, np.load(output / 'trajectory.npz') as route:
        origin, resolution = grid['origin_xy'], float(grid['resolution_m'])
        actual, planned = route['position_world_m'][:, :2], route['planned_xy_m']
        raster = np.where(grid['occupied'], 0, np.where(grid['free'], 2, 1))
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        extent = [origin[0] - resolution/2, origin[0] + (raster.shape[1]-.5)*resolution,
                  origin[1] - resolution/2, origin[1] + (raster.shape[0]-.5)*resolution]
        ax.imshow(raster, origin='lower', extent=extent, cmap=ListedColormap(['#303b49', '#c7d0dc', '#f5f7fa']), vmin=0, vmax=2)
        ax.plot(*planned.T, '--', color='#ee8e22', lw=2.5, label='Planned route')
        if 'reference_xy_m' in route:
            ax.plot(*route['reference_xy_m'].T, color='#9a54ba', lw=1.5, label='Continuous reference')
        ax.plot(*actual.T, color='#1678c8', lw=2, label='Actual R1Pro path')
        ax.scatter(*actual[0], marker='o', s=85, c='#22aa77', edgecolors='white', zorder=5, label='Start')
        ax.scatter(*planned[-1], marker='*', s=180, c='#df4e61', edgecolors='white', zorder=5, label='Goal')
        for idx in np.linspace(0, len(actual)-1, min(8, len(actual))).astype(int):
            yaw = route['yaw_rad'][idx]
            ax.arrow(*actual[idx], .14*np.cos(yaw), .14*np.sin(yaw), head_width=.06, color='#1678c8', zorder=4)
        bounds = np.vstack([actual, planned])
        lo, hi = bounds.min(0)-1.0, bounds.max(0)+1.0
        ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), aspect='equal', xlabel='World X (m)', ylabel='World Y (m)',
               title='BEHAVIOR / R1Pro: planned and executed expert route\nDark: obstacles or unsupported floor | Gray: robot clearance buffer')
        ax.legend(loc='best')
        ax.grid(alpha=.15)
        fig.savefig(output / 'trajectory.png', dpi=180)
        plt.close(fig)


def generate_route(evaluator, body, output, grid=None, path=None, visualize=True, final_yaw=None):
    import omnigibson as og
    from omnigibson.utils import transform_utils as T
    from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
    from curvenav.data.depth import PinholeIntrinsics, preprocess_depth
    from robot import hold_action, navigation_action, camera_calibration, step_navigation

    started = time.perf_counter()
    if grid is None:
        grid = build_map(evaluator, body)
    np.savez_compressed(output / 'map.npz', **grid)
    print(json.dumps({'event': 'route_map', 'collision_meshes': int(grid['collision_mesh_count']),
                      'low_obstacle_meshes': int(grid['low_obstacle_mesh_count']),
                      'free_cells': int(grid['free'].sum())}), flush=True)
    if path is None:
        path = plan_route(evaluator, grid)
    reference_xy, reference_yaw, reference_controls = continuous_reference(path, grid, final_yaw=final_yaw)
    robot = evaluator.robot
    camera = robot.sensors[evaluator.robot_camera_names['head'].split('::', 1)[1]]
    intrinsics = PinholeIntrinsics.from_matrix(np.array(body['camera_intrinsics']), width=camera.image_width, height=camera.image_height)
    hold = hold_action(robot)
    positions, quaternions, yaws, times, actions, depths, camera_poses, frame_steps = [], [], [], [], [], [], [], []
    failure, settled, reference_index = None, 0, 0
    aligned = False
    previous = np.zeros(3)
    dt = og.sim.get_sim_step_dt()
    max_steps = max(1500, len(reference_xy)+600)
    for step in range(max_steps+1):
        position, quaternion = robot.get_position_orientation()
        xy = position[:2].cpu().numpy()
        roll, pitch, yaw = (float(v) for v in T.quat2euler(quaternion))
        positions.append(position.cpu().numpy().copy())
        quaternions.append(quaternion.cpu().numpy().copy())
        yaws.append(yaw)
        times.append(og.sim.current_time)
        if step % 3 == 0:
            og.sim.render()
            depth = camera.get_obs()[0]['depth_linear'].cpu().numpy()
            encoded, calibrated_k = preprocess_depth(depth, source_intrinsics=intrinsics, maximum_m=5., height=camera.image_height, width=camera.image_width)
            depths.append(encoded.astype(np.float16))
            camera_poses.append(camera_calibration(evaluator)[1].cpu().numpy())
            frame_steps.append(step)
        pixel = np.rint((xy-grid['origin_xy'])/RESOLUTION).astype(int)
        if (not (0 <= pixel[0] < grid['free'].shape[1] and 0 <= pixel[1] < grid['free'].shape[0])
                or not grid['physical_support'][pixel[1], pixel[0]]):
            failure = 'robot base has no collision-enabled floor support'
            break
        if abs(float(position[2]) - float(grid['expected_base_z_m'])) > .08:
            failure = 'robot base left the supported floor height'
            break
        if max(abs(roll), abs(pitch)) > math.radians(10):
            failure = 'robot tipped beyond upright navigation posture'
            break
        if grid['clearance_m'][pixel[1], pixel[0]] < CLEARANCE:
            failure = 'robot entered the required obstacle clearance margin'
            break
        if step and detect_robot_collision_in_sim(robot, ignore_obj_in_hand=False):
            failure = 'non-ground scene contact detected'
            break
        if step == max_steps:
            failure = 'route time budget exceeded'
            break
        if not aligned:
            initial_error = math.atan2(math.sin(reference_yaw[0]-yaw), math.cos(reference_yaw[0]-yaw))
            aligned = abs(initial_error) < math.radians(2)
        target, heading = reference_xy[reference_index], reference_yaw[reference_index]
        feedforward = np.zeros(3)
        if aligned and reference_index < len(reference_xy)-1:
            speed, turn = reference_controls[reference_index]
            feedforward = np.array([speed*math.cos(heading), speed*math.sin(heading), turn])
        velocity, yaw_error = tracking_command(xy, yaw, target, heading, feedforward, previous, dt)
        at_goal = (reference_index == len(reference_xy)-1 and np.linalg.norm(xy-path[-1]) < .005
                   and abs(yaw_error) < math.radians(2))
        if at_goal:
            # Brake within the accepted goal tolerance instead of chasing sub-friction pose errors forever.
            velocity, _ = tracking_command(xy, yaw, xy, yaw, np.zeros(3), previous, dt)
            settled = settled+1 if not velocity.any() else 0
            if settled >= 10:
                break
        else:
            settled = 0
        previous = velocity.copy()
        if aligned:
            reference_index = min(reference_index+1, len(reference_xy)-1)
        action = navigation_action(robot, hold, velocity)
        step_navigation(robot, action, render=(step+1)%3 == 0)
        actions.append(action.cpu().numpy())
    step_navigation(robot, navigation_action(robot, hold, [0., 0., 0.]))
    np.savez_compressed(output / 'trajectory.npz', planned_xy_m=path, reference_xy_m=reference_xy, reference_yaw_rad=reference_yaw,
                        reference_controls=reference_controls, position_world_m=positions,
                        quaternion_xyzw=quaternions, yaw_rad=yaws, time_s=times, actions=actions,
                        depth_frame_steps=frame_steps, camera_intrinsics=calibrated_k,
                        camera_to_body_optical=camera_poses)
    np.save(output / 'depth.npy', np.stack(depths))
    actual = np.asarray(positions)[:, :2]
    pixels = np.rint((actual-grid['origin_xy'])/RESOLUTION).astype(int)
    pixels = np.clip(pixels, 0, np.array(grid['free'].shape[::-1])-1)
    result = {'success': failure is None, 'failure': failure, 'steps': len(actions),
              'duration_s': float(times[-1]-times[0]), 'wall_seconds': time.perf_counter()-started, 'depth_frames': len(depths),
              'path_length_m': float(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum()),
              'goal_heading_error_deg': float(abs(math.degrees(math.atan2(math.sin(yaws[-1]-reference_yaw[-1]), math.cos(yaws[-1]-reference_yaw[-1]))))),
              'settled_frames': settled,
              'goal_error_m': float(np.linalg.norm(actual[-1]-path[-1])),
              'minimum_grid_clearance_m': float(grid['clearance_m'][pixels[:,1],pixels[:,0]].min()),
              'minimum_base_z_m': float(np.min(np.asarray(positions)[:, 2])),
              'maximum_base_z_m': float(np.max(np.asarray(positions)[:, 2])),
              'robot_radius_m': body['circular_envelope_radius_m'], 'planning_margin_m': CLEARANCE,
              'scope': 'One fixed-posture navigation demonstration; floor-map support and conservative projected collision geometry. Not yet a compiled CurveNav training dataset.'}
    (output / 'route.json').write_text(json.dumps(result, indent=2)+'\n')
    if visualize:
        plot_route(output)
    print(json.dumps({'event': 'route_complete', **result}), flush=True)
    return result
