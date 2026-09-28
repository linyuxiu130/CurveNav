"""R1Pro navigation through the live BEHAVIOR robot and its native controllers."""
import numpy as np
import torch

from omnigibson.controllers.controller_view import ControllerView
from omnigibson.utils import transform_utils as T


def hold_action(robot):
    action = torch.zeros(robot.action_dim)
    for name, (group, index) in robot.controllers.items():
        action[robot.controller_action_idx[name]] = ControllerView.compute_no_op_action(group, index)
    return action


def navigation_action(robot, hold, velocity):
    """Body-frame velocity setpoint [vx m/s, vy m/s, yaw rad/s]; preserve upper-body targets."""
    group, _ = robot.controllers['base']
    velocity = torch.as_tensor(velocity, dtype=torch.float32)
    if velocity.shape != (3,) or not torch.isfinite(velocity).all():
        raise ValueError('Base velocity must contain three finite values')
    command = ControllerView.reverse_preprocess_command(group, velocity)
    lo, hi = ControllerView.get_command_input_limits(group)
    if torch.any((command < lo) | (command > hi)):
        raise ValueError('Requested velocity exceeds the BEHAVIOR base controller limits')
    action = hold.clone()
    action[robot.controller_action_idx['base']] = command
    return action


def step_navigation(robot, action, render=True):
    """Keep native physics/rendering; navigation does not use task reward or full RGB-D observations."""
    import omnigibson as og
    robot.apply_action(action)
    with og.sim.render_on_step(render):
        og.sim.step()


def robot_contract(evaluator, output):
    robot = evaluator.robot
    group, _ = robot.controllers['base']
    if (robot.model != 'r1pro'
            or ControllerView.get_controller_type_str(group) != 'HolonomicBaseJointController'
            or ControllerView.get_motor_type(group) != 'velocity'):
        raise ValueError('This adapter requires the official R1Pro holonomic velocity controller')
    base_from_world = T.pose_inv(T.pose2mat(robot.get_position_orientation()))
    points, meshes = [], []
    for link in robot.links.values():
        for mesh in link.collision_meshes.values():
            if mesh.prim.GetAttribute('physics:collisionEnabled').Get():
                world_points = mesh.transform_local_points_to_world(mesh.points)
                points.append(T.transform_points(world_points, base_from_world))
                meshes.append(mesh.prim_path)
    points = torch.cat(points).cpu().numpy()
    np.save(output / 'robot_collision_points_base_m.npy', points)
    camera = robot.sensors[evaluator.robot_camera_names['head'].split('::', 1)[1]]
    intrinsic, camera_to_body = camera_calibration(evaluator)
    hold = hold_action(robot)
    return {
        'model': robot.model, 'base_frame': robot.base_footprint_link_name,
        'motion_model': 'holonomic_body_velocity',
        'velocity_units': ['m/s', 'm/s', 'rad/s'],
        'base_action_indices': robot.controller_action_idx['base'].tolist(),
        'hold_action': hold.tolist(), 'joint_names': list(robot.joints),
        'joint_positions': robot.get_joint_positions().tolist(),
        'collision_meshes': meshes,
        'collision_bounds_base_m': [points.min(0).tolist(), points.max(0).tolist()],
        'circular_envelope_radius_m': float(np.linalg.norm(points[:, :2], axis=1).max()),
        'camera_intrinsics': intrinsic.tolist(),
        'camera_to_body_optical': camera_to_body.tolist(),
        'image_size_hw': [camera.image_height, camera.image_width],
        'scope': 'Collision vertices at the recorded pose; no carried object or other arm postures. Not a navigation safety certificate.',
    }


def camera_calibration(evaluator):
    robot = evaluator.robot
    base_from_world = T.pose_inv(T.pose2mat(robot.get_position_orientation()))
    camera_name = evaluator.robot_camera_names['head'].split('::', 1)[1]
    camera = robot.sensors[camera_name]
    # USD camera: +X right, +Y up, -Z forward; optical: +X right, +Y down, +Z forward.
    world_from_camera = T.pose2mat(camera.get_position_orientation())
    optical_flip = torch.diag(torch.tensor([1., -1., -1., 1.]))
    camera_to_body = base_from_world @ world_from_camera @ optical_flip
    return camera.intrinsic_matrix, camera_to_body


def configure_navigation_camera(evaluator, width, height):
    """Render only the training depth raster while preserving both camera fields of view."""
    import omnigibson as og
    head_name = evaluator.robot_camera_names['head'].split('::', 1)[1]
    camera = evaluator.robot.sensors[head_name]
    expected = camera.intrinsic_matrix.cpu().numpy().copy()
    expected[0] *= width/camera.image_width
    expected[1] *= height/camera.image_height
    if (camera.image_width, camera.image_height) != (width, height):
        camera.image_width = width
        camera.image_height = height
    with og.sim.editing_usd():
        for name, sensor in evaluator.robot.sensors.items():
            if hasattr(sensor, 'render_product'):
                sensor.render_product.hydra_texture.set_updates_enabled(name == head_name)
    for _ in range(4):
        og.sim.render()
    np.testing.assert_allclose(camera.intrinsic_matrix.cpu().numpy(), expected, atol=1e-3, rtol=1e-5)
    og.sim.update_handles()
    return camera
