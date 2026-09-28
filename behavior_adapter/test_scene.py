import unittest
from scene import training_instance


class TrainingSplitTest(unittest.TestCase):
    def test_training_ids_exclude_test_split(self):
        self.assertEqual(training_instance('0'), 0)
        self.assertEqual(training_instance(300), 300)
        for value in (-1, 301, 340):
            with self.assertRaises(ValueError):
                training_instance(value)


def check_motion(evaluator):
    """Check native velocity targets and record the actual simulated response."""
    import json
    import omnigibson as og
    import torch
    from omnigibson.controllers.controller_view import ControllerView
    from omnigibson.utils import transform_utils as T
    from robot import hold_action, navigation_action

    robot = evaluator.robot
    hold = hold_action(robot)
    records = []
    dt = og.sim.get_sim_step_dt()
    group, index = robot.controllers['base']
    for axis in range(3):
        velocity = torch.zeros(3)
        velocity[axis] = 0.1
        action = navigation_action(robot, hold, velocity)
        rates = []
        for _ in range(20):
            initial = T.pose2mat(robot.get_position_orientation())
            evaluator.env.step(action)
            relative = T.pose_inv(initial) @ T.pose2mat(robot.get_position_orientation())
            rates.append(torch.stack([relative[0, 3], relative[1, 3],
                                      torch.atan2(relative[1, 0], relative[0, 0])]) / dt)
        measured = torch.stack(rates[-5:]).mean(0)
        record = {'axis': axis, 'requested_velocity': velocity.tolist(),
                  'measured_velocity_each_step': torch.stack(rates).tolist(),
                  'steady_velocity': measured.tolist(),
                  'controller_target': ControllerView.get_goal(group, index)['target'].tolist()}
        records.append(record)
        print(json.dumps(record), flush=True)
        for _ in range(20):
            evaluator.env.step(navigation_action(robot, hold, -velocity))
    evaluator.env.step(navigation_action(robot, hold, [0., 0., 0.]))
    validate_motion(records)
    return records


def validate_motion(records):
    """Velocity commands are setpoints, not a promise of ideal motion under physics."""
    import numpy as np

    for record in records:
        requested = np.asarray(record['requested_velocity'])
        target = np.asarray(record['controller_target'])
        measured = np.asarray(record['steady_velocity'])
        # Native controller rotates XY into its canonical frame, preserving magnitude.
        np.testing.assert_allclose(np.linalg.norm(target[:2]), np.linalg.norm(requested[:2]), atol=1e-6)
        np.testing.assert_allclose(target[2], requested[2], atol=1e-6)
        axis = record['axis']
        assert measured[axis] > 0.5 * requested[axis], record
        np.testing.assert_allclose(np.delete(measured, axis), 0, atol=0.005)


class RouteTrackingTest(unittest.TestCase):
    def test_sparse_astar_preserves_four_neighbor_path(self):
        import numpy as np
        from route import astar, RoutePlanningError
        free = np.ones((7, 8), dtype=bool)
        free[1:6, 3] = False
        path = astar(free, (3, 1), (3, 6))
        np.testing.assert_array_equal(path[[0, -1]], [[3, 1], [3, 6]])
        self.assertEqual(len(path)-1, 11)
        self.assertTrue(free[tuple(path.T)].all())
        self.assertTrue((np.abs(np.diff(path, axis=0)).sum(1) == 1).all())
        free[:, 3] = False
        with self.assertRaises(RoutePlanningError):
            astar(free, (3, 1), (3, 6))

    def test_continuous_tracking_and_slew_limits(self):
        import math
        import numpy as np
        from route import tracking_command

        zero = np.zeros(3)
        command, error = tracking_command(np.zeros(2), math.pi-.01, np.ones(2),
                                           -math.pi+.01, zero, zero, 1/30)
        self.assertAlmostEqual(error, .02)
        self.assertLessEqual(np.linalg.norm(command[:2]), .5/30+1e-12)
        self.assertLessEqual(abs(command[2]), 1/30+1e-12)
        command, _ = tracking_command(np.zeros(2), 0., np.zeros(2), 0.,
                                      np.array([.25, 0., .1]), np.array([.25, 0., .1]), 1/30)
        np.testing.assert_allclose(command, [.25, 0., .1])
        command, _ = tracking_command(np.array([.998, 0.]), 0., np.array([1., 0.]),
                                      0., zero, zero, 1/30)
        np.testing.assert_allclose(command, [.005, 0., 0.], atol=1e-9)

    def test_reference_starts_and_stops_without_speed_jump(self):
        import numpy as np
        from route import continuous_reference

        path = np.array([[1., 1.], [2., 1.]])
        grid = {'origin_xy': np.zeros(2), 'free': np.ones((80, 80), dtype=bool)}
        xy, yaw, controls = continuous_reference(path, grid)
        np.testing.assert_allclose(xy[[0, -1]], path)
        np.testing.assert_allclose(controls[[0, -1]], 0., atol=1e-9)
        self.assertLessEqual(np.max(abs(np.diff(controls[:, 0])))*30, .35+1e-9)
        self.assertLessEqual(controls[:, 0].max(), .25)
        self.assertLess(np.max(abs(np.diff(controls[:, 0], n=2)))*30**2, 1.)
        grid['free'][:] = False
        with self.assertRaises(ValueError):
            continuous_reference(path, grid)


class RouteExportTest(unittest.TestCase):
    def test_source_chart_and_camera_world_pose_are_preserved(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from scipy.spatial.transform import Rotation
        from collect import export_route
        from curvenav.data_generation.audit import validate_route_pose
        from curvenav.data.privileged import SourceConfigurationGrid

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'route'
            directory.mkdir()
            rotation = Rotation.from_euler('xyz', [.02, -.01, .4])
            position = np.array([[1., 2., .005], [1.1, 2., .005]])
            camera = np.repeat(np.eye(4)[None], 2, axis=0)
            camera[:, :3, 3] = [.2, 0., 1.4]
            np.savez(directory/'trajectory.npz', depth_frame_steps=[0, 1],
                     position_world_m=position, yaw_rad=[.4, .4],
                     quaternion_xyzw=np.repeat(rotation.as_quat()[None], 2, axis=0),
                     camera_to_body_optical=camera, camera_intrinsics=np.eye(3), time_s=[0., .1])
            clearance = np.arange(25).reshape(5, 5)*.01+.2
            grid = {'origin_xy': np.array([.9, 1.9]), 'resolution_m': .05,
                    'free': np.ones((5, 5), dtype=bool), 'clearance_m': clearance}
            export_route(directory, grid)
            xy = np.load(directory/'traj_xy.npy')
            yaw = np.load(directory/'traj_yaw.npy')
            pose = np.load(directory/'body_to_world.npy')
            validate_route_pose(xy, yaw, pose)
            original = np.repeat(np.eye(4)[None], 2, axis=0)
            original[:, :3, :3] = rotation.as_matrix()
            original[:, :3, 3] = position
            np.testing.assert_allclose(pose @ np.load(directory/'camera_to_body.npy'), original @ camera, atol=1e-6)
            query = SourceConfigurationGrid.load(directory.parent/'navigation_grid.npz')
            np.testing.assert_allclose(query.query_world(np.array([[1., -2.]])), [clearance[2, 2]], atol=1e-6)


if __name__ == '__main__':
    unittest.main()


def benchmark_step(evaluator):
    """Measure native rendering modes against identical physics and 10 Hz depth."""
    import time
    import cProfile
    import pstats
    import io
    import numpy as np
    import omnigibson as og
    from robot import hold_action, navigation_action, step_navigation
    from omnigibson.utils import transform_utils as T
    head_name = evaluator.robot_camera_names['head'].split('::', 1)[1]
    camera = evaluator.robot.sensors[head_name]
    records, samples = {}, {}
    evaluator.robot._disable_grasp_handling = False
    camera.add_modality('rgb')
    camera.image_width = camera.image_height = 720
    with og.sim.editing_usd():
        for sensor in evaluator.robot.sensors.values():
            if hasattr(sensor, 'render_product'):
                sensor.render_product.hydra_texture.set_updates_enabled(True)
    og.sim.update_handles()
    for name in ('all_cameras', 'native_depth', 'native_depth_10hz', 'native_nav_10hz'):
        if name == 'native_depth':
            from robot import configure_navigation_camera
            camera = configure_navigation_camera(evaluator, 224, 224)
            camera.remove_modality('rgb')
            og.sim.update_handles()
        if name == 'native_nav_10hz':
            evaluator.robot._disable_grasp_handling = True
        reset_start = time.perf_counter()
        evaluator.env.env.reset(get_obs=False)
        records[name] = {'reset_seconds': time.perf_counter()-reset_start}
        robot = evaluator.robot
        hold = hold_action(robot)
        for _ in range(10):
            step_navigation(robot, navigation_action(robot, hold, [0., 0., 0.]))
        poses, depth = [], []
        sim_start = og.sim.current_time
        started = time.perf_counter()
        for i in range(90):
            action = navigation_action(robot, hold, [.05, 0., .03 if i >= 30 else 0.])
            step_navigation(robot, action, render=not name.endswith('10hz') or (i+1)%3 == 0)
            if (i+1)%3 == 0:
                if name.endswith('10hz'):
                    og.sim.render()
                poses.append(T.pose2mat(robot.get_position_orientation()).cpu().numpy().copy())
                depth.append(camera.get_obs()[0]['depth_linear'].cpu().numpy().copy())
        records[name].update(wall_seconds=time.perf_counter()-started, simulation_seconds=og.sim.current_time-sim_start)
        from curvenav.data.depth import PinholeIntrinsics, preprocess_depth
        intrinsics = PinholeIntrinsics.from_matrix(camera.intrinsic_matrix.cpu().numpy(), width=camera.image_width, height=camera.image_height)
        depth = [preprocess_depth(d, source_intrinsics=intrinsics, maximum_m=5., height=224, width=224)[0]*5 for d in depth]
        samples[name] = (np.asarray(poses), np.asarray(depth))
        if name != 'all_cameras':
            records[name]['pose_max_difference'] = float(np.max(abs(samples[name][0]-samples['all_cameras'][0])))
            np.testing.assert_allclose(samples[name][0], samples['all_cameras'][0], atol=.002, rtol=0)
            delta = abs(np.nan_to_num(samples[name][1], nan=5., posinf=5.).clip(0,5)-np.nan_to_num(samples['native_depth' if name.endswith('10hz') else 'all_cameras'][1], nan=5., posinf=5.).clip(0,5))
            records[name].update(depth_mae_m=float(delta.mean()), depth_p99_m=float(np.quantile(delta,.99)), depth_fraction_over_2cm=float((delta>.02).mean()))
        print({'benchmark_mode':name, **records[name]}, flush=True)
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(15):
        step_navigation(robot, navigation_action(robot, hold, [0., 0., 0.]))
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(25)
    records['profile'] = stream.getvalue()
    records['physics_dt'] = og.sim.get_physics_dt()
    records['render_dt'] = og.sim.get_rendering_dt()
    evaluator.env.env.reset(get_obs=False)
    return records
