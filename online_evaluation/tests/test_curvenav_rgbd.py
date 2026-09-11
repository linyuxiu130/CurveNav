"""RGB-D wire and actual-camera calibration checks without loading Isaac Sim."""

import json
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from baselines.curvenav.curvenav_server import create_app
from navbench.scene_evaluator import add_robot_state, parse_observations


def test_rgbd_server_preserves_sensor_snapshot_over_raw_transport():
    from io import BytesIO
    from curvenav.config import CurveNavConfig
    from curvenav.data.observation import DepthContextBuffer

    class Runtime:
        batch_size = 1
        config = CurveNavConfig()

        def step(self, goal, context):
            self.context = context
            return SimpleNamespace(path=np.zeros((1, 63, 3), np.float32))

    runtime = Runtime()
    buffer = DepthContextBuffer(runtime.config.data)
    buffer.reset(1)
    K = np.array([[[326.4, 0, 320], [0, 326.4, 180], [0, 0, 1]]], np.float32)
    eye = np.eye(4, dtype=np.float32)[None]
    context = buffer.update(
        np.full((1,360,640,1),2,np.float32), eye, K, eye, np.array([0.]))
    data = {"state_data": json.dumps({"planning_goal": [[1.,2.]]})}
    for name, value in context.items():
        key = 'depth_context_' + name
        data[key] = (BytesIO(value.tobytes()), key + '.raw')
        data[key + '_shape'] = json.dumps(value.shape)
        data[key + '_dtype'] = value.dtype.str
    response = create_app(runtime).test_client().post('/pointgoal_step', data=data,
        content_type='multipart/form-data')
    assert response.status_code == 200
    for name, value in context.items():
        np.testing.assert_array_equal(runtime.context[name], value)


def test_sensor_optical_pose_is_composed_with_gravity_aligned_planning_frame():
    class Scene(dict):
        pass

    def matrix_from_quat(quaternion):
        return torch.tensor(
            Rotation.from_quat(quaternion[:, [1, 2, 3, 0]].numpy()).as_matrix(),
            dtype=torch.float32,
        )

    def quaternion(rotation):
        return torch.tensor(rotation.as_quat()[[3, 0, 1, 2]][None], dtype=torch.float32)

    root = Rotation.from_euler("xyz", [0.1, 0.2, 0.6])
    camera = Rotation.from_euler("xyz", [1.2, -0.3, 0.8])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=quaternion(root), root_pos_w=torch.tensor([[1.0, 2.0, 0.1]])
        )
    )
    sensor = SimpleNamespace(
        quat_w_ros=quaternion(camera),
        pos_w=torch.tensor([[1.2, 2.1, 0.7]]),
        intrinsic_matrices=torch.eye(3)[None],
    )
    scene = Scene(robot=robot)
    scene.sensors = {
        "camera_sensor": SimpleNamespace(
            data=sensor, _timestamp_last_update=torch.tensor([2.5])
        )
    }
    goal = torch.tensor([[4.0, 3.0, 0.1]])
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, _goal_pos_w=goal))
    obs = add_robot_state({}, env, SimpleNamespace(matrix_from_quat=matrix_from_quat))
    combined = obs["body_to_world"] @ obs["camera_to_body"]
    torch.testing.assert_close(combined[:, :3, :3], matrix_from_quat(sensor.quat_w_ros))
    torch.testing.assert_close(combined[:, :3, 3], sensor.pos_w)
    torch.testing.assert_close(
        obs["body_to_world"][:, :3, 2], torch.tensor([[0.0, 0.0, 1.0]])
    )
    assert obs["timestamps"].dtype == torch.float64
    parsed = parse_observations(obs, {})
    assert set(parsed) == {
        "body_to_world",
        "camera_to_body",
        "camera_intrinsics",
        "timestamps",
        "planning_goal",
    }
