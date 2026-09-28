import math
import numpy as np
import pytest
import torch


def test_cuda_memory_replays_depth_clearing_and_motion_exactly():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cuda')
    from curvenav.data.obstacle_memory import ObstacleMemory
    from curvenav.data.obstacle_memory_cuda import CudaObstacleMemory, _BASE, _SHIFT
    geometry = (.42, .02, 1.5)
    cpu = ObstacleMemory(3.6, 5., geometry)
    gpu = CudaObstacleMemory(3.6, 5., geometry)
    k = np.array([[30.,0.,20.],[0.,30.,16.],[0.,0.,1.]])
    camera = np.eye(4)
    rng = np.random.default_rng(42)
    for i in range(16):
        body = np.eye(4)
        angle = i*.07
        body[:2,:2] = [[math.cos(angle),-math.sin(angle)], [math.sin(angle),math.cos(angle)]]
        body[:3,3] = [-2.+i*.12, 1.-i*.08, 0.]
        depth = rng.uniform(.05,.4,(32,40)).astype(np.float16)
        depth[::4] = 0
        depth[1::4] = 1
        if i in (0,7):
            depth[:] = 0
        actual = gpu.update(depth,k,camera,body)
        expected = cpu.update(depth,k,camera,body)
        np.testing.assert_array_equal(actual,expected)
        keys = gpu.keys.cpu().numpy()
        voxels = np.stack((keys//(_BASE*_BASE)-_SHIFT,
                           (keys//_BASE)%_BASE-_SHIFT, keys%_BASE-_SHIFT),axis=1)
        np.testing.assert_array_equal(voxels,cpu.voxels)
