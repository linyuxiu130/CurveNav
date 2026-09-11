import numpy as np
import pytest
import torch

from curvenav.data.depth_bank import PackedDepthBankSpec, PackedDepthRun
from curvenav.training.prefetch import CudaPrefetchLoader


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_prefetch_preserves_batch_values_and_order(tmp_path) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    packed = np.linspace(0, 1, 24 * 8 * 8, dtype=np.float16).reshape(24, 8, 8)
    packed_path = tmp_path / "depth.npy"
    np.save(packed_path, packed)
    depth_bank = PackedDepthBankSpec(
        runs=(PackedDepthRun(path=packed_path, offset=0, frames=24),),
        total_frames=24,
        height=8,
        width=8,
    )
    cpu_batches = [
        {
            "depth_indices": (
                torch.arange(index * 8, (index + 1) * 8).view(2, 4).pin_memory()
            ),
            "point_goal": torch.tensor([[index, -index]], dtype=torch.float32)
            .repeat(2, 1)
            .pin_memory(),
        }
        for index in range(3)
    ]

    actual = list(CudaPrefetchLoader(cpu_batches, depth_bank, device))
    torch.cuda.synchronize(device)

    assert len(actual) == len(cpu_batches)
    for gpu_batch, cpu_batch in zip(actual, cpu_batches, strict=True):
        assert set(gpu_batch) == {"depth", "point_goal"}
        assert gpu_batch["point_goal"].device == device
        assert torch.equal(gpu_batch["point_goal"].cpu(), cpu_batch["point_goal"])
        indices = cpu_batch["depth_indices"].numpy()
        expected_depth = torch.from_numpy(packed[indices]).unsqueeze(2)
        assert gpu_batch["depth"].device == device
        assert torch.equal(gpu_batch["depth"].cpu(), expected_depth)
