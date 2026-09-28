import numpy as np
import torch

from curvenav.data.depth_bank import PackedDepthBank, PackedDepthBankSpec, PackedDepthRun


def test_parallel_depth_cache_preserves_frame_offsets_and_reopens(tmp_path, monkeypatch):
    monkeypatch.setenv("CURVENAV_DEPTH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CURVENAV_DEPTH_CACHE_WORKERS", "4")
    expected = np.arange(24 * 3 * 4, dtype=np.float16).reshape(24, 3, 4)
    runs = []
    for index in range(8):
        path = tmp_path / f"{index}.npy"
        np.save(path, expected[index * 3:(index + 1) * 3])
        runs.append(PackedDepthRun(path, index * 3, 3))
    spec = PackedDepthBankSpec(tuple(reversed(runs)), 24, 3, 4)
    indices = torch.tensor([[23, 0, 7], [4, 18, 12]])
    for _ in range(2):
        bank = PackedDepthBank(spec)
        np.testing.assert_array_equal(bank.values.numpy(), expected)
        np.testing.assert_array_equal(bank.gather(indices).numpy()[:, :, 0], expected[indices.numpy()])
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 1
    assert not list((tmp_path / "cache").glob("*.tmp"))
