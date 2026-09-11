"""Regression checks for the shared backbone and per-model inference contracts."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_sand_history_and_required_normalization(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'baselines/sandplanner'))
    from sand_planner.config import InferenceConfig
    from sand_planner.agent.depth_processor import ArrayDepthProcessor
    from sand_planner.utils.normalize import TrajectoryNormalizer

    config = InferenceConfig()
    normalizer = TrajectoryNormalizer(config.stats_path, margin=0.0)
    assert np.isfinite(normalizer.normalized_zero()).all()
    cache = ArrayDepthProcessor(config)
    for value in range(1, 6):
        cache.add_frame_to_cache(np.full((480, 640), value, np.float32))
    sequence = cache.get_sequence_from_cache()
    assert sequence.shape == (1, 4, 1, 168, 224)
    torch.testing.assert_close(sequence[0, :, 0, 0, 0], torch.tensor([2., 3., 4., 5.]) / 8)
    cache.clear_cache()
    with pytest.raises(ValueError, match='缓存为空'):
        cache.get_sequence_from_cache()


def test_shared_attention_matches_explicit_equation():
    from navbench.vision.depth_anything_v2.dinov2_layers.attention import Attention

    torch.manual_seed(42)
    model = Attention(24, num_heads=3, qkv_bias=True).eval()
    x = torch.randn(2, 7, 24)
    q, k, v = model.qkv(x).reshape(2, 7, 3, 3, 8).permute(2, 0, 3, 1, 4)
    expected = torch.softmax((q * 8**-0.5) @ k.transpose(-2, -1), dim=-1) @ v
    expected = model.proj(expected.transpose(1, 2).reshape(2, 7, 24))
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_xnavdp_device_and_strict_weights(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'baselines/x-navdp'))
    network = importlib.import_module('eval.src.policy_network_embodiment')
    agent_module = importlib.import_module('eval.src.policy_agent')

    class Policy(torch.nn.Module):
        def __init__(self, *args, device):
            super().__init__()
            self.device = device
            self.weight = torch.nn.Parameter(torch.ones(2))

    monkeypatch.setattr(network, 'NavDP_Policy_Embodiment', Policy)
    path = tmp_path / 'weights.pt'
    torch.save({'weight': torch.zeros(2), 'critic_head.weight': torch.ones(1)}, path)
    agent = agent_module.NavDP_Agent(np.eye(3), navi_model=path, device='cpu')
    assert agent.navi_former.device == 'cpu'
    assert torch.equal(agent.navi_former.weight, torch.zeros(2))
    torch.save({'critic_head.weight': torch.ones(1)}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        agent_module.NavDP_Agent(np.eye(3), navi_model=path, device='cpu')
