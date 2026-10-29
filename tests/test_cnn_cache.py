from types import SimpleNamespace
import numpy as np
import pytest
import torch
from curvenav import build_policy
from test_policy import tiny_config
from curvenav.data.cnn_cache import write_cache,load_cache,image_modules
from curvenav.training.bspline_image_batch import step_feature_maps


def test_frozen_cache_matches_live_maps_and_rejects_changed_cnn(tmp_path):
    actor=SimpleNamespace(policy=build_policy(tiny_config()).eval())
    raw=dict(depths=np.random.default_rng(1).random((2,64,64),dtype=np.float32))
    records=[(dict(raw_sha256='raw',memory_plan=dict(frame_ids=np.array([0,1,0]))),raw,None,None)]
    with torch.no_grad():expected=step_feature_maps(actor,records)
    write_cache(actor,records,tmp_path/'cache')
    cache=load_cache(actor,records,tmp_path/'cache/manifest.json')
    actual=step_feature_maps(actor,records,cache)
    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert all(not p.requires_grad for module in image_modules(actor) for p in module.parameters())
    assert all(p.requires_grad for p in actor.policy.depth_encoder.geometry_projection.parameters())
    assert all(p.requires_grad for p in actor.policy.trajectory_decoder.parameters())
    with torch.no_grad():next(image_modules(actor)[0].parameters()).add_(.001)
    with pytest.raises(ValueError,match='CNN cache does not match'):load_cache(actor,records,tmp_path/'cache/manifest.json')
