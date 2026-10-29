"""Batch independent geometry encodings without changing inputs or gradients."""
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from curvenav.conditioning.transformer import PolicyConditionEncoder
from curvenav.encoders.temporal_geometry import FeatureGeometryDecoder
from curvenav.deployment.streaming_memory import encode_memory,encode_memories


def test_batch_memory_fields_tokens_and_all_gradients_match_scalar():
    torch.manual_seed(35)
    policy=nn.Module();policy.depth_encoder=SimpleNamespace(max_depth_m=10.)
    policy.condition_encoder=PolicyConditionEncoder(observation_frames=6,planning_horizon_m=3.6,model_dim=32)
    policy.condition_encoder.motion_encoder.neural_dtype=torch.float32
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8);head.neural_dtype=torch.float32
    other_policy=deepcopy(policy);other_head=deepcopy(head)
    plans=[]
    for sample in range(2):
        pose=np.eye(4);pose[:2,3]=[sample*.23,-.16]
        poses=np.tile(pose,(6,1,1));poses[:,0,3]-=np.linspace(.6,0,6)
        rng=np.random.default_rng(35+sample);n=9
        plans.append(dict(points=rng.uniform(-2.,2.,(n,3)).astype(np.float32),depth=np.full(n,2.,np.float32),
            keys=np.column_stack((np.arange(n),np.zeros(n),np.arange(n)%2)).astype(np.int64),origins=np.zeros((n,3),np.float32),pose=pose,
            selection=dict(valid=np.ones(6,bool),poses=poses,ages=np.linspace(6,0,6).astype(np.float32))))
    visual=[torch.randn(9,32,requires_grad=True) for _ in plans]
    reference_visual=[v.detach().clone().requires_grad_() for v in visual]
    goals=[torch.tensor([2.,1.]),torch.tensor([9.,-4.])];present=[True,False]
    scalar=[encode_memory(policy,head,p,v,g,q) for p,v,g,q in zip(plans,visual,goals,present,strict=True)]
    actual,(field,logits)=encode_memories(other_policy,other_head,deepcopy(plans),reference_visual,goals,present)
    for name in [f.name for f in fields(actual)]:
        expected=torch.cat([getattr(s[0],name) for s in scalar])
        torch.testing.assert_close(getattr(actual,name),expected,rtol=3e-5,atol=2e-5)
    expected_field=torch.cat([s[1][0] for s in scalar]);expected_logits=torch.cat([s[1][1] for s in scalar])
    torch.testing.assert_close(field,expected_field,rtol=3e-5,atol=2e-5)
    torch.testing.assert_close(logits,expected_logits,rtol=3e-5,atol=2e-5)
    reference_loss=sum(s[0].tokens.square().mean()+s[1][0].square().mean()+s[1][1].square().mean() for s in scalar)/2
    actual_loss=actual.tokens.square().mean()+field.square().mean()+logits.square().mean()
    a=torch.autograd.grad(reference_loss,list(policy.parameters())+list(head.parameters())+visual,allow_unused=True)
    b=torch.autograd.grad(actual_loss,list(other_policy.parameters())+list(other_head.parameters())+reference_visual,allow_unused=True)
    for expected,actual in zip(a,b,strict=True):
        assert (expected is None)==(actual is None)
        if expected is not None:torch.testing.assert_close(actual,expected,rtol=5e-4,atol=3e-5)
    assert not actual_loss.isnan()
