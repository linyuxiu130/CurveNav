"""Native B-spline memory actor: absent-goal semantics and portable state."""
from dataclasses import asdict, replace

import torch

from curvenav import build_policy
from curvenav.config import CurveNavConfig, TrajectoryDecoderConfig, DataConfig
from curvenav.deployment import bspline_bundle as bundle_api
from curvenav.deployment.bspline_bundle import BSplineMemoryBundle, load_bundle, save_bundle
from curvenav.encoders.temporal_geometry import FeatureGeometryDecoder
from test_policy import tiny_config, make_encoded_condition


def test_absent_goal_and_goal_do_not_change_the_memory_features():
    import numpy as np
    cfg=tiny_config();cfg=replace(cfg,data=replace(cfg.data,observation_frames=6))
    policy=build_policy(cfg).eval()
    plan=dict(points=np.empty((0,3),np.float32),keys=np.empty((0,3),np.int64),depth=np.empty(0,np.float32),
        origins=np.empty((0,3),np.float32),times=np.empty(0,np.float64),clock=0.,pose=np.eye(4),
        selection=dict(valid=np.array([False]*5+[True]),poses=np.tile(np.eye(4),(6,1,1)),ages=np.zeros(6,np.float32)))
    class Memory:
        def write(self):pass
        def packet(self):return plan,torch.empty(0,32)
    head=FeatureGeometryDecoder(40,policy.planning_horizon_m,.167584539,width=8).eval()
    actor=BSplineMemoryBundle(None,policy,head,{},{});memory=Memory()
    absent=actor.encode(memory,[1.,2.],False);present=actor.encode(memory,[1.,2.],True)
    assert not absent.terminal_goal.any() and not absent.goal_present.any()
    torch.testing.assert_close(present.terminal_goal,torch.tensor([[1.,2.]]))
    torch.testing.assert_close(absent.tokens,present.tokens,rtol=0,atol=0)
    assert present.tokens.shape[1]==261 and present.token_valid.sum()==256


def test_nogoal_native16_masks_q_goal_and_keeps_two_flow_gradients():
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 1)
    encoded = replace(encoded, goal_present=torch.zeros(1, 1))
    actor = BSplineMemoryBundle(None, policy, None,
        {}, {})
    first = actor.candidates(encoded, [3., -2.])
    second = actor.candidates(replace(encoded, terminal_goal=torch.tensor([[-9., 8.]])), [-9., 8.])
    assert first['source_ids'] == list(range(48, 64))
    assert first['source_groups'] == ['no_goal']*16
    assert first['paths'].shape == (16, 64, 2)
    for name in ('values', 'paths', 'scores'):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    assert first['selected_index'] == second['selected_index']
    (first['values'].square().mean()+first['scores'].square().mean()).backward()
    assert policy.trajectory_decoder.velocity_readout.projection.weight.grad.abs().sum() > 0
    assert policy.trajectory_evaluator.readout[2].weight.grad.abs().sum() > 0


def test_current_bundle_roundtrip_has_one_graph(tmp_path):
    cfg=CurveNavConfig(data=DataConfig(observation_frames=6),trajectory_decoder=TrajectoryDecoderConfig(transformer_layers=1))
    actor=bundle_api.create_bundle(cfg)
    raw=dict(data=asdict(cfg.data),training=asdict(cfg.training),model=dict(trajectory=asdict(cfg.trajectory),
        depth_encoder=asdict(cfg.depth_encoder),condition_encoder=asdict(cfg.condition_encoder),trajectory_decoder=asdict(cfg.trajectory_decoder)))
    path=tmp_path/'actor.pt'
    save_bundle(path,base_config=raw,policy=actor.policy,head=actor.head,architecture=actor.architecture,provenance=actor.provenance)
    restored=load_bundle(path)
    for name in ('policy','head'):
        for key,value in getattr(actor,name).state_dict().items():
            torch.testing.assert_close(value,getattr(restored,name).state_dict()[key],rtol=0,atol=0)
    assert restored.policy.curve_codec.coordinate_dim==14
    assert restored.policy.condition_encoder.motion_encoder.state_tokens==5
    assert restored.head.stem[0].in_channels==397
    assert not hasattr(restored.policy.condition_encoder,'configuration_encoder')
    assert not hasattr(restored,'reader')
    assert restored.provenance['initialization']=='fresh'


def test_evidence_q_does_not_treat_absent_goal_as_origin():
    from curvenav.models.bspline_evidence_evaluator import EvidenceTrajectoryEvaluator
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 2)
    encoded = replace(encoded, goal_present=torch.tensor([[0.], [1.]]),
        configuration_field=encoded.configuration_field[:1].expand(2, -1, -1, -1))
    critic = EvidenceTrajectoryEvaluator(encoded.tokens.shape[-1], 4, policy.planning_horizon_m)
    policy.trajectory_evaluator = critic
    path = torch.stack((torch.linspace(0, 1, 64), torch.zeros(64)), -1)
    candidates = path[None, None].expand(2, 1, -1, -1)
    quality = []
    handle = critic.input_embedding[0].register_forward_pre_hook(lambda _, args: quality.append(args[0].detach()))
    try:
        scores = policy.score_candidate_paths(encoded, torch.zeros(2, 2), candidates)
    finally:
        handle.remove()
    assert torch.isfinite(scores).all()
    assert (quality[0][0,:,-1]==0).all()
    assert (quality[0][1,:,-1]==1).all()
    torch.testing.assert_close(quality[0][0,:,:-1],quality[0][1,:,:-1])




def test_two_euler_steps_query_the_actual_noise_and_intermediate_curves(monkeypatch):
    policy = build_policy(tiny_config()).eval()
    encoded = replace(make_encoded_condition(policy, 1), goal_present=torch.zeros(1, 1))
    actor = BSplineMemoryBundle(None, policy, None, {}, {})
    decoder = policy.trajectory_decoder
    calls = []
    original = decoder._trajectory_geometry
    def query(path, condition):
        calls.append(path.detach().clone())
        return original(path, condition)
    monkeypatch.setattr(decoder, '_trajectory_geometry', query)
    states = []
    original_velocity = policy._predict_velocity
    def velocity(state, time, condition, memory):
        states.append(state.detach().clone())
        return original_velocity(state, time, condition, memory)
    monkeypatch.setattr(policy, '_predict_velocity', velocity)
    actor.native_bank(encoded)
    assert len(calls) == 2
    for state, queried in zip(states, calls, strict=True):
        expected, _ = policy.curve_codec.decode(state)
        torch.testing.assert_close(queried, expected, rtol=0, atol=0)
    assert not torch.equal(calls[0], calls[1])
    assert calls[0].unique(dim=0).shape[0] == 16
