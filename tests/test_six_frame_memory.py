"""Causal timestamp sampling, single writes, source replay, and persistent geometry."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from torch import nn
from curvenav.data.six_frame_memory import SixFrameMemory,sample_frames
from curvenav.deployment.streaming_memory import fuse_sources,fuse_sources_batch
from curvenav.encoders.depth import DepthObservationEncoder


def policy():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.arange(1.,5.))
            self.geometry_projection=nn.Linear(6,4);self.time_projection=nn.Linear(1,4)
            self.output_norm=nn.LayerNorm(4);self.max_depth_m=5.;self.neural_dtype=torch.float32
            self.spatial_projection=SimpleNamespace(out_channels=4);self.calls=0
            self.metric_projector=SimpleNamespace(planning_horizon_m=3.6,max_depth_m=5.)
        def image_features(self,depth):
            self.calls+=1
            return depth.mean((-1,-2),keepdim=True)*self.weight[None,:,None,None]
        sample_pixel_features=staticmethod(DepthObservationEncoder.sample_pixel_features)
    result=nn.Module();result.depth_encoder=Encoder();return result


def memory(monkeypatch,seed=4):
    from curvenav.data import six_frame_memory as module
    p=policy();m=SixFrameMemory(p,seed=seed)
    monkeypatch.setattr(module,'volumes_observed_clear',lambda boxes,*args:np.zeros(len(boxes),bool))
    def project(frame):
        points=np.array([[.8,0.,.4],[.81,0.,.4]])@frame.pose[:3,:3].T+frame.pose[:3,3]
        return dict(world=points,pixels=np.array([0,1]),depth=np.ones(2),kind=np.ones(2,dtype=bool))
    monkeypatch.setattr(m,'project',project)
    return m


def observe(m,time,pose=None,depth=.2):
    pose=np.eye(4) if pose is None else pose
    m.update(np.full((4,4),depth,np.float32),np.array([[2.,0.,2.],[0.,2.,2.],[0.,0.,1.]]),np.eye(4),pose,time)


@pytest.mark.parametrize('dt',[.1,.5])
def test_timestamp_strata_are_causal_distinct_and_cover_six_seconds(dt):
    times=np.arange(0,12+dt/2,dt);rng=np.random.default_rng(9);draws=[]
    for _ in range(30):
        ids,valid=sample_frames(times,rng);ages=times[-1]-times[ids]
        assert valid.all() and ids[-1]==len(times)-1 and len(set(ids))==6
        assert np.all(np.diff(times[ids])>0) and ages.max()<=6
        assert np.all((ages[:-1]>np.array([4.8,3.6,2.4,1.2,0.]))&(ages[:-1]<=np.array([6,4.8,3.6,2.4,1.2])+1e-12))
        draws.append(tuple(ids))
    assert len(set(draws))>1


def test_startup_padding_is_invalid_not_duplicated_evidence():
    ids,valid=sample_frames([2.],np.random.default_rng(1))
    assert ids.tolist()==[0]*6 and valid.tolist()==[False]*5+[True]
    for times in ([],[0.,0.],[1.,0.],[0.,float('nan')]):
        with pytest.raises(ValueError):sample_frames(times,np.random.default_rng(1))


def test_capture_only_then_one_write_and_repeat_read_is_pure(monkeypatch):
    m=memory(monkeypatch)
    for i in range(13):observe(m,i*.5)
    assert m.policy.depth_encoder.calls==0 and not m.cells
    m.write();plan,visual=m.packet();calls=m.policy.depth_encoder.calls
    assert len(m.writes)==1 and calls==6 and len(plan['selection']['indices'])==6
    assert all(v.untyped_storage().nbytes()==v.numel()*v.element_size() for v in m.visual.values())
    state=copy.deepcopy(m.rng.bit_generator.state)
    m.write();again_plan,again_visual=m.packet()
    assert len(m.writes)==1 and m.policy.depth_encoder.calls==calls and m.rng.bit_generator.state==state
    torch.testing.assert_close(visual,again_visual,rtol=0,atol=0)
    assert np.array_equal(plan['keys'],again_plan['keys'])


def test_replay_fusion_matches_online_and_all_source_parameters_receive_gradients(monkeypatch):
    m=memory(monkeypatch)
    for i in range(13):observe(m,i*.5,depth=.1+i*.01)
    m.write();_,online=m.packet();plan=m.read_plan();encoder=m.policy.depth_encoder
    maps={int(i):encoder.image_features(torch.from_numpy(m.observations[int(i)].depth)[None,None]) for i in np.unique(plan['frame_ids'])}
    replay=fuse_sources(m.policy,plan,maps,{i:4 for i in maps})
    torch.testing.assert_close(online,replay,rtol=1e-6,atol=1e-6)
    (replay*torch.arange(1.,5.)).sum().backward()
    assert encoder.weight.grad.abs().sum()>0
    assert encoder.geometry_projection.weight.grad.abs().sum()>0
    assert encoder.time_projection.weight.grad.abs().sum()>0


def test_old_obstacle_survives_raw_buffer_expiry_and_reads_in_current_pose(monkeypatch):
    m=memory(monkeypatch);observe(m,0.);m.write()
    keys=list(m.cells);world=np.concatenate(m.cells[keys[0]]['sources'][0]['world'][None]).mean(0)
    monkeypatch.setattr(m,'project',lambda frame:dict(world=np.empty((0,3)),pixels=np.empty(0,dtype=int),depth=np.empty(0),kind=np.empty(0,dtype=bool)))
    pose=np.eye(4);pose[0,3]=1.
    observe(m,20.,pose);m.write();plan,visual=m.packet()
    assert list(m.cells)==keys and list(m.observations)==[1]
    np.testing.assert_allclose(plan['points'][0],world-pose[:3,3],atol=1e-6)
    assert len(visual)==len(keys)


def test_replay_static_cache_keeps_fresh_features_and_gradients_after_update(monkeypatch):
    m=memory(monkeypatch)
    for i in range(13):observe(m,i*.5,depth=.1+i*.01)
    m.write();plan=m.read_plan();encoder=m.policy.depth_encoder
    def forward(plan):
        maps={int(i):encoder.image_features(torch.from_numpy(m.observations[int(i)].depth)[None,None]) for i in np.unique(plan['frame_ids'])}
        value=fuse_sources(m.policy,plan,maps,{i:4 for i in maps})
        grads=torch.autograd.grad((value*torch.arange(1.,5.)).sum(),tuple(encoder.parameters()))
        return value.detach(),grads
    before,_=forward(plan);cached=plan['_fusion_inputs']
    with torch.no_grad():encoder.weight.add_(torch.tensor([.2,-.1,.3,-.2]))
    warm,warm_grads=forward(plan)
    cold,cold_grads=forward({k:v for k,v in plan.items() if not k.startswith('_')})
    assert plan['_fusion_inputs'] is cached and not torch.equal(before,warm)
    torch.testing.assert_close(warm,cold,rtol=0,atol=0)
    for a,b in zip(warm_grads,cold_grads,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_cross_state_fusion_matches_independent_writes_and_parameter_gradients(monkeypatch):
    m=memory(monkeypatch);plans=[]
    for step in range(13):
        observe(m,step*.5,depth=.1+step*.01)
        if step in (4,8,12):m.write();plans.append(m.read_plan())
    other=copy.deepcopy(m.policy)
    def features(p):
        return [{int(i):p.depth_encoder.image_features(torch.from_numpy(m.observations[int(i)].depth)[None,None])
            for i in np.unique(plan['frame_ids'])} for plan in plans]
    maps=features(m.policy);other_maps=features(other);widths=[{i:4 for i in values} for values in maps]
    from curvenav.deployment.streaming_memory import source_features
    expected=[];encoder=m.policy.depth_encoder
    for p,v,w in zip(plans,maps,widths,strict=True):
        geometry=torch.from_numpy(np.column_stack((p['offsets']/encoder.max_depth_m,p['depths']/encoder.max_depth_m,
            p['depths']<encoder.max_depth_m,p['kinds']))).float()
        tokens=encoder.output_norm(source_features(encoder,p,v,w)+encoder.geometry_projection(geometry)+encoder.time_projection(torch.from_numpy(p['ages'])[:,None]))
        expected.append(torch.segment_reduce(tokens,reduce='mean',lengths=torch.from_numpy(p['lengths'])))
    actual=fuse_sources_batch(other,copy.deepcopy(plans),other_maps,widths)
    for a,b in zip(actual,expected,strict=True):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    probes=[torch.randn_like(x) for x in expected]
    a=torch.autograd.grad(sum((x*p).sum() for x,p in zip(actual,probes,strict=True)),tuple(other.parameters()))
    b=torch.autograd.grad(sum((x*p).sum() for x,p in zip(expected,probes,strict=True)),tuple(m.policy.parameters()))
    for x,y in zip(a,b,strict=True):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)


def test_shared_cells_compute_once_with_fresh_weights_and_distinct_ages(monkeypatch):
    from curvenav.deployment.streaming_memory import source_features
    m=memory(monkeypatch)
    for step in range(13):observe(m,step*.5,depth=.1+step*.01)
    m.write();plan=m.read_plan();changed=copy.deepcopy(plan);changed['ages']=changed['ages']+.5
    plans=[plan,copy.deepcopy(plan),changed];other=copy.deepcopy(m.policy)
    ids=np.unique(plan['frame_ids'])
    maps={int(i):torch.randn(1,4,2,2,requires_grad=True) for i in ids}
    reference={i:value.detach().clone().requires_grad_() for i,value in maps.items()}
    widths={i:4 for i in maps};encoder=other.depth_encoder
    expected=[]
    for p in plans:
        geometry=torch.from_numpy(np.column_stack((p['offsets']/encoder.max_depth_m,p['depths']/encoder.max_depth_m,
            p['depths']<encoder.max_depth_m,p['kinds']))).float()
        tokens=encoder.output_norm(source_features(encoder,p,reference,widths)+encoder.geometry_projection(geometry)+encoder.time_projection(torch.from_numpy(p['ages'])[:,None]))
        expected.append(torch.segment_reduce(tokens,reduce='mean',lengths=torch.from_numpy(p['lengths'])))
    processed=[]
    hook=m.policy.depth_encoder.geometry_projection.register_forward_pre_hook(lambda module,args:processed.append(len(args[0])))
    actual=fuse_sources_batch(m.policy,plans,[maps]*3,[widths]*3);hook.remove()
    assert processed==[2*len(plan['pixels'])]
    for a,b in zip(actual,expected,strict=True):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    probes=[torch.randn_like(x) for x in actual]
    a=torch.autograd.grad(sum((x*p).sum() for x,p in zip(actual,probes)),list(m.policy.parameters())+list(maps.values()),allow_unused=True)
    b=torch.autograd.grad(sum((x*p).sum() for x,p in zip(expected,probes)),list(other.parameters())+list(reference.values()),allow_unused=True)
    for x,y in zip(a,b,strict=True):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
    with torch.no_grad():m.policy.depth_encoder.geometry_projection.bias.add_(.2)
    fresh=fuse_sources_batch(m.policy,plans,[maps]*3,[widths]*3)
    assert not torch.equal(fresh[0],actual[0])


def test_replay_caches_depth_inputs_but_recomputes_cnn_each_step():
    from curvenav.training.bspline_image_batch import step_feature_maps
    p=policy();actor=SimpleNamespace(policy=p);raw=dict(depths=np.ones((2,4,4),np.float16))
    item=dict(raw_sha256='raw',memory_plan=dict(frame_ids=np.array([1,0,1])))
    records=[(item,raw,None,None)]*2
    first=step_feature_maps(actor,records);assert p.depth_encoder.calls==1
    inputs=raw['_device_depths'][torch.device('cpu')];pointer=inputs[0].data_ptr()
    with torch.no_grad():p.depth_encoder.weight.add_(1)
    second=step_feature_maps(actor,records);assert p.depth_encoder.calls==2
    assert inputs[0].data_ptr()==pointer
    torch.testing.assert_close(second['raw',0],first['raw',0]+1)
    second['raw',0].sum().backward();assert p.depth_encoder.weight.grad is not None


def test_visible_refutation_cannot_be_undone_by_randomly_resampling_old_frame(monkeypatch):
    from curvenav.data import six_frame_memory as module
    m=memory(monkeypatch);observe(m,0.,depth=.2);m.write();assert m.cells
    original=m.project
    monkeypatch.setattr(m,'project',lambda frame:original(frame) if frame.index==0 else dict(world=np.empty((0,3)),pixels=np.empty(0,dtype=int),depth=np.empty(0),kind=np.empty(0,dtype=bool)))
    monkeypatch.setattr(module,'volumes_observed_clear',lambda boxes,depth,*args:np.full(len(boxes),depth[0,0]>.9))
    observe(m,1.,depth=1.);m.write();assert not m.cells
    observe(m,2.,depth=.3);m.plan_write([-1,-1,-1,-1,0,2])
    assert not m.cells


def test_state_restore_preserves_sampling_rng_and_rejects_future_sources(monkeypatch):
    m=memory(monkeypatch)
    for i in range(13):observe(m,i*.5)
    m.write();saved=m.state_dict();other=memory(monkeypatch);other.policy.load_state_dict(m.policy.state_dict());other.load_state_dict(saved)
    observe(m,6.5);observe(other,6.5);m.write();other.write()
    assert m.selection['indices']==other.selection['indices']
    np.testing.assert_array_equal(m.read_plan()['frame_ids'],other.read_plan()['frame_ids'])
    with pytest.raises(ValueError,match='increase'):observe(m,6.)
    observe(m,7.)
    with pytest.raises(ValueError,match='current acquisition'):m.plan_write([-1,-1,-1,-1,0,99])


def test_each_observation_retains_its_own_camera_calibration(monkeypatch):
    m=memory(monkeypatch);observe(m,0.);m.write()
    camera=np.eye(4);camera[0,3]=.2
    k=np.array([[3.,0.,2.],[0.,3.,2.],[0.,0.,1.]])
    m.update(np.full((4,4),.2,np.float32),k,camera,np.eye(4),.5);m.write()
    assert m.observations[0].camera[0,3]==0 and m.observations[1].camera[0,3]==.2
    assert m.observations[0].intrinsic[0,0]==2 and m.observations[1].intrinsic[0,0]==3
    sources=next(iter(m.cells.values()))['sources']
    assert {float(source['origin'][0]) for source in sources}=={0.,.2}


@pytest.mark.parametrize("native", [False, True])
def test_replay_preparer_consumes_explicit_calibration_and_write_records(tmp_path,monkeypatch,native):
    import json
    from curvenav.data.prepare_six_frame_replay import prepare
    from curvenav.data.bspline_replay import sha
    from curvenav.data.six_frame_memory import MEMORY_CONTRACT
    original=memory(monkeypatch)
    for i in range(3):observe(original,i*.5);original.plan_write(original.sample())
    observations=list(original.observations.values())
    raw=dict(depths=np.stack([f.depth for f in observations]),timestamps=np.array([f.time for f in observations]),
        camera_intrinsics=np.stack([f.intrinsic for f in observations]),camera_to_body=np.stack([f.camera for f in observations]),
        body_to_world=np.stack([f.pose for f in observations]),memory_writes=original.writes,input_contract=MEMORY_CONTRACT)
    raw_path=tmp_path/'raw.pt';torch.save(raw,raw_path)
    item=dict(raw_path=str(raw_path),raw_sha256=sha(raw_path),clock=1.,acquisitions=3,teacher_values=torch.arange(14.),
        critic_replay_paths=torch.zeros(16,64,2),critic_replay_provenance={})
    item_path=tmp_path/'label.pt';torch.save(item,item_path)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(items=[dict(path=str(item_path),sha256=sha(item_path))])))
    rebuilt=memory(monkeypatch)
    replay=rebuilt
    actor=SimpleNamespace(new_memory=lambda:replay)
    if native:
        replay=SimpleNamespace(device=torch.device('cuda'),update=rebuilt.update,plan_write=rebuilt.plan_write,read_plan=rebuilt.read_plan)
        actor.policy=rebuilt.policy;actor.provenance=dict(bundle_sha256='a'*64)
        def condition(actor,item,raw,*,feature_provider):
            assert item['memory_plan']['raw_sha256']==item['raw_sha256']
            assert feature_provider(0) is feature_provider(0)
            return None,None
        monkeypatch.setattr('curvenav.data.prepare_six_frame_replay.live_condition',condition)
        actor.native_bank=lambda condition:dict(paths=torch.full((1,16,64,2),.125))
    prepare([manifest],actor,tmp_path/'out')
    emitted=json.loads((tmp_path/'out/family_000/manifest.json').read_text())
    assert emitted['geometry_input_contract']==('cuda_native_projection_v1' if native else 'cpu_collection_pending_native_replay_v1')
    assert emitted['training_ready']==native
    row=torch.load(emitted['items'][0]['path'],weights_only=False)
    plan=torch.load(row['memory_snapshot_path'],weights_only=False)
    np.testing.assert_array_equal(plan['frame_ids'],original.read_plan()['frame_ids'])
    assert plan['selection']['indices']==original.selection['indices']
    assert sha(row['memory_snapshot_path'])==row['memory_plan_sha256']
    assert torch.equal(row['teacher_values'],item['teacher_values'])
    assert rebuilt.policy.depth_encoder.calls==int(native)
    if native:
        assert (row['critic_replay_paths']==.125).all()
        binding=row['critic_replay_provenance']
        assert sha(binding['result_path'])==binding['result_sha256']
        assert sha(binding['candidates_path'])==binding['candidates_sha256']
        assert emitted['candidate_preparation']['model_sha256']==actor.provenance['bundle_sha256']
        with np.load(binding['candidates_path']) as saved:
            np.testing.assert_array_equal(saved['paths'][0],row['critic_replay_paths'].numpy())


def test_first_sample_of_old_frame_cannot_revive_space_already_seen_clear(monkeypatch):
    from curvenav.data import six_frame_memory as module
    m=memory(monkeypatch);old_project=m.project
    monkeypatch.setattr(m,'project',lambda frame:old_project(frame) if frame.index==0 else dict(world=np.empty((0,3)),pixels=np.empty(0,dtype=int),depth=np.empty(0),kind=np.empty(0,dtype=bool)))
    monkeypatch.setattr(module,'volumes_observed_clear',lambda boxes,depth,*args:np.full(len(boxes),depth[0,0]>.9))
    observe(m,0.,depth=.2);observe(m,.5,depth=.3);observe(m,1.,depth=1.)
    m.plan_write([-1,-1,-1,-1,1,2])  # Obstacle frame0 was buffered but never consumed.
    assert not m.cells
    observe(m,1.5,depth=.3)
    m.plan_write([-1,-1,-1,0,1,3])  # Newly sampled frame0 is older than consumed free frame2.
    assert not m.cells
