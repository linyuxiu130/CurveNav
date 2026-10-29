"""FP32 sample-mean oracle for actual batched native14 kernels and source contracts."""
from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy
import torch,numpy as np,pytest
from curvenav import build_policy
from curvenav.types import ConditionFeatures
from curvenav.deployment.bspline_bundle import BSplineMemoryBundle
from curvenav.training import bspline_objective as batched
from curvenav.models.policy import repeat_condition
from torch.nn import functional as F
from curvenav.training.bspline_replay import batch_indices,restore_full,full_state,epoch_budget,save_checkpoint,PROTOCOL
from curvenav.data.bspline_dataset import case_groups,sample_index
from curvenav.training.bspline_replay import capture_rng
from test_policy import tiny_config


def scalar_reference(actor,item,raw,tree,utility):
    encoded,geometry=scalar.live_condition(actor,item,raw)
    cfm=encoded.tokens.new_zeros(())
    if item['teacher_values'] is not None:
        condition=encoded;teacher=item['teacher_values']
        if item['goal_present'] and item['no_goal_teacher'] is not None and bool((torch.rand_like(encoded.goal_present)<.2).item()):
            teacher=item['no_goal_teacher']['teacher_values'];condition=replace(encoded,goal_present=torch.zeros_like(encoded.goal_present),terminal_goal=torch.zeros_like(encoded.terminal_goal))
        clean=actor.policy.curve_codec.coordinates_from_values(teacher[None]);z=torch.randn_like(clean);time=torch.tensor([1., .5],dtype=clean.dtype)
        state=(1-time[:,None])*clean+time[:,None]*z
        velocity=actor.policy._predict_velocity(state,time,repeat_condition(condition,2),actor.policy.trajectory_decoder.project_condition_memory(condition))
        cfm=(velocity-(z-clean)).square().mean()
    bank=actor.candidates(encoded,item['goal']);gf,_=scalar.observed_risk(bank['values'],tree,item['point_error_m'],actor.head.radius)
    lh=scalar.geometry_reconstruction_loss(*geometry,item['target_field'][None])
    lq=encoded.tokens.new_zeros(())
    if item['q_supervision_available']:
        qbatch=dict(source_grid_index=torch.zeros(1,dtype=torch.long),source_origin_xy=torch.tensor(item['view'][:2],dtype=torch.float64)[None],source_yaw_rad=torch.tensor([item['view'][2]],dtype=torch.float64),point_goal=item['goal'][None],goal_present=encoded.goal_present[:,0].bool())
        n=len(item['q_expert_values']);paths=bank['paths']
        if n:
            expert,_=actor.policy.curve_codec.decode_values(item['q_expert_values']);paths=torch.cat((expert,paths))
        paths=paths.detach();target=utility(paths[None],qbatch).score
        error=F.smooth_l1_loss(actor.policy.score_candidate_paths(encoded,item['goal'][None],paths[None]),target,reduction='none')
        recorded=item['critic_replay_paths'].detach();rt=utility(recorded[None],qbatch).score
        re=F.smooth_l1_loss(actor.policy.score_candidate_paths(encoded,item['goal'][None],recorded[None]),rt)
        proposal=(error[:,n:].mean()+re)/2;lq=.25*error[:,:n].mean()+.75*proposal if n else proposal
    trajectory=encoded.tokens.new_zeros(())
    if item['teacher_values'] is not None:
        targets,_=actor.policy.curve_codec.decode_values(item['trajectory_teacher_values'])
        trajectory=torch.stack([(((path-targets)/actor.policy.planning_horizon_m).square().mean((-1,-2))
            +((torch.diff(path,dim=-2)-torch.diff(targets,dim=-2))*(path.shape[-2]-1)/actor.policy.planning_horizon_m).square().mean((-1,-2))).min() for path in bank['paths']]).mean()
    return dict(CFM=cfm,TRAJ=trajectory,GF=gf,LH=lh,LQ=lq),bank

scalar=SimpleNamespace(losses=scalar_reference,live_condition=None,observed_risk=None,geometry_reconstruction_loss=None)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.velocity=torch.nn.Linear(18,14)
    def project_condition_memory(self,condition):return condition.tokens.mean((1,2))[:,None]


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.trajectory_decoder=Decoder();self.trajectory_evaluator=torch.nn.Linear(3,1)
        self.curve_codec=build_policy(tiny_config()).curve_codec;self.integration_steps=2;self.planning_horizon_m=3.6
        self.register_buffer('inference_source',torch.randn(32,2,14));self.calls=[];self.q_calls=[]
    def _predict_velocity(self,state,time,condition,memory):
        self.calls.append((len(state),len(memory)))
        m=memory.repeat_interleave(len(state)//len(memory),0)
        x=torch.cat((state,time[:,None],condition.goal_present,condition.terminal_goal[:,:1]*condition.goal_present,m),-1)
        return self.trajectory_decoder.velocity(x)
    def score_candidate_paths(self,condition,goal,paths):
        self.q_calls.append(tuple(paths.shape[:2]));batch,count=paths.shape[:2]
        actual=torch.where(condition.goal_present.bool(),goal,torch.zeros_like(goal))[:,0]
        x=torch.stack((paths.mean((-1,-2)),condition.tokens.mean((1,2))[:,None].expand(batch,count),actual[:,None].expand(batch,count)),-1)
        return self.trajectory_evaluator(x).squeeze(-1)


def fixture(monkeypatch):
    torch.manual_seed(321);policy=ToyPolicy();head=torch.nn.Linear(1,1,bias=False);head.radius=.3;reader=torch.nn.Linear(1,1,bias=False)
    actor=BSplineMemoryBundle(None,policy,head,{},{});actor.reader=reader
    items=[];records=[];counts=dict(live=0)
    for index in range(6):
        goal=index%2==0;partial=index==3;teacher=torch.linspace(.02,.12,14)*(index+1)
        item=dict(task_state=dict(panel="J",case={}),q_supervision_available=not partial,teacher_values=None if partial else teacher,goal_present=goal,goal=torch.tensor([.2*(index+1),.1]) if goal else torch.zeros(2),
            trajectory_teacher_values=torch.empty(0,14) if partial else teacher[None],
            no_goal_teacher=dict(teacher_values=teacher*.5) if index==0 else None,q_expert_values=torch.stack([teacher]*([1,2,1,0,0,2][index])) if [1,2,1,0,0,2][index] else torch.zeros(0,14),
            critic_replay_paths=torch.full((32,64,2),.05*(index+1)),utility_goal=torch.tensor([.1*index,0.]),view=np.array([index,2.,0.]),point_error_m=0.,
            target_field=torch.full((6,2,2),.02*index),loss_availability=dict(CFM=False,GF=True,LH=True,LQ=False,own_expert=False) if partial else None)
        def utility(paths,batch,index=index):
            assert batch['source_origin_xy'][0,0]==index
            return SimpleNamespace(score=(paths[:,:,-1,0]*.1+batch['point_goal'][:,0,None]).detach())
        items.append(item);records.append((item,index,index+1.,utility))
    def live(actor,item,raw,**kwargs):
        counts['live']+=1;t=(actor.head.weight.mean()*actor.reader.weight.mean()+raw*.03).expand(1,4,3)
        encoded=ConditionFeatures(tokens=t,token_valid=torch.ones(1,4,dtype=torch.bool),metric_position=torch.zeros(1,4,3),surface_hit=torch.zeros(1,4,dtype=torch.bool),frame_age=torch.zeros(1,4),motion_token=torch.zeros(1,4,dtype=torch.bool),terminal_goal=item['goal'][None],goal_present=torch.tensor([[float(item['goal_present'])]]),configuration_field=actor.head.weight.mean().expand(1,6,2,2))
        return encoded,(actor.head.weight.mean().expand(1,6,2,2),actor.head.weight.mean().expand(1,2,2,2))
    risk=lambda values,tree,*args:(values.square().mean()*tree,{})
    lh=lambda field,logits,target,*args,**kwargs:(field-target).square().mean()
    for module in (batched,scalar):
        monkeypatch.setattr(module,'live_condition',live);monkeypatch.setattr(module,'geometry_reconstruction_loss',lh)
    def live_batch(actor,records,maps):
        values=[batched.live_condition(actor,*r[:2]) for r in records]
        return batched.cat_conditions([c for c,_ in values]),tuple(torch.cat([v[1][i] for v in values]) for i in range(2))
    monkeypatch.setattr(batched,'live_conditions',live_batch)
    monkeypatch.setattr(scalar,'observed_risk',risk)
    from curvenav.training import bspline_image_batch
    monkeypatch.setattr(bspline_image_batch,'step_feature_maps',lambda *a,**kw:{})
    monkeypatch.setattr(batched,'_nearest_sites',lambda queries,records,bounds:None)
    monkeypatch.setattr(batched,'_risk_from_sites',lambda actor,values,paths,pad,sites,records:torch.stack([risk(values[i],r[2])[0] for i,r in enumerate(records)]).mean())
    monkeypatch.setattr(torch,'rand_like',lambda x:torch.zeros_like(x))  # Exercise paired no-goal teacher exactly.
    return actor,items,records,counts


def test_true_batched_two_step_losses_and_all_gradients_match_fp32_sample_mean(monkeypatch):
    actor,items,records,counts=fixture(monkeypatch);params=list(actor.policy.parameters())+list(actor.head.parameters())+list(actor.reader.parameters())
    torch.manual_seed(1001);reference=[scalar.losses(actor,*r)[0] for r in records]
    expected={name:torch.stack([r[name] for r in reference]).mean() for name in reference[0]}
    oldgrads=torch.autograd.grad(sum(expected.values()),params,allow_unused=True)
    actor.policy.calls.clear();actor.policy.q_calls.clear();counts['live']=0
    torch.manual_seed(1001);actual,bank=batched.batched_losses(actor,records)
    forward_calls=list(actor.policy.calls)
    newgrads=torch.autograd.grad(sum(actual.values()),params,allow_unused=True)
    for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=2e-5,atol=2e-6)
    for old,new in zip(oldgrads,newgrads):
        assert (old is None)==(new is None)
        if old is not None:torch.testing.assert_close(new,old,rtol=3e-5,atol=3e-6)
    assert counts['live']==6  # Original B1 history forward per item, real batch thereafter.
    assert forward_calls[0]==(6*2,6)
    assert len(forward_calls)==3  # CFM once, exactly two G calls for the entire mixed batch.
    assert actor.policy.q_calls==[(6,50)]
    assert bank['paths'].shape==(6,16,64,2)


def test_goal_nogoal_native_sources_batched_without_unused_q_scoring(monkeypatch):
    actor,items,records,_=fixture(monkeypatch);conditions=[batched.live_condition(actor,r[0],r[1])[0] for r in records]
    condition=batched.cat_conditions(conditions);reference=[actor.candidates(c,i['goal'])['values'] for c,i in zip(conditions,items)]
    actor.policy.q_calls.clear();bank=actor.native_bank(condition,checkpoint_steps=True)
    assert actor.policy.q_calls==[]
    torch.testing.assert_close(bank['values'],torch.stack(reference),rtol=2e-5,atol=2e-6)
    # Decoder input non-goal mask never opens, and all original fields are kept when slicing.
    for index,c in enumerate(conditions):
        selected=batched.select_condition(condition,[index])
        for name in vars(c):torch.testing.assert_close(getattr(c,name),getattr(selected,name),rtol=0,atol=0)


def test_recorded_q_cache_reuses_static_labels_but_refreshes_current_candidates(monkeypatch):
    actor,items,records,_=fixture(monkeypatch);item=items[0]
    condition,_=batched.live_condition(actor,*records[0][:2]);calls=[]
    def utility(paths,batch):
        calls.append(paths.shape[1]);return records[0][3](paths,batch)
    a=batched.prepare_critic_inputs(actor,item,condition.goal_present,torch.zeros(16,64,2),utility)
    batched.label_critic_inputs([a])
    b=batched.prepare_critic_inputs(actor,item,condition.goal_present,torch.ones(16,64,2),utility)
    batched.label_critic_inputs([b])
    assert calls==[32,17,17]
    assert a['recorded_targets'] is b['recorded_targets']
    assert not torch.equal(a['targets'][:,1:],b['targets'][:,1:])


def test_duplicate_samples_share_only_current_step_encoding(monkeypatch):
    actor,items,records,counts=fixture(monkeypatch)
    records=[records[1],records[1]];params=list(actor.policy.parameters())+list(actor.head.parameters())+list(actor.reader.parameters())
    torch.manual_seed(51);expected=[scalar.losses(actor,*r)[0] for r in records]
    expected={k:torch.stack([r[k] for r in expected]).mean() for k in expected[0]}
    expected_grads=torch.autograd.grad(sum(expected.values()),params,allow_unused=True)
    counts['live']=0;torch.manual_seed(51);actual,_=batched.batched_losses(actor,records)
    actual_grads=torch.autograd.grad(sum(actual.values()),params,allow_unused=True)
    assert counts['live']==1
    for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=2e-5,atol=2e-6)
    for a,b in zip(actual_grads,expected_grads,strict=True):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
    batched.batched_losses(actor,records)
    assert counts['live']==2


def test_real_batch_sampler_uses_one_uniform_task_cycle():
    rows=[(dict(trial_id=f't{i%8}',raw_sha256=f'r{i}'),None,None,None) for i in range(32)]
    g=case_groups(rows,range(16));m=case_groups(rows,range(16,32))
    for rank in range(8):
        assert batch_indices(g,m,3,7,rank,8,2)==[sample_index(g+m,160+rank*2+s) for s in range(2)]
        selected=batch_indices(g,m,3,7,rank,8,4)
        positions=[320+rank*4+s for s in range(4)]
        assert selected==[sample_index(g+m,p) for p in positions]


def test_production_adamw_schedule_ema_full_roundtrip_and_epoch_checkpoint(tmp_path,monkeypatch):
    import random,json
    from curvenav.training.optimizer import build_optimizer,build_cosine_schedule
    from curvenav.training.ema import ExponentialMovingAverage
    from curvenav.training import bspline_replay as caller
    cfg=tiny_config().training
    goals=[[[0]*15] for _ in range(16)];memories=[[[0]*43] for _ in range(24)]
    for batch,steps in ((2,108),(4,54),(8,27)):
        budget=epoch_budget(goals,memories,8,batch,cfg,1e-5)
        assert budget['steps_per_epoch']==steps and budget['total_steps']==50*steps and budget['warmup_steps']==5*steps
    def make():
        model=torch.nn.ModuleDict(dict(policy=torch.nn.Linear(3,2),head=torch.nn.Linear(2,1)))
        opt=build_optimizer(model,1e-5,.01);schedule=build_cosine_schedule(opt,10,2,.01);ema=ExponentialMovingAverage(model,.9999)
        return model,opt,schedule,ema
    torch.manual_seed(18);np.random.seed(18);random.seed(18)
    model,opt,schedule,ema=make();assert not opt.state and ema.num_updates==0
    assert all(g['betas']==(.9,.95) and g['fused'] for g in opt.param_groups)
    assert [g['weight_decay'] for g in opt.param_groups]==[.01,0.]
    def update(m,o,s,e):
        x=torch.randn(4,3)+np.random.rand()+random.random();o.zero_grad(set_to_none=True)
        m['head'](m['policy'](x)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();s.step();e.update()
    update(model,opt,schedule,ema);raw=deepcopy(model.state_dict());rng=capture_rng(torch.device('cpu'))
    contract=dict(cnn_cache_sha256=None,world_size=8,effective_batch=16,row_counts=dict(all=345,goal=127,memory=218),epoch_budget=dict(total_steps=10,warmup_steps=2))
    state=full_state(model,opt,schedule,ema,[rng]*8,1,contract,'raw','ema',dict(sha256='warmstart_only',legacy_optimizer_restored=False));torch.save(state,tmp_path/'full.pt')
    update(model,opt,schedule,ema);expected=deepcopy(model.state_dict());expected_ema=deepcopy(ema.state_dict());expected_rng=torch.get_rng_state().clone()
    restored,ro,rs,re=make();restored.load_state_dict(raw)
    loaded=restore_full(tmp_path/'full.pt',restored,ro,rs,re,contract=contract,raw_sha='raw',rank=0,device=torch.device('cpu'));assert loaded['steps']==1
    update(restored,ro,rs,re)
    for name,value in restored.state_dict().items():torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
    assert rs.state_dict()==schedule.state_dict() and re.num_updates==2
    for name,value in re.shadow.items():torch.testing.assert_close(value,expected_ema['shadow'][name],rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    for badcontract,badsha in ((dict(contract,effective_batch=32),'raw'),(contract,'ema')):
        with pytest.raises(ValueError):restore_full(tmp_path/'full.pt',restored,ro,rs,re,contract=badcontract,raw_sha=badsha,rank=0,device=torch.device('cpu'))
    torch.save(dict(format='bspline_memory_training_v1'),tmp_path/'legacy.pt')
    with pytest.raises(ValueError,match='complete AdamW/schedule/EMA'):restore_full(tmp_path/'legacy.pt',restored,ro,rs,re,contract=contract,raw_sha='raw',rank=0,device=torch.device('cpu'))
    def save_bundle(path,**kwargs):
        torch.save(dict(weights_kind=kwargs['provenance']['weights_kind'],components={k:deepcopy(kwargs[k].state_dict()) for k in ('policy','head')}),path)
    monkeypatch.setattr(caller,'save_bundle',save_bundle)
    actor=SimpleNamespace(policy=restored['policy'],head=restored['head'],provenance={},architecture={})
    snapshot=tmp_path/'checkpoints/epoch_001';payloads=[dict(rng=capture_rng(torch.device('cpu')),performance=dict(rank=i)) for i in range(8)]
    before=deepcopy(restored.state_dict())
    save_checkpoint(snapshot,actor,restored,ro,rs,re,dict(contract,gradient_communication='nccl_cuda_fp32_sum_mean_v1',cnn_batch_size=16,cnn_feature_contract='same_sensor_step_full_image_batch_v1_not_B1_bitexact'),dict(sha256='warmstart'),{},2,1,2,dict(epochs=50),'parent',None,payloads,[],tmp_path/'source',create=True)
    proof=json.loads((snapshot/'result.json').read_text());assert proof['complete'] and proof['evaluation_weights_kind']=='ema'
    assert torch.load(snapshot/'model.pt',weights_only=False)['weights_kind']=='raw'
    assert torch.load(snapshot/'ema_model.pt',weights_only=False)['weights_kind']=='ema'
    saved=torch.load(snapshot/'training_state.pt',weights_only=False);assert saved['steps']==saved['ema']['num_updates']==saved['scheduler']['last_epoch']==2
    for name,value in restored.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)


def test_new_data_stage_retains_adamw_ema_rng_and_current_lr(tmp_path):
    from curvenav.training.optimizer import build_optimizer,build_cosine_schedule
    from curvenav.training.ema import ExponentialMovingAverage
    model=torch.nn.Linear(3,2)
    optimizer=build_optimizer(model,2e-4,.01)
    scheduler=build_cosine_schedule(optimizer,20,2,.01)
    ema=ExponentialMovingAverage(model,.9999)
    model(torch.ones(2,3)).square().mean().backward();optimizer.step();scheduler.step();ema.update()
    contract=dict(actor_semantics='native',precision='bf16',objective_version='same',optimizer='AdamW',
        integration_steps=2,candidates=16,validation_data=dict(files={'validation':'frozen'}),row_counts=dict(all=566))
    rng=capture_rng(torch.device('cpu'))
    state=full_state(model,optimizer,scheduler,ema,[rng],1,contract,'raw','ema',{})
    path=tmp_path/'state.pt';torch.save(state,path)
    restored=torch.nn.Linear(3,2);restored.load_state_dict(model.state_dict())
    lr=optimizer.param_groups[0]['lr']
    opt=build_optimizer(restored,lr,.01);schedule=build_cosine_schedule(opt,5500,0,.01)
    average=ExponentialMovingAverage(restored,.9999)
    updated=dict(contract,row_counts=dict(all=3019),sampling='uniform_trial',epoch_budget=dict(total_steps=5500))
    restore_full(path,restored,opt,schedule,average,contract=updated,raw_sha='raw',rank=0,device=torch.device('cpu'),new_data_stage=True)
    assert schedule.last_epoch==0 and opt.param_groups[0]['lr']==lr
    assert schedule.base_lrs==[lr,lr] and average.num_updates==ema.num_updates
    for name,value in ema.shadow.items():torch.testing.assert_close(average.shadow[name],value,rtol=0,atol=0)
    for previous,current in zip(optimizer.state.values(),opt.state.values(),strict=True):
        for name in previous:torch.testing.assert_close(current[name],previous[name],rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),rng['torch'])
    for bad in (dict(updated,objective_version='different'),dict(updated,validation_data=dict(files={'validation':'changed'}))):
        with pytest.raises(ValueError):restore_full(path,restored,opt,schedule,average,contract=bad,raw_sha='raw',rank=0,device=torch.device('cpu'),new_data_stage=True)
    with pytest.raises(ValueError,match='contract differs'):
        restore_full(path,restored,opt,schedule,average,contract=updated,raw_sha='raw',rank=0,device=torch.device('cpu'))


@pytest.mark.parametrize('pool_size',[0,1])
def test_partial_defined_q_keeps_gf_lh_and_no_generator_q_gradient(monkeypatch,pool_size):
    actor,items,records,_=fixture(monkeypatch)
    item=items[3];item['loss_availability']['LQ']=True;item['q_supervision_available']=True
    item['q_expert_values']=torch.full((pool_size,14),.1)
    terms,_=batched.batched_losses(actor,[records[3]])
    assert terms['CFM']==0 and terms['GF']>0 and terms['LH']>0 and terms['LQ'].requires_grad
    decoder=list(actor.policy.trajectory_decoder.parameters())
    grads=torch.autograd.grad(terms['LQ'],decoder+list(actor.head.parameters())+list(actor.reader.parameters()),allow_unused=True)
    assert all(g is None for g in grads[:len(decoder)])
    assert all(g is not None and torch.isfinite(g).all() for g in grads[len(decoder):])


def test_partial_undefined_q_never_queries_source_and_retains_gf_lh(monkeypatch):
    actor,items,records,_=fixture(monkeypatch)
    item,raw,tree,_=records[3]
    def forbidden(*args):pytest.fail('Undefined source utility queried')
    terms,_=batched.batched_losses(actor,[(item,raw,tree,forbidden)])
    assert terms['CFM']==terms['LQ']==0 and terms['GF']>0 and terms['LH']>0
    parameters=list(actor.policy.trajectory_evaluator.parameters())
    assert all(g is None for g in torch.autograd.grad(sum(terms.values()),parameters,allow_unused=True))


def test_g_euler_checkpoint_matches_fp32_direct_batched_gradients(monkeypatch):
    actor,items,records,_=fixture(monkeypatch)
    parameters=list(actor.policy.parameters())+list(actor.head.parameters())+list(actor.reader.parameters())
    results=[]
    for checkpointed in (False,True):
        conditions=[batched.live_condition(actor,*record[:2])[0] for record in records[:2]]
        bank=actor.native_bank(batched.cat_conditions(conditions),checkpoint_steps=checkpointed)
        loss=bank['values'].square().mean()+bank['paths'].square().mean()
        gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        results.append((bank,loss,gradients))
    original,checkpointed=results
    for key in ('values','paths'):torch.testing.assert_close(original[0][key],checkpointed[0][key],rtol=0,atol=0)
    torch.testing.assert_close(original[1],checkpointed[1],rtol=0,atol=0)
    for a,b in zip(original[2],checkpointed[2]):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_overlapped_nearest_matches_per_row_risk_and_gradients():
    from scipy.spatial import cKDTree
    from curvenav.data.bspline_replay import dense_curve,observed_risk
    torch.manual_seed(17)
    values=(torch.randn(3,32,14)*.2).requires_grad_()
    records=[(dict(point_error_m=.01*i),None,cKDTree(np.random.default_rng(i).normal(size=(200,2))),None) for i in range(3)]
    actor=SimpleNamespace(head=SimpleNamespace(radius=.3))
    paths,pad=dense_curve(values.flatten(0,1));paths=paths.reshape(3,-1,2).double()
    bounds=pad.reshape(3,-1).amax(-1).detach().double().numpy()+np.array([.3+.01*i for i in range(3)])
    sites=batched._nearest_sites(paths.detach().cpu().numpy(),records,bounds)
    actual=batched._risk_from_sites(actor,values,paths,pad,sites,records)
    expected=torch.stack([observed_risk(values[i],r[2],r[0]['point_error_m'],.3)[0] for i,r in enumerate(records)]).mean()
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    old=torch.autograd.grad(expected,values,retain_graph=True)[0]
    new=torch.autograd.grad(actual,values)[0]
    torch.testing.assert_close(new,old,rtol=1e-6,atol=1e-7)


def test_nearest_search_overlaps_critic_work(monkeypatch):
    import threading
    actor,items,records,_=fixture(monkeypatch)
    started=threading.Event();release=threading.Event()
    def nearest(queries,rows,bounds):
        started.set()
        assert release.wait(10)
        return None
    original=batched.batched_critic
    def critic(*args,**kwargs):
        assert started.wait(10)
        release.set()
        return original(*args,**kwargs)
    monkeypatch.setattr(batched,'_nearest_sites',nearest)
    monkeypatch.setattr(batched,'batched_critic',critic)
    terms,_=batched.batched_losses(actor,records)
    sum(terms.values()).backward()
    assert release.is_set()


def test_native16_selects_sources_before_exactly_two_euler_calls(monkeypatch):
    actor,items,records,_=fixture(monkeypatch)
    condition=batched.cat_conditions([batched.live_condition(actor,*r[:2])[0] for r in records[:2]])
    scale=torch.tensor(.3,requires_grad=True)
    calls=[]
    def velocity(state,time,condition,memory):
        calls.append((state.shape, time.detach().clone(), condition.goal_present.detach().clone()))
        return scale*state+time[:,None]
    monkeypatch.setattr(actor.policy,'_predict_velocity',velocity)
    bank=actor.native_bank(condition)
    z=torch.stack((actor.policy.inference_source[:16,0],actor.policy.inference_source[16:32,1]))
    expected_state=(1-scale/2)**2*z-.5*(1-scale/2)-.25
    expected=actor.policy.curve_codec.values_from_coordinates(expected_state.flatten(0,1)).reshape(2,16,14)
    torch.testing.assert_close(bank['values'],expected,rtol=1e-6,atol=1e-6)
    actual_grad=torch.autograd.grad(bank['values'].square().sum(),scale,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.square().sum(),scale)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-6,atol=1e-5)
    assert len(calls)==2 and all(shape==(32,14) for shape,_,_ in calls)  # Two scenes, sixteen each.
    assert torch.equal(calls[0][1],torch.ones(32)) and torch.equal(calls[1][1],torch.full((32,),.5))
    assert calls[0][2][:16].all() and not calls[0][2][16:].any()


def test_final_trajectory_supervises_every_candidate_and_only_matching_teachers(monkeypatch):
    actor,items,_,_=fixture(monkeypatch)
    teacher=items[0]['teacher_values'].clone().requires_grad_()
    item=dict(items[0],trajectory_teacher_values=teacher[None])
    target=actor.policy.curve_codec.decode_values(teacher[None])[0].detach()
    paths=target[:,None].expand(1,16,-1,-1).clone()
    paths[:,1:]+=1.
    paths.requires_grad_()
    loss=batched.generated_trajectory_loss(actor,paths,[item])
    expected=15/16/actor.policy.planning_horizon_m**2
    torch.testing.assert_close(loss,torch.tensor(expected))
    gradient,target_gradient=torch.autograd.grad(loss,(paths,teacher),allow_unused=True)
    assert target_gradient is None and not gradient[:,0].any() and (gradient[:,1:]>0).all()


def test_loss_gradient_ownership_includes_generated_trajectory(monkeypatch):
    actor,_,records,_=fixture(monkeypatch)
    terms,_=batched.batched_losses(actor,records)
    generator=list(actor.policy.trajectory_decoder.parameters())
    critic=list(actor.policy.trajectory_evaluator.parameters())
    shared=list(actor.head.parameters())+list(actor.reader.parameters())
    for name in ('CFM','TRAJ','GF'):
        grads=torch.autograd.grad(terms[name],generator+critic+shared,retain_graph=True,allow_unused=True)
        assert any(g is not None and g.abs().sum()>0 for g in grads[:len(generator)])
        assert all(g is None for g in grads[len(generator):len(generator)+len(critic)])
        assert all(g is not None and g.abs().sum()>0 for g in grads[-len(shared):])
    qgrads=torch.autograd.grad(terms['LQ'],generator+critic+shared,retain_graph=True,allow_unused=True)
    assert all(g is None for g in qgrads[:len(generator)])
    assert all(g is not None and g.abs().sum()>0 for g in qgrads[len(generator):])


def test_critic_accepts_mixed_historical32_and_current16_banks(monkeypatch):
    actor,items,records,_=fixture(monkeypatch)
    for item in items[::2]:item['critic_replay_paths']=item['critic_replay_paths'][:16]
    terms,_=batched.batched_losses(actor,records)
    assert torch.isfinite(terms['LQ'])
    assert actor.policy.q_calls==[(6,50)]


def test_generated_trajectory_loss_penalizes_oscillation_not_only_point_error(monkeypatch):
    actor,items,_,_=fixture(monkeypatch)
    teacher=torch.zeros(1,14)
    item=dict(items[0],trajectory_teacher_values=teacher)
    target=actor.policy.curve_codec.decode_values(teacher)[0]
    shift=torch.ones_like(target);shift[:,0]=0
    oscillation=shift.clone();oscillation[:,::2]*=-1
    smooth=(target+shift)[:,None].requires_grad_()
    curled=(target+oscillation)[:,None].requires_grad_()
    smooth_loss=batched.generated_trajectory_loss(actor,smooth,[item])
    curled_loss=batched.generated_trajectory_loss(actor,curled,[item])
    assert curled_loss>smooth_loss*10
    assert torch.autograd.grad(curled_loss,curled)[0].abs().sum()>0
