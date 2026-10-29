"""Full native B-spline joint training: batched math, AdamW, schedule and all-module EMA."""
from pathlib import Path
import argparse,json,os,time,random
import numpy as np
import torch
import torch.distributed as dist
from curvenav import experiment_io
from curvenav.data.bspline_replay import sha
from curvenav.deployment.bspline_bundle import load_bundle,save_bundle,CANDIDATES,INTEGRATION_STEPS
from curvenav.data.bspline_dataset import SAMPLING,load_items,case_groups,sample_index,dataset_manifests
from curvenav.training.bspline_objective import batched_losses


PROTOCOL='native_bspline_six_frame_joint_v3'
SOURCE_ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=tuple(str(path.relative_to(SOURCE_ROOT)) for path in sorted(SOURCE_ROOT.rglob('*.py')))

def capture_rng(device):
    return dict(torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        numpy=np.random.get_state(), python=random.getstate())

def average_gradients(parameters, device, world):
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters])
    assert torch.isfinite(flat).all()
    if world>1:dist.all_reduce(flat)
    flat.div_(world)
    offset=0
    for parameter in parameters:
        parameter.grad=flat[offset:offset+parameter.numel()].reshape_as(parameter).to(device)
        offset+=parameter.numel()
    return flat.norm()



def rank_batch_sizes(world,batch_per_rank,global_batch=None):
    total=world*batch_per_rank if global_batch is None else global_batch
    count,remainder=divmod(total,world)
    return [count+(rank<remainder) for rank in range(world)]


def batch_indices(goal_cases,memory_cases,outer,data_start,rank,world,batch_per_rank,*,global_batch=None):
    sizes=rank_batch_sizes(world,batch_per_rank,global_batch)
    offset=sum(sizes[:rank]);cases=goal_cases+memory_cases
    return [sample_index(cases,(data_start+outer)*sum(sizes)+offset+slot)
        for slot in range(sizes[rank])]


def balanced_batch_indices(goal_cases, memory_cases, outer, data_start, rank,
                           world, batch_per_rank, source_keys,*,global_batch=None):
    """Keep the uniform-task global batch, assigning shared frames to one rank."""
    quotas = rank_batch_sizes(world,batch_per_rank,global_batch)
    global_items = []
    for source_rank in range(world):
        for index in batch_indices(
                goal_cases, memory_cases, outer, data_start, source_rank,
                world, batch_per_rank,global_batch=global_batch):
            global_items.append((len(global_items), index))

    rank_keys = [set() for _ in range(world)]
    assigned = [[] for _ in range(world)]
    for order, index in sorted(
            global_items, key=lambda item: (-len(source_keys[item[1]]), item[0])):
        candidates = [r for r in range(world) if len(assigned[r]) < quotas[r]]
        row_keys = source_keys[index]
        target = min(candidates, key=lambda r: (len(row_keys - rank_keys[r]), len(rank_keys[r] | row_keys), r))
        rank_keys[target].update(row_keys)
        assigned[target].append((order, index))

    assert [len(rows) for rows in assigned]==quotas
    return [index for _, index in sorted(assigned[rank])]


FULL_FORMAT='native_batched_joint_full_training_v3'


def epoch_budget(goal_cases,memory_cases,world,batch_per_rank,config,learning_rate,*,global_batch=None):
    import math
    batch=sum(rank_batch_sizes(world,batch_per_rank,global_batch));cases=goal_cases+memory_cases
    steps=math.ceil(max(len(c[0]) for c in cases)*len(cases)/batch)
    return dict(checkpoint_every_epochs=config.checkpoint_every_epochs,steps_per_epoch=steps,epochs=config.epochs,warmup_epochs=config.warmup_epochs,
        total_steps=steps*config.epochs,warmup_steps=steps*config.warmup_epochs,
        minimum_factor=config.min_learning_rate_factor,peak_learning_rate=learning_rate,
        weight_decay=config.weight_decay,ema_decay=config.ema_decay)


def full_state(model,optimizer,scheduler,ema,rank_rng,steps,contract,raw_sha,ema_sha,warmstart):
    names={id(p):name for name,p in model.named_parameters()}
    state=dict(optimizer=optimizer.state_dict(),optimizer_names=[[names[id(p)] for p in group['params']] for group in optimizer.param_groups],
        rank_rng=rank_rng,steps=steps,data_steps=steps,critic_updates=ema.num_updates,contract=contract,model_sha256=raw_sha)
    state.update(format=FULL_FORMAT,scheduler=scheduler.state_dict(),ema=ema.state_dict(),ema_model_sha256=ema_sha,warmstart=warmstart)
    return state


def restore_full(path,model,optimizer,scheduler,ema,*,contract,raw_sha,rank,device,new_data_stage=False):
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if state['format']!=FULL_FORMAT:
        raise ValueError("Resume requires this trainer's complete AdamW/schedule/EMA state")
    if new_data_stage:
        for key in ('actor_semantics','precision','objective_version','optimizer','integration_steps','candidates'):
            if state['contract'][key]!=contract[key]:raise ValueError(f'New data stage changes model/objective contract: {key}')
        if state['contract']['validation_data']['files']!=contract['validation_data']['files']:
            raise ValueError('New data stage must retain the original validation files')
    elif state['contract']!=contract:raise ValueError('Full training source/data/batch/epoch budget contract differs')
    if state['model_sha256']!=raw_sha:raise ValueError('Resume requires the raw model bundle matching training_state, not EMA')
    names={id(p):name for name,p in model.named_parameters()}
    groups=[[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    if groups!=state['optimizer_names']:raise ValueError('Full AdamW decay/no-decay parameter groups differ')
    rates=[group['lr'] for group in optimizer.param_groups]
    optimizer.load_state_dict(state['optimizer'])
    if new_data_stage:
        for group,rate,base in zip(optimizer.param_groups,rates,scheduler.base_lrs,strict=True):
            group['lr']=rate;group['initial_lr']=base
    else:scheduler.load_state_dict(state['scheduler'])
    ema.load_state_dict(state['ema'])
    rng=state['rank_rng'][rank];torch.set_rng_state(rng['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'],device)
    np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    return state


def save_checkpoint(folder,actor,model,optimizer,scheduler,ema,contract,warmstart,base_config,steps,start,
                    batch_per_rank,budget,raw_parent_sha,resume_path,payloads,logs,source_root,*,create,validation_history=()):
    if create:folder.mkdir(parents=True,exist_ok=False)
    provenance=dict(actor.provenance,training=PROTOCOL,training_protocol=PROTOCOL,objective_version='independent_free_task_admissible_omni_q_v6',weights_kind='raw',
        trained_for_current_protocol=True,candidates=CANDIDATES,integration_steps=INTEGRATION_STEPS,
        objective='CFM2 + final generated trajectory fit + observed-hit GF + sensor LH + detached-path Q',
        full_stage_steps=steps,cumulative_updates=ema.num_updates,incremental_updates=steps-start,data_steps=steps,critic_updates=ema.num_updates,
        critic_steps_per_joint=1,batch_per_rank=batch_per_rank,cnn_feature_contract=contract['cnn_feature_contract'],effective_batch=contract['effective_batch'],warmstart=warmstart,
        new_AdamW_stage=resume_path is None,new_EMA_stage=resume_path is None,parent_bundle_sha256=raw_parent_sha,
        optimizer_parent_state_sha256=sha(resume_path) if resume_path else None,epoch_budget=budget,
        learning_rates=[g['lr'] for g in optimizer.param_groups],EMA_num_updates=ema.num_updates,deterministic_algorithms=True,
        gradient_communication=contract['gradient_communication'],nccl_protocol='LL',control_communication='gloo_cpu_totals_and_rng_v1')
    raw_model=folder/'model.pt';save_bundle(raw_model,base_config=base_config,policy=actor.policy,head=actor.head,architecture=actor.architecture,provenance=provenance)
    ema_model=folder/'ema_model.pt'
    with ema.average_parameters(model):
        save_bundle(ema_model,base_config=base_config,policy=actor.policy,head=actor.head,architecture=actor.architecture,provenance=dict(provenance,weights_kind='ema'))
    state=full_state(model,optimizer,scheduler,ema,[p['rng'] for p in payloads],steps,contract,sha(raw_model),sha(ema_model),warmstart)
    state['validation_history']=list(validation_history)
    torch.save(state,folder/'training_state.pt')
    experiment_io.write(folder/'result.json',dict(validation_history=list(validation_history),complete=True,provenance=provenance,contract=contract,all_modules_trainable=contract['cnn_cache_sha256'] is None,all_learning_rates_positive=True,
        training_rows=contract['row_counts']['all'],goal_rows=contract['row_counts']['goal'],memory_rows=contract['row_counts']['memory'],source_snapshot_root=str(source_root.resolve()),model_sha256=state['model_sha256'],ema_model_sha256=state['ema_model_sha256'],
        evaluation_model=str(ema_model.resolve()),evaluation_weights_kind='ema',training_state_sha256=sha(folder/'training_state.pt'),
        performance_by_rank=[p['performance'] for p in payloads],trace=logs))


def main():
    from curvenav.training.optimizer import build_optimizer,build_cosine_schedule
    from curvenav.training.ema import ExponentialMovingAverage
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True,help='Current model checkpoint, or its matched raw bundle for --resume')
    parser.add_argument('--validation-manifests',type=Path,nargs='+')
    parser.add_argument('--stop-on-cnn-plateau',action='store_true')
    parser.add_argument('--cnn-cache',type=Path,help='Manifest from the fixed CNN; freezes only backbone and image projection')
    parser.add_argument('--resume',type=Path,help='Complete state from this trainer')
    parser.add_argument('--new-data-stage',action='store_true',help='Retain model/AdamW/EMA/RNG; start a new data epoch budget and cosine schedule without repeating warmup')
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--dataset',type=Path,help='Admitted dataset_summary.json; defaults to data/curvenav_expert/dataset_summary.json')
    inputs.add_argument('--manifests',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,help='Actual pilot/continuation steps; fixed full schedule budget remains unchanged')
    parser.add_argument('--lr',type=float,help='Peak LR; default1e-5, or the saved current LR for a new data stage')
    parser.add_argument('--batch-per-rank',type=int,default=2)
    parser.add_argument('--global-batch',type=int,help='Preserve the effective batch across an uneven GPU allocation; batch-per-rank is the capacity limit')
    args=parser.parse_args()
    if args.batch_per_rank<1:parser.error('--batch-per-rank must be positive')
    if args.new_data_stage and not args.resume:parser.error('--new-data-stage requires the complete --resume state')
    proof=None
    if args.resume:
        proof=experiment_io.read(args.resume.with_name('result.json'))
        assert proof['complete'] and proof['training_state_sha256']==sha(args.resume)
    if args.lr is None:args.lr=proof['provenance']['learning_rates'][0] if args.new_data_stage else 1e-5
    if not np.isfinite(args.lr) or args.lr<=0 or (args.updates is not None and args.updates<1):parser.error('Positive finite peak lr and positive actual updates required')
    dataset_rows=None
    if args.dataset or not args.manifests:
        if args.validation_manifests:parser.error('--validation-manifests is only used with explicit --manifests')
        args.dataset=args.dataset or Path('data/curvenav_expert/dataset_summary.json')
        manifests,dataset_rows=dataset_manifests(args.dataset)
        args.manifests=manifests['train'];args.validation_manifests=manifests['validation']
    if args.new_data_stage and not args.validation_manifests:parser.error('A new data stage requires the original validation manifests')
    if os.environ.get('NCCL_PROTO')!='LL':parser.error('Explicit NCCL_PROTO=LL is required; no automatic fallback')
    if args.stop_on_cnn_plateau and (not args.validation_manifests or args.cnn_cache):parser.error('CNN plateau requires live CNN and held-out-state manifests')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);assert world>=1
    global_batch=world*args.batch_per_rank if args.global_batch is None else args.global_batch
    if global_batch<world:parser.error('--global-batch must give each rank at least one sample')
    batch_sizes=rank_batch_sizes(world,args.batch_per_rank,global_batch)
    if max(batch_sizes)>args.batch_per_rank:parser.error('--batch-per-rank is smaller than the required global-batch partition')
    local_batch=batch_sizes[rank]
    torch.set_num_threads(1);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',torch.cuda.current_device())
    dist.init_process_group('nccl');control=dist.new_group(backend='gloo')
    torch.manual_seed(20261027+rank);torch.cuda.manual_seed_all(20261027+rank);np.random.seed(20261027+rank);random.seed(20261027+rank)
    actor=load_bundle(args.bundle,device);assert actor.policy.integration_steps==INTEGRATION_STEPS
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    modules=dict(policy=actor.policy,head=actor.head)
    for module in modules.values():module.eval().requires_grad_(True)
    model=torch.nn.ModuleDict(modules)
    items,goal,memory,data=load_items(args.manifests,actor)
    validation=[];validation_data=None
    if args.validation_manifests:validation,_,_,validation_data=load_items(args.validation_manifests,actor)
    if dataset_rows is not None:
        assert len(items)==dataset_rows['train'] and len(validation)==dataset_rows['validation']
    cnn_cache=None
    if args.cnn_cache:
        from curvenav.data.cnn_cache import load_cache
        cnn_cache=load_cache(actor,items+validation,args.cnn_cache)
    parameters=[p for p in model.parameters() if p.requires_grad]
    goals=case_groups(items,goal);memories=case_groups(items,memory)
    source_keys=[frozenset((item['raw_sha256'],int(frame)) for frame in np.unique(item['memory_plan']['frame_ids'])) for item,*_ in items]
    budget=epoch_budget(goals,memories,world,args.batch_per_rank,actor.config.training,args.lr,global_batch=global_batch)
    if args.new_data_stage:budget.update(warmup_epochs=0,warmup_steps=0)
    optimizer=build_optimizer(model,args.lr,budget['weight_decay'])
    scheduler=build_cosine_schedule(optimizer,budget['total_steps'],budget['warmup_steps'],budget['minimum_factor'])
    ema=ExponentialMovingAverage(model,budget['ema_decay'])
    package=Path(__file__).resolve().parents[1]
    source_names=SOURCE_FILES
    contract=dict(gradient_clipping=False,validation_data=validation_data,monitoring='EMA_full_train_validation_equal_trial_mean_v1',dataset_summary_sha256=sha(args.dataset) if args.dataset else None,stop_on_cnn_plateau=args.stop_on_cnn_plateau,cnn_cache_sha256=sha(args.cnn_cache) if args.cnn_cache else None,world_size=world,effective_batch=global_batch,rank_batch_sizes=batch_sizes,sampling=SAMPLING,objective_version='independent_free_task_admissible_omni_q_v6',
        data=data,sources={name:sha(package/name) for name in source_names},actor_semantics=actor.architecture['actor_semantics'],precision=actor.architecture['precision'],
        integration_steps=INTEGRATION_STEPS,candidates=CANDIDATES,goal_distance_backend=os.environ.get('CURVENAV_GOAL_DISTANCE_BACKEND','scipy'),critic_steps_per_joint=1,
        training_protocol=PROTOCOL,history_execution='six_frame_single_spatial_fusion_v1',history_torch_version=torch.__version__,rank_assignment='incremental_source_union_trial_capacity_v4',generator_autograd='saved_Euler_step_activations_v1',nearest_query_workers=8,nearest_query_support='radius_error_max_sampling_pad',batch_per_rank=args.batch_per_rank,cnn_feature_contract='current_step_full_image_chunk64_geometry4_shared_cell_writes_v4',cnn_execution='torch_compile_default_image_features' if cnn_cache is None else 'provided_frozen_cache',gradient_communication='nccl_cuda_fp32_weighted_sample_mean_v2',nccl_protocol='LL',
        optimizer='production_fused_AdamW_norm_bias_no_decay_betas_0.9_0.95',epoch_budget=budget,row_counts=dict(all=len(items),goal=len(goal),memory=len(memory)),EMA='production_EMA_warmup_all_three_components')
    validation_history=[]
    start=0;raw_parent_sha=sha(args.bundle)
    warmstart=dict(bundle=str(args.bundle.resolve()),sha256=raw_parent_sha,previous_optimizer_restored=False)
    if args.resume:
        state=restore_full(args.resume,model,optimizer,scheduler,ema,contract=contract,raw_sha=raw_parent_sha,rank=rank,device=device,new_data_stage=args.new_data_stage)
        if args.new_data_stage:
            warmstart=dict(bundle=str(args.bundle.resolve()),sha256=raw_parent_sha,previous_optimizer_restored=True,
                data_transition=dict(parent=str(args.resume.resolve()),parent_state_sha256=sha(args.resume),
                    parent_data_steps=state['steps'],parent_EMA_updates=state['ema']['num_updates'],initial_learning_rate=args.lr,
                    parent_scheduler=state['scheduler'],parent_row_counts=state['contract']['row_counts'],
                    schedule='New data epoch0; continue AdamW/EMA/RNG; cosine over the new budget without repeated warmup'))
        else:start=state['steps'];warmstart=state['warmstart'];validation_history=state['validation_history']
    updates=budget['total_steps']-start if args.updates is None else args.updates
    if updates<1 or start+updates>budget['total_steps']:raise ValueError('Actual updates exceed the fixed full schedule budget')
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
        for name in source_names:
            p=args.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((package/name).read_bytes())
    base_config=torch.load(args.bundle,map_location='cpu',mmap=True,weights_only=False)['base_config']
    if cnn_cache is None:
        # Repeated CNN calls before one backward corrupt saved tensors with CUDA graphs on torch 2.7.
        actor.policy.depth_encoder.image_features=torch.compile(actor.policy.depth_encoder.image_features,fullgraph=True,dynamic=False,mode='default')
    actor.policy.trajectory_decoder.project_condition_memory=torch.compile(actor.policy.trajectory_decoder.project_condition_memory,fullgraph=True,dynamic=False,mode='default')
    actor.policy._predict_velocity=torch.compile(actor.policy._predict_velocity,fullgraph=True,dynamic=False,mode='default')
    checkpoint_interval=budget['steps_per_epoch']*actor.config.training.checkpoint_every_epochs
    dist.barrier();torch.cuda.reset_peak_memory_stats(device);logs=[];durations=[];began=time.monotonic()
    for outer in range(updates):
        outer_began=time.monotonic();ids=balanced_batch_indices(goals,memories,outer,start,rank,world,args.batch_per_rank,source_keys,global_batch=global_batch)
        optimizer.zero_grad(set_to_none=True);terms,_=batched_losses(actor,[items[index] for index in ids],cnn_cache=cnn_cache);value=sum(terms.values())*(world*local_batch/global_batch);assert torch.isfinite(value)
        value.backward();grad_norm=average_gradients(parameters,device,world)
        assert torch.isfinite(grad_norm)
        used_rates=[g['lr'] for g in optimizer.param_groups]
        optimizer.step();scheduler.step();ema.update()
        totals=torch.stack([v.detach().double() for v in terms.values()]).cpu()*local_batch;dist.all_reduce(totals,group=control);totals.div_(global_batch)
        elapsed=time.monotonic()-outer_began;durations.append(elapsed)
        if rank==0:
            record=dict(stage_step=start+outer+1,critic_step=ema.num_updates,data_step=start+outer+1,seconds=time.monotonic()-began,
                step_seconds=elapsed,global_samples_per_second=global_batch/elapsed,learning_rates_used=used_rates,grad_norm=float(grad_norm),weights_kind='raw',aggregation='optimization_batch_pre_update',
                **dict(zip(terms,totals.tolist(),strict=True)));logs.append(record);print(json.dumps(record),flush=True)
        if (start+outer+1)%checkpoint_interval==0:
            from curvenav.training.convergence import comparison_losses,cnn_plateau
            if validation:
                with ema.average_parameters(model):
                    metrics=comparison_losses(actor,items,validation,rank,world,device,control,cnn_cache)
                validation_history.append(dict(epoch=(start+outer+1)//budget['steps_per_epoch'],step=start+outer+1,
                    weights_kind='ema',aggregation=SAMPLING,train=metrics['train'],**metrics['validation']))
                if rank==0:print(json.dumps(dict(validation=validation_history[-1],cnn_converged=cnn_plateau(validation_history))),flush=True)
            torch.cuda.synchronize(device)
            epoch_payloads=[None]*world
            stat=dict(rank=rank,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                device_total_bytes=torch.cuda.get_device_properties(device).total_memory,elapsed_seconds=time.monotonic()-began,
                local_samples=(outer+1)*local_batch,local_samples_per_second=(outer+1)*local_batch/(time.monotonic()-began))
            dist.all_gather_object(epoch_payloads,dict(rng=capture_rng(device),performance=stat),group=control)
            if rank==0:
                checkpoint=args.output/'checkpoints'/f"epoch_{(start+outer+1)//budget['steps_per_epoch']:03d}"
                save_checkpoint(checkpoint,actor,model,optimizer,scheduler,ema,contract,warmstart,base_config,start+outer+1,start,
                    args.batch_per_rank,budget,raw_parent_sha,args.resume,epoch_payloads,logs,args.output/'source',create=True,validation_history=validation_history)
            dist.barrier()
            if args.stop_on_cnn_plateau and cnn_plateau(validation_history):break

    updates=outer+1
    torch.cuda.synchronize(device);elapsed=time.monotonic()-began
    perf=dict(rank=rank,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        device_total_bytes=torch.cuda.get_device_properties(device).total_memory,elapsed_seconds=elapsed,local_samples=updates*local_batch,
        local_samples_per_second=updates*local_batch/elapsed,outer_step_seconds=durations)
    payloads=[None]*world;dist.all_gather_object(payloads,dict(rng=capture_rng(device),performance=perf),group=control)
    if rank==0:
        save_checkpoint(args.output,actor,model,optimizer,scheduler,ema,contract,warmstart,base_config,start+updates,start,
            args.batch_per_rank,budget,raw_parent_sha,args.resume,payloads,logs,args.output/'source',create=False,validation_history=validation_history)
    dist.barrier();dist.destroy_process_group(control);dist.destroy_process_group()


if __name__=='__main__':main()
