"""Native14 joint math with step-batched history CNN and recomputed Euler activations."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields,replace
import torch
from torch.nn import functional as F
from curvenav.types import ConditionFeatures
from curvenav.deployment.bspline_bundle import INTEGRATION_STEPS
from curvenav.models.policy import repeat_condition
from curvenav.models.exploration import GOAL_DROPOUT_PROBABILITY
from curvenav.data.bspline_replay import dense_curve,SAMPLES
from curvenav.encoders.temporal_geometry import geometry_reconstruction_loss


from curvenav.deployment.streaming_memory import fuse_sources_batch,encode_memories,source_groups
import numpy as np


def live_condition(actor,item,raw,*,feature_provider):
    maps={(item['raw_sha256'],int(index)):feature_provider(int(index)) for index,_ in source_groups(item['memory_plan'])}
    return live_conditions(actor,[(item,raw)],maps)


def live_conditions(actor,records,feature_maps):
    """Fuse this step's writes together, then bound geometry-network activations."""
    maps=[];widths=[];plans=[r[0]['memory_plan'] for r in records]
    for item,raw,*_ in records:
        ids=[int(index) for index,_ in source_groups(item['memory_plan'])]
        maps.append({index:feature_maps[item['raw_sha256'],index] for index in ids})
        widths.append({index:raw['depths'][index].shape[-1] for index in ids})
    visuals=fuse_sources_batch(actor.policy,plans,maps,widths);conditions=[];fields=[];logits=[]
    for start in range(0,len(records),4):
        chunk=records[start:start+4]
        condition,(field,logit)=encode_memories(actor.policy,actor.head,
            plans[start:start+4],visuals[start:start+4],[r[0]['goal'] for r in chunk],
            [r[0]['goal_present'] for r in chunk])
        conditions.append(condition);fields.append(field);logits.append(logit)
    return cat_conditions(conditions),(torch.cat(fields),torch.cat(logits))


def prepare_critic_inputs(actor, item, goal_present, paths, utility):
    device = paths.device
    if not item['q_supervision_available']:
        return None
    # Cache only immutable replay inputs/labels; current generated paths stay fresh.
    cache=item.setdefault('_critic_inputs',{})
    if device not in cache:
        fixed=dict(task_states=[item['task_state']],source_grid_index=torch.zeros(1,dtype=torch.long,device=device),
            source_origin_xy=torch.as_tensor(item['view'][:2],device=device,dtype=torch.float64)[None],
            source_yaw_rad=torch.tensor([item['view'][2]],device=device,dtype=torch.float64),
            point_goal=item['goal'][None].to(device),observed_field=item['target_field'][None].to(device))
        with torch.no_grad():
            if len(item['q_expert_values']):expert,_=actor.policy.curve_codec.decode_values(item['q_expert_values'].to(device))
            else:expert=paths.new_empty((0,*paths.shape[1:]))
        recorded=item['critic_replay_paths'].to(device).detach()
        cache[device]=(fixed,expert.detach(),recorded,utility(recorded[None],dict(fixed,goal_present=goal_present[:,0].bool())).score.detach())
    fixed,expert,recorded,recorded_targets=cache[device]
    qbatch=dict(fixed,goal_present=goal_present[:,0].bool())
    paths=paths.detach();expert_count=len(expert)
    if expert_count:paths=torch.cat((expert,paths))
    return dict(paths=paths.detach(), qbatch=qbatch, utility=utility, expert_count=expert_count, recorded=recorded, recorded_targets=recorded_targets)


def label_critic_inputs(inputs):
    """Trace candidates sharing one source oracle in a single batched query."""
    groups=defaultdict(list)
    for prepared in inputs:groups[prepared['utility']].append(prepared)
    for utility,group in groups.items():
        experts=max(p['expert_count'] for p in group)
        paths=torch.stack([torch.cat((F.pad(p['paths'][:p['expert_count']],(0,0,0,0,0,experts-p['expert_count'])),
            p['paths'][p['expert_count']:])) for p in group])
        batch={key:torch.cat([p['qbatch'][key] for p in group]) for key in group[0]['qbatch'] if key!='task_states'}
        batch['task_states']=[state for p in group for state in p['qbatch']['task_states']]
        targets=utility(paths,batch).score.detach()
        for row,p in enumerate(group):
            p['targets']=torch.cat((targets[row,:p['expert_count']],targets[row,experts:]))[None]

def cat_conditions(conditions):
    return ConditionFeatures(**{f.name:torch.cat([getattr(c,f.name) for c in conditions],0) for f in fields(ConditionFeatures)})


def select_condition(condition,indices):
    ids=torch.as_tensor(indices,device=condition.tokens.device,dtype=torch.long)
    return ConditionFeatures(**{f.name:getattr(condition,f.name).index_select(0,ids) for f in fields(ConditionFeatures)})


def batched_cfm(actor,conditions,items):
    # One bridge per teacher; supervise the two actual Euler query times (1, 1/2).
    dropped_goals=[];indices=[];states=[];targets=[];times=[];count=INTEGRATION_STEPS;device=conditions.tokens.device
    time=torch.arange(count,0,-1,device=device,dtype=torch.float32)/count
    for index,item in enumerate(items):
        if item['teacher_values'] is None:continue
        teacher=item['teacher_values'];dropped=False
        if item['goal_present'] and item['no_goal_teacher'] is not None:
            if bool((torch.rand_like(conditions.goal_present[index:index+1])<GOAL_DROPOUT_PROBABILITY).item()):
                teacher=item['no_goal_teacher']['teacher_values']
                dropped=True
        cache=item.setdefault('_cfm_targets',{});key=(device,dropped)
        if key not in cache:
            with torch.no_grad():cache[key]=actor.policy.curve_codec.coordinates_from_values(teacher.to(device)[None])
        clean=cache[key]
        source=torch.randn_like(clean)
        states.append((1-time[:,None])*clean+time[:,None]*source)
        targets.append((source-clean).expand(count,-1));times.append(time)
        dropped_goals.append(dropped);indices.append(index)
    result=conditions.tokens.new_zeros(len(items))
    if indices:
        padding=len(items)-len(indices)
        condition=conditions if not padding else select_condition(conditions,indices+[indices[0]]*padding)
        if any(dropped_goals):
            dropped=torch.tensor(dropped_goals+dropped_goals[:1]*padding,device=device)[:,None]
            condition=replace(condition,goal_present=condition.goal_present.masked_fill(dropped,0),terminal_goal=condition.terminal_goal.masked_fill(dropped,0))
        # Keep the compiled batch fixed; padded queries contribute no loss or RNG draws.
        if padding:states.append(states[0].repeat(padding,1));times.append(time.repeat(padding))
        memory=actor.policy.trajectory_decoder.project_condition_memory(condition)
        velocity=actor.policy._predict_velocity(torch.cat(states),torch.cat(times),repeat_condition(condition,count),memory)
        velocity=velocity[:len(indices)*count]
        errors=(velocity-torch.cat(targets)).square().reshape(len(indices),count,-1).mean((1,2))
        result=result.index_copy(0,torch.tensor(indices,device=device),errors)
    return result.mean()


def generated_trajectory_loss(actor, paths, items):
    """Every final candidate matches a qualified teacher for the same observable task."""
    losses=[paths.new_zeros(()) for _ in items];groups=defaultdict(list);targets_by_item={}
    for index,item in enumerate(items):
        teachers = item['trajectory_teacher_values']
        if not len(teachers):continue
        cache=item.setdefault('_trajectory_targets',{})
        if paths.device not in cache:
            with torch.no_grad():
                targets,_=actor.policy.curve_codec.decode_values(teachers.to(paths.device))
                cache[paths.device]=(targets,torch.diff(targets,dim=-2)*(targets.shape[-2]-1))
        targets_by_item[index]=cache[paths.device];groups[len(teachers)].append(index)
    for indices in groups.values():
        predicted=paths[indices]
        targets=torch.stack([targets_by_item[i][0] for i in indices])
        target_derivative=torch.stack([targets_by_item[i][1] for i in indices])
        # Choose a teacher for every candidate, not one good candidate for the entire bank.
        error = ((predicted[:, :, None]-targets[:,None]) / actor.policy.planning_horizon_m).square().mean((-1, -2))
        derivative=torch.diff(predicted,dim=-2)*(predicted.shape[-2]-1)
        error=error+((derivative[:,:,None]-target_derivative[:,None])/actor.policy.planning_horizon_m).square().mean((-1,-2))
        for index,value in zip(indices,error.amin(-1).mean(-1),strict=True):losses[index]=value
    return torch.stack(losses).mean()


def batched_critic(actor,condition,items,paths,utilities):
    inputs=[]
    for index,item in enumerate(items):
        prepared=prepare_critic_inputs(actor,item,condition.goal_present[index:index+1],paths[index],utilities[index])
        inputs.append(prepared)
    available=[p for p in inputs if p is not None]
    if not available:return condition.tokens.new_zeros(())
    label_critic_inputs(available)
    experts=max(p['expert_count'] for p in available);recorded=max(len(p['recorded']) for p in available)
    native=paths.shape[1];total=experts+native+recorded
    banks=[];targets=[];weights=[]
    for p in inputs:
        weight=np.zeros(total,dtype=np.float32)
        if p is None:
            banks.append(paths.new_zeros(total,*paths.shape[-2:]));targets.append(paths.new_zeros(total))
        else:
            count=p['expert_count'];replay=len(p['recorded']);factor=.75 if count else 1.
            banks.append(torch.cat((F.pad(p['paths'][:count],(0,0,0,0,0,experts-count)),p['paths'][count:],
                F.pad(p['recorded'],(0,0,0,0,0,recorded-replay)))))
            targets.append(torch.cat((F.pad(p['targets'][0,:count],(0,experts-count)),p['targets'][0,count:],
                F.pad(p['recorded_targets'][0],(0,recorded-replay)))))
            if count:weight[:count]=.25/count
            weight[experts:experts+native]=factor/(2*native)
            weight[experts+native:experts+native+replay]=factor/(2*replay)
        weights.append(weight)
    goals=torch.stack([item['goal'].to(paths.device) for item in items])
    scores=actor.policy.score_candidate_paths(condition,goals,torch.stack(banks))
    errors=F.smooth_l1_loss(scores.float(),torch.stack(targets),reduction='none')
    return (errors*torch.as_tensor(np.stack(weights),device=paths.device)).sum(-1).mean()


def _nearest_sites(queries,records,bounds):
    sites=[]
    for i,(_,_,tree,_) in enumerate(records):
        nearest=tree.query(queries[i],workers=8,distance_upper_bound=float(bounds[i]))[1]
        # Every hit is outside GF support when no neighbor is found; any real hit gives zero loss/gradient.
        nearest[nearest==tree.n]=0
        sites.append(tree.data[nearest])
    return np.stack(sites)


def _risk_from_sites(actor,values,paths,pad,sites,records):
    batch,count=values.shape[:2]
    sites=torch.as_tensor(sites,device=paths.device)
    errors=paths.new_tensor([item['point_error_m'] for item,*_ in records])[:,None,None]
    margin=(paths-sites).norm(dim=-1).reshape(batch,count,SAMPLES)-errors-actor.head.radius
    deficit=((pad.reshape(batch,count,1)-margin)/actor.head.radius).clamp_min(0)
    return (4*torch.where(deficit<=1,deficit.square(),2*deficit-1).amax(-1)).mean(-1).mean()


def batched_losses(actor,records,*,cnn_cache=None):
    from curvenav.training.bspline_image_batch import step_feature_maps
    feature_maps=step_feature_maps(actor,records,cache=cnn_cache)
    items=[r[0] for r in records]
    unique={id(r[0]):r for r in records}
    condition,(field,logits)=live_conditions(actor,list(unique.values()),feature_maps)
    if len(unique)!=len(items):
        positions={key:index for index,key in enumerate(unique)}
        ids=torch.tensor([positions[id(item)] for item in items],device=condition.tokens.device)
        condition=select_condition(condition,ids);field=field.index_select(0,ids);logits=logits.index_select(0,ids)
    cfm=batched_cfm(actor,condition,items)
    bank=actor.native_bank(condition)
    trajectory=generated_trajectory_loss(actor,bank['paths'],items)
    batch=bank['values'].shape[0]
    paths,pad=dense_curve(bank['values'].flatten(0,1))
    paths=paths.reshape(batch,-1,2).double()
    queries=paths.detach().cpu().numpy()
    bounds=pad.reshape(batch,-1).amax(-1).detach().double().cpu().numpy()
    bounds+=np.asarray([item['point_error_m']+actor.head.radius for item,*_ in records])
    # Search conservatively across CPU/GPU FP64 rounding; the actual GF formula is unchanged.
    bounds+=8*np.finfo(np.float64).eps*np.maximum(1,bounds)
    # The nearest-hit search is CPU-only; run it while the GPU computes Q and LH.
    with ThreadPoolExecutor(max_workers=1) as executor:
        nearest=executor.submit(_nearest_sites,queries,records,bounds)
        lh=geometry_reconstruction_loss(field,logits,torch.stack([item['target_field'] for item in items]))
        lq=batched_critic(actor,condition,items,bank['paths'],[r[3] for r in records])
        gf=_risk_from_sites(actor,bank['values'],paths,pad,nearest.result(),records)

    return dict(CFM=cfm,TRAJ=trajectory,GF=gf,LH=lh,LQ=lq),bank
