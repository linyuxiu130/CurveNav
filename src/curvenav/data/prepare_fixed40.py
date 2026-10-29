"""Collect physically executed teacher trajectories and causal six-frame supervision."""
from pathlib import Path
from dataclasses import replace
import argparse,json,pickle,hashlib,copy
import numpy as np
import torch
from scipy.spatial import cKDTree
from curvenav.data import bspline_replay as replay
from curvenav.data.bspline_teachers import source_geometry_teacher
from curvenav.data.fixed40_targets import task_target,branch_at,task_state_from_history
from curvenav.data.goal_distance import GoalDistanceQuery
from curvenav.data.observed_bspline_routes import ObservedBSplineRoutes,ObservedHitHistory
from curvenav.data.observed_geometry import target_from_observations,GEOMETRY_LABEL_CONTRACT
from curvenav.data.six_frame_memory import MEMORY_CONTRACT
from curvenav.data_generation.geometry import Grid
from curvenav.deployment.bspline_bundle import load_bundle
from curvenav.evaluation import bspline_scenarios as scenario,bspline_execution as execution
from curvenav.evaluation.bspline40 import new_route_progress,snapshot_sources,local_knowledge_audit
from curvenav.training.critic import RouteUtilityTeacher

DATA_CONTRACT='omni_teacher40_continuous_geometry_v3'


def selected_states(count,partition):
    """Keep every captured state; the independent episode owns its split."""
    return [(index,partition) for index in range(count)]


def reached(task,env,reference,fork_seen,initial_known,initial_walls):
    panel=task['panel'];case=task['case'];view=env.view
    if panel=='J':return np.linalg.norm(view[:2]-np.asarray(task['target_world']))<=.35
    if panel=='K':
        progress,inside=new_route_progress(view,case)
        novel=int((env.knowledge.known&~initial_known).sum())+int((env.knowledge.wall_seen&~initial_walls).sum())
        return progress>=1.5 and inside and novel>0
    if panel=='L':
        origin=np.asarray(case['recorded_known_route'][0]);delta=view[:2]-origin
        return delta@np.asarray(case['other_direction'])>=1.5 and abs(delta@np.asarray(case['known_direction']))<case['hall']/2
    fork=np.asarray(case['deadend_arms'][task['arm']]['fork'])
    excluded=sorted(set(reference['initially_confirmed_arms']+[task['arm']]))
    return fork_seen and np.linalg.norm(view[:2]-fork)>=1.5 and branch_at(view[:2],case,excluded) is None


@torch.no_grad()
def collect_case(actor,task,output,*,partition,memory_seed=0,initial_offset_body_m=None):
    if partition not in ('train','validation'):raise ValueError('episode partition must be train or validation')
    if initial_offset_body_m is not None:
        initial_offset_body_m=np.asarray(initial_offset_body_m,dtype=np.float64)
        if initial_offset_body_m.shape!=(2,) or not np.isfinite(initial_offset_body_m).all():
            raise ValueError('initial offset must contain two finite body-frame metres')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_snapshot=snapshot_sources(output,scenario.RUNTIME_ASSETS_PATH)
    case=task['case'];trial=task['trial_id'];device=next(actor.policy.parameters()).device
    factory=actor.new_memory;actor=copy.copy(actor)
    actor.new_memory=lambda:factory(seed=memory_seed)
    capture_spec=dict(memory_seed=memory_seed,initial_offset_body_m=initial_offset_body_m.tolist() if initial_offset_body_m is not None else None)
    env=scenario.Environment(actor,case,gpu=0,timestamp_origin=0. if task['panel']=='J' else .5)
    teacher=ObservedBSplineRoutes(env.k,env.camera);history=ObservedHitHistory(env.k,env.camera)
    views=[np.asarray(v,dtype=np.float64).copy() for v in case['prefix_views']]
    depths=[env.cache[tuple(np.round(v,7))].copy() for v in views]
    origin=0. if task['panel']=='J' else .5;times=(origin+np.arange(len(views))*.5).tolist()
    for image,view in zip(depths,views,strict=True):teacher.update(image,view,executed=False);history.update(image,view)
    reference=dict(trial_id=trial)
    if task['panel']=='L':reference['initial_audit']=local_knowledge_audit(env,case)
    if task['panel']=='D2':
        audits={i:scenario.audit_branch(case,i,np.asarray(views),env.cache,env.k,env.camera) for i in range(len(case['deadend_arms']))}
        reference.update(prefix_audit=audits,initially_confirmed_arms=[i for i,a in audits.items() if a['confirmed']])
    reference_path=output/'initial_reference.json'
    reference_path.write_text(json.dumps(reference,indent=2)+'\n')
    grid_path=output/'source_grid.npz';query=replay.source_grid(case,grid_path,actor.head.radius)
    grid=replace(Grid.load(grid_path),minimum_clearance_m=0.);source_distance=GoalDistanceQuery(query)
    utility=RouteUtilityTeacher(query,actor.policy.planning_horizon_m,actor.head.radius)
    prelude=None
    if initial_offset_body_m is not None:
        pose=scenario.body_pose(env.view)
        target_offset=pose[:2,3]+initial_offset_body_m@pose[:2,:2].T
        points=(history.records()['world'][:,:2]-pose[:2,3])@pose[:2,:2]
        qualified,rejection=source_geometry_teacher(actor,env.view,pose,target_offset,cKDTree(points),history.error_bound,case['boxes'],query,grid)
        if qualified is None:raise RuntimeError(f'{trial}: independent-start qualification failed: {rejection}')
        local,_=actor.policy.curve_codec.decode_values(qualified['teacher_values'][None].to(next(actor.policy.parameters()).device))
        prelude=execution.advance_path(env,local[0].cpu().numpy())
        if prelude['status']=='collision':raise RuntimeError(f'{trial}: independent-start execution collided')
        prelude['certificate']=qualified['teacher_certificate']
        image=env.cache[tuple(np.round(env.view,7))].copy();views.append(env.view.copy());depths.append(image);times.append(env.clock)
        teacher.update(image,env.view,executed=False);history.update(image,env.view)
    target,identity=task_target(task,reference,grid,view=env.view)
    initial_distance=float(np.linalg.norm(target-env.view[:2]));initial_known=env.knowledge.known.copy();initial_walls=env.knowledge.wall_seen.copy()
    prefix=len(views);decisions=[];trace=[];world_path=None;last_label=None;last_label_travel=0.;fork_seen=False;reached_once=False;status='budget_exhausted'
    try:
        for step in range(task['budget']):
            env.memory.write()
            pose=scenario.body_pose(env.view)
            if task['panel']=='D2':
                fork=np.asarray(identity['fork']);fork_seen |= bool(np.linalg.norm(env.view[:2]-fork)<=.75)
            angle=0. if last_label is None else abs(np.arctan2(np.sin(env.view[2]-last_label[2]),np.cos(env.view[2]-last_label[2])))
            # Sample by actual travelled distance/heading, never by repeated stalled clock ticks.
            continuing=task['panel']=='D2' and reached_once
            capture=last_label is None or env.path-last_label_travel>=.2 or angle>=.35 or (
                continuing and (decisions[-1]['teacher_phase']=='task_route' or
                    np.linalg.norm(target-env.view[:2])<=execution.REFERENCE_SPEED_MPS*execution.DT_POLICY))
            if capture:
                if continuing:
                    frontier,detail=teacher.source_frontier_target(env.view,source_distance)
                    if frontier is None:
                        raise RuntimeError(f'{trial}: no certified continuation frontier: {detail}')
                    target=frontier
                points=(history.records()['world'][:,:2]-pose[:2,3])@pose[:2,:2]
                if not len(points):raise ValueError('teacher state lacks actual observed hits')
                tree=cKDTree(points);required=np.asarray(identity['fork']) if task['panel']=='D2' and not fork_seen else None
                qualified,rejection=source_geometry_teacher(actor,env.view,pose,target,tree,history.error_bound,
                    case['boxes'],query,grid,required,.75 if required is not None else None)
                if qualified is None:
                    (output/'rejection.json').write_text(json.dumps(dict(step=step,view=env.view.tolist(),
                        target_world=target.tolist(),reason=rejection),indent=2)+'\n')
                    raise RuntimeError(f'{trial}: teacher qualification failed at step {step}; inspect rejection.json')
                local,_=actor.policy.curve_codec.decode_values(qualified['teacher_values'][None].to(next(actor.policy.parameters()).device))
                world_path=local[0].cpu().numpy()@pose[:2,:2].T+pose[:2,3]
                present=task['panel']=='J';goal=((target-pose[:2,3])@pose[:2,:2]).astype(np.float32) if present else np.zeros(2,np.float32)
                field=target_from_observations(teacher,pose,env.k,env.camera,actor.policy.depth_encoder.metric_projector,tree,history.error_bound)
                task_history=dict(task=task,case=case,acquired_views=views)
                if task['panel']=='D2':task_history['task_evidence']=dict(initially_confirmed_arms=reference['initially_confirmed_arms'])
                label=utility(local[None],dict(task_states=[task_state_from_history(task_history,len(views))],
                    source_grid_index=torch.zeros(1,dtype=torch.long,device=device),
                    source_origin_xy=torch.as_tensor(env.view[:2],dtype=torch.float64,device=device)[None],
                    source_yaw_rad=torch.tensor([env.view[2]],dtype=torch.float64,device=device),
                    point_goal=torch.as_tensor(goal,device=device)[None],goal_present=torch.tensor([present],device=device),
                    observed_field=field[None].to(device)))
                if label.failed.any():raise RuntimeError(f'{trial}: expert contradicts original physical/task failure labels at step {step}')
                qualified['teacher_checks']['task_admissible']=True
                condition=actor.encode(env.memory,goal,present);bank=actor.native_bank(condition)
                row=dict(step=step,view=env.view.copy(),clock=env.clock,acquisitions=len(views),goal=torch.from_numpy(goal),goal_present=present,
                    teacher_values=qualified['teacher_values'],teacher_checks=qualified['teacher_checks'],teacher_certificate=qualified['teacher_certificate'],
                    target_field=field,point_error_m=history.error_bound,tree=tree,
                    teacher_phase='continued_exploration' if continuing else 'task_route',
                    critic_replay_paths=bank['paths'][0].cpu(),input_frame_ids=list(env.memory.selection['indices']))
                if device.type=='cuda':row['plan']=env.memory.read_plan()
                decisions.append(row)
                last_label=env.view.copy();last_label_travel=env.path
                print(json.dumps(dict(trial_id=trial,decision=len(decisions),step=step,travel_m=env.path)),flush=True)
            local=(world_path-pose[:2,3])@pose[:2,:2]
            before=env.view.copy();executed=execution.advance_path(env,local)
            trace.append(dict(step=step,view=before.tolist(),next_view=env.view.tolist(),status=executed['status'],margin=executed['margin_m']))
            if executed['status']=='collision':raise RuntimeError(f'{trial}: qualified teacher execution collided at step {step}')
            image=env.cache[tuple(np.round(env.view,7))].copy();views.append(env.view.copy());depths.append(image);times.append(env.clock)
            teacher.update(image,env.view,executed=False);history.update(image,env.view)
            if task['panel']=='D2' and fork_seen and branch_at(env.view[:2],case,reference['initially_confirmed_arms']) is not None:
                raise RuntimeError(f'{trial}: teacher reentered a confirmed branch after reaching the fork')
            if reached(task,env,reference,fork_seen,initial_known,initial_walls):
                reached_once=True;status='task_reached'
                if task['panel']!='D2':break
    finally:env.sim.close()
    actual=np.asarray(views[prefix-1:]);span=float(np.linalg.norm(actual[:,:2]-actual[:1,:2],axis=1).max())
    full_budget=task['panel']!='D2' or len(trace)==task['budget']
    quality=dict(status=status,decisions=len(decisions),planning_span_m=span,travel_m=env.path,
        executed_steps=len(trace),required_steps=task['budget'] if task['panel']=='D2' else None,
        initial_target_distance_m=initial_distance,passed=reached_once and full_budget and bool(decisions))
    (output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    if not quality['passed']:raise RuntimeError(f'{trial}: execution/coverage admission failed: {quality}')
    selected=selected_states(len(decisions),partition)
    raw_path=output/'raw.pt';raw=dict(depths=np.asarray(depths),acquired_views=np.asarray(views),timestamps=np.asarray(times),
        camera_intrinsics=np.repeat(env.k[None],len(views),axis=0),camera_to_body=np.repeat(env.camera[None],len(views),axis=0),
        body_to_world=np.stack([scenario.body_pose(v) for v in views]),memory_writes=env.memory.writes,case=case,task=task,input_contract=MEMORY_CONTRACT,
        task_evidence=dict(initially_confirmed_arms=reference['initially_confirmed_arms'],
            reference_path=str(Path(reference_path).resolve()),reference_sha256=replay.sha(reference_path)) if task['panel']=='D2' else {},
        provenance=dict(source_kind=DATA_CONTRACT,split='development',partition=partition,trial_id=trial,driver=execution.EXECUTOR,capture_bundle_sha256=actor.provenance['bundle_sha256']),
        capture_spec=capture_spec,collection_prelude=prelude)
    torch.save(raw,raw_path);raw_sha=replay.sha(raw_path)
    result_path=output/'result.json';result_path.write_text(json.dumps(dict(complete=True,trial_id=trial,panel=task['panel'],quality=quality,trace=trace,
        raw_path=str(raw_path.resolve()),raw_sha256=raw_sha),indent=2)+'\n')
    bank_path=output/'candidates.npz';np.savez_compressed(bank_path,paths=np.stack([d['critic_replay_paths'].numpy() for d in decisions]))
    records=[]
    for index,partition in selected:
        row=decisions[index];tree=row.pop('tree')
        if device.type=='cuda':
            plan=row.pop('plan');plan['raw_sha256']=raw_sha
            plan_path=output/f'memory_{index:04d}.pt';torch.save(plan,plan_path)
            row.update(memory_snapshot_path=str(plan_path.resolve()),memory_plan_sha256=replay.sha(plan_path))
        tree_path=output/f'hits_{index:04d}.pkl'
        with tree_path.open('wb') as stream:pickle.dump(dict(tree=tree),stream,protocol=pickle.HIGHEST_PROTOCOL)
        row.update(geometry_label_contract=GEOMETRY_LABEL_CONTRACT,label_schema='qualified_only',trial_id=trial,no_goal_teacher=None,input_contract=MEMORY_CONTRACT,raw_path=str(raw_path.resolve()),raw_sha256=raw_sha,
            index_path=str(tree_path.resolve()),index_sha256=replay.sha(tree_path),
            source_grid=str(grid_path.resolve()),source_grid_sha256=replay.sha(grid_path),source_grid_frame='canonical_world_xy',
            critic_replay_provenance=dict(raw_sha256=raw_sha,decision_step=row['step'],acquisitions=row['acquisitions'],view_float64_hex=np.asarray(row['view'],dtype=np.float64).tobytes().hex(),
                native_paths_float32_sha256=hashlib.sha256(row['critic_replay_paths'].numpy().tobytes()).hexdigest(),result_path=str(result_path.resolve()),result_sha256=replay.sha(result_path),
                candidates_path=str(bank_path.resolve()),candidates_sha256=replay.sha(bank_path)))
        path=output/f'item_{index:04d}.pt';torch.save(row,path)
        records.append(dict(path=str(path.resolve()),sha256=replay.sha(path),partition=partition,goal_present=row['goal_present'],step=row['step']))
    manifest=dict(geometry_label_contract=GEOMETRY_LABEL_CONTRACT,complete=True,trial_id=trial,split='development',partition=partition,source_kind=DATA_CONTRACT,input_contract=MEMORY_CONTRACT,
        geometry_input_contract='cuda_native_projection_v1' if device.type=='cuda' else 'cpu_collection_pending_native_replay_v1',training_ready=device.type=='cuda',
        items=records,quality=quality,source_snapshot=source_snapshot,capture_spec=capture_spec,
        scope='Independent physically executed episode; all captured states share its declared partition')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(complete=True,trial_id=trial,quality=quality,rows=len(records))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--trials',nargs='+',help='Task IDs from --cases; collect all when omitted')
    parser.add_argument('--partition',choices=('train','validation'),required=True)
    parser.add_argument('--memory-seed',type=int,default=0)
    parser.add_argument('--initial-offset-body-m',type=float,nargs=2)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-assets',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=True)
    scenario.configure(args.runtime_assets)
    actor=load_bundle(args.bundle,'cuda:0');actor.provenance['bundle_sha256']=replay.sha(args.bundle);tasks={r['trial_id']:r for r in json.loads(args.cases.read_text())['cases']}
    for trial in args.trials if args.trials is not None else tasks:
        collect_case(actor,tasks[trial],args.output/trial,partition=args.partition,
            memory_seed=args.memory_seed,initial_offset_body_m=args.initial_offset_body_m)

if __name__=='__main__':main()
