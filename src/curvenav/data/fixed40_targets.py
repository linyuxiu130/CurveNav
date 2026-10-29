"""Offline task targets for the explicitly admitted forty development cases."""
from collections import deque
import numpy as np
from curvenav.data_generation.geometry import PlanningError
from curvenav.physical import EXTRA_CLEARANCE_M

def branch_membership(points,route,hall,fork):
    points=np.asarray(points);route=np.asarray(route)
    low=np.minimum(route[:-1],route[1:])-hall/2
    high=np.maximum(route[:-1],route[1:])+hall/2
    return (abs(points-np.asarray(fork)).max(-1)>.8)&((points[...,None,:]>=low)&(points[...,None,:]<=high)).all(-1).any(-1)


def branch_at(point,case,indices):
    for index in indices:
        if branch_membership(point,case['deadend_arms'][index]['route'],case['hall'],case['deadend_arms'][index]['route'][0]):return index
    return None


def task_state_from_history(raw,acquisitions):
    """Only prefix certification and already executed views determine task state."""
    task=raw['task'];case=raw['case'];panel=task['panel']
    state=dict(panel=panel,case=case)
    views=np.asarray(raw['acquired_views'])[len(case['prefix_views'])-1:acquisitions,:2]
    if panel=='L':
        state['fork_seen']=task['phase']=='returned' or bool((np.linalg.norm(views,axis=-1)<=.75).any())
    elif panel=='D2':
        state['arm']=task['arm']
        state['confirmed_arms']=raw['task_evidence']['initially_confirmed_arms']
        fork=np.asarray(case['deadend_arms'][task['arm']]['fork'])
        state['fork_seen']=task['stage']=='returned' or bool((np.linalg.norm(views-fork,axis=-1)<=.75).any())
    return state


def task_failure_mask(world_paths,states):
    """The existing fixed40 no-reentry conditions, evaluated along each candidate.

    Starting inside a known leg is legal before reaching its fork. J permits
    closed-branch entry when following the visible point goal.
    """
    invalid=np.zeros(world_paths.shape[:-1],dtype=bool)
    for row,(points,state) in enumerate(zip(world_paths,states,strict=True)):
        case=state['case'];panel=state['panel']
        if panel=='K':
            for route in case['branch_routes'][case['known_complete_branch']]:
                invalid[row]|=branch_membership(points,route,case['hall'],np.asarray(case['start'])[:2])
        elif panel=='L':
            seen=state['fork_seen']|np.maximum.accumulate(np.linalg.norm(points,axis=-1)<=.75,axis=-1)
            known=points@np.asarray(case['known_direction']);other=points@np.asarray(case['other_direction'])
            invalid[row]=seen&(known>.8)&(abs(other)<case['hall']/2)
        elif panel=='D2':
            fork=np.asarray(case['deadend_arms'][state['arm']]['fork'])
            seen=state['fork_seen']|np.maximum.accumulate(np.linalg.norm(points-fork,axis=-1)<=.75,axis=-1)
            for index in state['confirmed_arms']:
                invalid[row]|=branch_membership(points,case['deadend_arms'][index]['route'],case['hall'],case['deadend_arms'][index]['route'][0])
            invalid[row]&=seen
        elif panel!='J':raise ValueError(f'Unknown fixed40 panel: {panel}')
    return invalid


def task_target(task,result,grid,view=None):
    case=task['case'];panel=task['panel']
    if panel=='J':return np.asarray(task['target_world'],dtype=np.float64),dict(kind='actual_point_goal')
    if panel=='K':
        assert task['qualification']['qualified'] and case['qualified']
        name='other_route' if case['expected_open_branch']=='other' else 'side_route'
        return np.asarray(case[name][-1],dtype=np.float64),dict(kind='qualified_knowledge_task',
            expected_open_branch=case['expected_open_branch'],known_complete_branch=case['known_complete_branch'],audit_sha256=task['audit_sha256'])
    if panel=='L':
        audit=result['initial_audit'];assert audit['known_wall_fraction']>=.9 and audit['other_wall_fraction']<.1
        origin=np.asarray(case['recorded_known_route'][0]);direction=np.asarray(case['other_direction'])
        return origin+direction*case['other_length'],dict(kind='qualified_other_branch',initial_audit=audit)
    assert panel=='D2'
    fork=np.asarray(case['deadend_arms'][task['arm']]['fork']);spacing=case['hall']+case['wall']
    origin=np.asarray(case['offset'])+case['wall']+case['hall']/2
    start=tuple(np.rint((fork-origin)/spacing).astype(int));graph={}
    for edge in case['graph_edges']:
        a,b=tuple(edge[:2]),tuple(edge[2:]);graph.setdefault(a,[]).append(b);graph.setdefault(b,[]).append(a)
    assert start in graph and np.allclose(origin+np.asarray(start)*spacing,fork,rtol=0,atol=1e-8)
    distance={start:0};queue=deque([start])
    while queue:
        node=queue.popleft()
        for other in graph[node]:
            if other not in distance:distance[other]=distance[node]+1;queue.append(other)
    position=np.asarray(case['start'][:2] if view is None else view[:2])
    current=min(graph,key=lambda node:np.linalg.norm(origin+np.asarray(node)*spacing-position))
    from_current={current:0};queue=deque([current])
    while queue:
        node=queue.popleft()
        for other in graph[node]:
            if other not in from_current:from_current[other]=from_current[node]+1;queue.append(other)
    excluded=sorted(set(result['initially_confirmed_arms']+[task['arm']]))
    candidates=[]
    for node,steps in distance.items():
        point=origin+np.asarray(node)*spacing
        if (np.linalg.norm(point-fork)>=task['scorer']['exit_distance_at_least_m']
                and branch_at(point,case,excluded) is None and grid.clearance(point[None])[0]>=EXTRA_CLEARANCE_M):
            candidates.append((from_current[node],float(np.linalg.norm(point-position)),tuple(point),point))
    if not candidates:raise PlanningError('no source-safe exit outside source/confirmed arms')
    chosen=min(candidates,key=lambda candidate:candidate[:3]);target=chosen[3]
    return target,dict(kind='canonical_source_safe_legal_exit',fork=fork.tolist(),
        excluded_source_or_confirmed_arms=excluded,graph_steps=chosen[0],
        rule='Nearest graph-distance source-safe exit from current state; Euclidean then lexicographic tie; endpoint beyond original1.5m criterion. No runtime rule or input.')
