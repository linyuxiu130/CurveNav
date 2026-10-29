"""Ideal omnidirectional geometric rollout; physical MPC adaptation is separate.

Translate along the chosen curve while holding body yaw. The fixed sensor clock
and nominal travel rate define the evaluation rollout, not learned path limits.
"""
import numpy as np
from curvenav.evaluation import bspline_scenarios as scenario

DT_POLICY=.5
REFERENCE_SPEED_MPS=.5
EXECUTOR='ideal_omni_reference_yaw_hold_v1'


def advance_path(env,path_body):
    start=env.view.copy();path=np.asarray(path_body,dtype=np.float64)
    segments=np.linalg.norm(np.diff(path,axis=0),axis=-1)
    path=path[np.r_[True,segments>0]]
    arc=np.r_[0.,np.linalg.norm(np.diff(path,axis=0),axis=-1).cumsum()]
    progress=0.
    if len(path)>1:
        delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=-1)
        fraction=np.clip((-path[:-1]*delta).sum(-1)/length**2,0.,1.)
        nearest=np.argmin(np.linalg.norm(path[:-1]+fraction[:,None]*delta,axis=-1))
        progress=float(arc[nearest]+fraction[nearest]*length[nearest])
    end=min(float(arc[-1]),progress+REFERENCE_SPEED_MPS*DT_POLICY)
    interior=path[(arc>progress)&(arc<end)]
    endpoint=np.array([np.interp(end,arc,path[:,axis]) for axis in range(2)])
    points=np.vstack((np.zeros(2),interior,endpoint))
    c,s=np.cos(start[2]),np.sin(start[2]);rotation=np.array([[c,-s],[s,c]])
    world=points@rotation.T+start[:2]
    margin=scenario.margin(world,np.asarray(env.case['boxes']),tolerance=0.)
    sweep=np.column_stack((world,np.full(len(world),start[2])))
    detail=dict(executed_sweep=sweep.tolist(),sweep_margin_m=margin,reference_arc_m=end-progress)
    if margin<0:
        detail['attempted_view']=sweep[-1].tolist()
        return dict(status='collision',margin_m=margin,path_m=0.,substeps=[detail])
    travel=float(np.linalg.norm(np.diff(world,axis=0),axis=-1).sum())
    env.path+=travel;env.write(sweep[-1]);detail['view']=env.view.tolist()
    return dict(status='running',margin_m=margin,path_m=travel,substeps=[detail])
