"""Six timestamp-sampled observations write one bounded world-cell memory."""
from collections import OrderedDict
from dataclasses import dataclass
import copy
import numpy as np
import torch
from curvenav.data.history import validate_transform
from curvenav.data.depth import PinholeIntrinsics
from curvenav.data.obstacle_memory import VOXEL_M, volumes_observed_clear
from curvenav.types import PolicyCondition

MEMORY_CONTRACT = 'six_frame_6s_log_strata_single_world_memory_v1'
FRAMES, HISTORY_SECONDS = 6, 6.
READ_RADIUS_M, MAX_READ_CELLS = 12., 8192


def sample_frames(timestamps, rng):
    """Current + one actual observation per time stratum, probability proportional to 1/age."""
    times=np.asarray(timestamps,dtype=np.float64)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('observation timestamps must be finite and strictly increasing')
    ages=times[-1]-times;edges=np.linspace(0,HISTORY_SECONDS,FRAMES)
    selected=[];valid=[]
    for low,high in reversed(list(zip(edges[:-1],edges[1:]))):
        candidates=np.flatnonzero((ages>low)&(ages<=high))
        if len(candidates):
            weights=1/ages[candidates];weights/=weights.sum()
            selected.append(int(rng.choice(candidates,p=weights)));valid.append(True)
        else:
            selected.append(len(times)-1);valid.append(False)
    return np.array(selected+[len(times)-1]),np.array(valid+[True])


@dataclass
class Observation:
    index: int
    time: float
    depth: np.ndarray
    pose: np.ndarray
    intrinsic: np.ndarray
    camera: np.ndarray


class SixFrameMemory:
    """update buffers raw observations; write performs one sampled fusion event.

    World cells store fused descriptors plus their exact immutable source plan.
    Repeated reads never write. Overlapping windows replace a cell, never add
    counts repeatedly; newer visible free evidence prevents old-frame resurrection.
    """
    def __init__(self,policy,*,capacity=32768,seed=0):
        if capacity<1:raise ValueError('memory capacity must be positive')
        self.policy=policy;self.device=next(policy.parameters()).device
        self.projector=policy.depth_encoder.metric_projector
        self.maximum=self.projector.max_depth_m
        self.resolution=2*self.projector.planning_horizon_m/63
        self.capacity=capacity;self.rng=np.random.default_rng(seed)
        self.observations=OrderedDict();self.cells=OrderedDict();self.cleared={}
        self.visual={};self.maps={};self.projections={};self.writes=[];self.seen=set()
        self.sequence=0;self.last_write=-1;self.clock=-float('inf')
        self.selection=None

    def update(self,depth,intrinsic,camera_to_body,body_to_world,timestamp):
        depth=np.asarray(depth,dtype=np.float32)
        if depth.ndim!=2 or not np.isfinite(depth).all() or (depth<0).any() or (depth>1).any():
            raise ValueError('depth must be finite normalized optical Z in [0,1]')
        PinholeIntrinsics.from_matrix(intrinsic,width=depth.shape[1],height=depth.shape[0])
        validate_transform(camera_to_body);validate_transform(body_to_world)
        if not np.allclose(body_to_world[:3,2],[0,0,1],atol=1e-4,rtol=0):
            raise ValueError('memory requires a gravity-aligned body frame')
        if not np.isfinite(timestamp) or timestamp<=self.clock:
            raise ValueError('acquisition timestamps must increase')
        self.clock=float(timestamp);self.pose=np.asarray(body_to_world).copy()
        self.observations[self.sequence]=Observation(self.sequence,self.clock,depth.copy(),self.pose.copy(),np.asarray(intrinsic).copy(),np.asarray(camera_to_body).copy())
        self.sequence+=1
        expired=[i for i,frame in self.observations.items() if self.clock-frame.time>HISTORY_SECONDS]
        for i in expired:
            del self.observations[i];self.maps.pop(i,None);self.projections.pop(i,None)
        self.cleared={key:t for key,t in self.cleared.items() if self.clock-t<=HISTORY_SECONDS}
        self.seen.intersection_update(self.observations)

    @torch.no_grad()
    def project(self,frame):
        if frame.index not in self.projections:
            device=self.device
            observation=PolicyCondition(depth=torch.as_tensor(frame.depth[None,None,None],device=device),
                point_goal=torch.zeros(1,2,device=device),camera_intrinsics=torch.as_tensor(frame.intrinsic[None,None],device=device,dtype=torch.float32),
                camera_to_body=torch.as_tensor(frame.camera[None,None],device=device,dtype=torch.float32),
                observation_to_current=torch.eye(4,device=device)[None,None],observation_age_s=torch.zeros(1,1,device=device),
                observation_valid=torch.ones(1,1,device=device,dtype=torch.bool),obstacle_memory=torch.zeros(1,64,64,device=device,dtype=torch.bool))
            projected=self.projector(observation,with_field=False,dense_body=True)
            valid=projected.token_valid[0,0].cpu().numpy()
            points=projected.points[0,0].cpu().numpy().astype(np.float64)
            world=points@frame.pose[:3,:3].T+frame.pose[:3,3]
            self.projections[frame.index]=dict(world=world[valid],pixels=projected.pixel_indices[0,0].cpu().numpy()[valid],
                depth=projected.depth[0,0].cpu().numpy()[valid],kind=projected.obstacle_valid[0,0].cpu().numpy()[valid])
        return self.projections[frame.index]

    def _refute(self,records,frame,*,persistent):
        keys=[key for key,value in records.items() if value['time']<frame.time]
        if not keys:return
        low=np.stack([records[key]['low'] for key in keys]);high=np.stack([records[key]['high'] for key in keys])
        corners=np.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)])
        clear=volumes_observed_clear(low[:,None]+corners*(high-low)[:,None],frame.depth,frame.intrinsic,frame.pose@frame.camera,self.maximum)
        for key in (key for key,flag in zip(keys,clear,strict=True) if flag):
            del records[key]
            if persistent:
                self.visual.pop(key,None);self.cleared[key]=frame.time

    def sample(self):
        frames=list(self.observations.values())
        ids,valid=sample_frames([frame.time for frame in frames],self.rng)
        return [frames[i].index if v else -1 for i,v in zip(ids,valid,strict=True)]

    @torch.no_grad()
    def plan_write(self,selected_indices):
        """Commit one explicit six-frame selection; replay uses the recorded selection unchanged."""
        if self.last_write==self.sequence-1:return []
        frames=list(self.observations.values());selected_indices=np.asarray(selected_indices,dtype=np.int64)
        if selected_indices.shape!=(FRAMES,):raise ValueError('write requires six frame IDs, -1 for padding')
        lookup={frame.index:i for i,frame in enumerate(frames)};valid=selected_indices>=0
        if not valid[-1] or selected_indices[-1]!=self.sequence-1:
            raise ValueError('write must end at the current acquisition')
        actual=selected_indices[valid]
        if len(np.unique(actual))!=len(actual) or (np.diff(actual)<=0).any() or any(i not in lookup for i in actual):
            raise ValueError('sources must be distinct chronological observations within six seconds')
        ids=np.array([lookup[int(i)] if v else len(frames)-1 for i,v in zip(selected_indices,valid,strict=True)])
        selected=[frames[i] for i in ids]
        self.selection=dict(indices=[frame.index if v else -1 for frame,v in zip(selected,valid,strict=True)],
            valid=valid,poses=np.stack([frame.pose for frame in selected]),ages=np.array([self.clock-frame.time for frame in selected],dtype=np.float32))
        window={}
        for frame,is_valid in zip(selected,valid,strict=True):
            if not is_valid:continue
            self._refute(self.cells,frame,persistent=True);self._refute(window,frame,persistent=False)
            projection=self.project(frame);world=projection['world']
            keys=np.column_stack((np.floor(world[:,:2]/self.resolution+.5).astype(np.int64),projection['kind']))
            unique,inverse=np.unique(keys,axis=0,return_inverse=True)
            for row,cell in enumerate(unique):
                key=tuple(map(int,cell))
                if frame.time<=self.cleared.get(key,-float('inf')):continue
                mask=inverse==row;points=world[mask]
                center=np.array([key[0]*self.resolution,key[1]*self.resolution,frame.pose[2,3]])
                source=dict(frame=frame.index,pixels=projection['pixels'][mask],offsets=(points-center).astype(np.float32),
                    world=points,depth=projection['depth'][mask],origin=(frame.pose@frame.camera)[:3,3].copy(),
                    time=frame.time,kind=key[2])
                if key not in window:window[key]=dict(sources=[],low=points.min(0)-VOXEL_M/2,high=points.max(0)+VOXEL_M/2,time=frame.time)
                record=window[key];record['sources'].append(source);record['time']=frame.time
                record['low']=np.minimum(record['low'],points.min(0)-VOXEL_M/2)
                record['high']=np.maximum(record['high'],points.max(0)+VOXEL_M/2)
        selected_ids={frame.index for frame,v in zip(selected,valid,strict=True) if v}
        # A newly sampled old image must also respect later evidence consumed at prior writes.
        for index in sorted(self.seen-selected_ids):
            self._refute(window,self.observations[index],persistent=False)
        self.seen.update(selected_ids)
        changed=[]
        for key,record in window.items():
            if key in self.cells and record['time']<=self.cells[key]['time']:continue
            record['write_time']=self.clock;cache_geometry(record)
            self.cells[key]=record;self.cells.move_to_end(key)
            self.visual.pop(key,None);changed.append(key)
        while len(self.cells)>self.capacity:
            key,_=self.cells.popitem(last=False);self.visual.pop(key,None)
        changed=[key for key in changed if key in self.cells]
        self.last_write=self.sequence-1
        self.writes.append(dict(acquisitions=self.sequence,indices=list(self.selection['indices'])))
        return changed

    @torch.no_grad()
    def write(self):
        from curvenav.deployment.streaming_memory import fuse_sources
        if self.last_write==self.sequence-1:return
        changed=self.plan_write(self.sample())
        if not changed:return
        plan=fusion_plan(self.cells,changed,self.pose,self.clock,self.selection)
        for index in np.unique(plan['frame_ids']):
            index=int(index)
            if index not in self.maps:
                frame=self.observations[index]
                self.maps[index]=self.policy.depth_encoder.image_features(torch.as_tensor(frame.depth[None,None],device=self.device))
        widths={index:self.observations[int(index)].depth.shape[1] for index in np.unique(plan['frame_ids'])}
        features=fuse_sources(self.policy,plan,self.maps,widths).detach().cpu()
        self.visual.update((key,value.clone()) for key,value in zip(changed,features,strict=True))

    def _read_keys(self):
        keys=[]
        for key,record in self.cells.items():
            world=record['world_mean']
            point=(world-self.pose[:3,3])@self.pose[:3,:3]
            if (np.abs(point[:2])<=READ_RADIUS_M).all():keys.append(key)
        if len(keys)>MAX_READ_CELLS:raise ValueError(f'memory read exceeds {MAX_READ_CELLS} cells; no truncation')
        return sorted(keys)

    def read_plan(self):
        return fusion_plan(self.cells,self._read_keys(),self.pose,self.clock,self.selection)

    def packet(self):
        plan=memory_geometry(self.cells,self._read_keys(),self.pose,self.clock,self.selection)
        visual=torch.stack([self.visual[tuple(key)] for key in plan['keys']]) if len(plan['keys']) else torch.empty(0,self.policy.depth_encoder.spatial_projection.out_channels)
        return plan,visual.to(self.device)

    def state_dict(self):
        return dict(contract=MEMORY_CONTRACT,capacity=self.capacity,resolution=self.resolution,clock=self.clock,pose=self.pose.copy(),
            sequence=self.sequence,last_write=self.last_write,selection=copy.deepcopy(self.selection),
            cells=copy.deepcopy(self.cells),cleared=dict(self.cleared),seen=set(self.seen),observations=copy.deepcopy(self.observations),
            visual={key:value.clone() for key,value in self.visual.items()},rng=copy.deepcopy(self.rng.bit_generator.state),writes=copy.deepcopy(self.writes))

    def load_state_dict(self,state):
        if state['contract']!=MEMORY_CONTRACT or state['capacity']!=self.capacity or state['resolution']!=self.resolution:
            raise ValueError('incompatible six-frame spatial memory snapshot')
        for name in ('clock','pose','sequence','last_write','selection','cells','cleared','seen','observations','visual','writes'):
            setattr(self,name,copy.deepcopy(state[name]))
        for record in self.cells.values():cache_geometry(record)
        self.rng.bit_generator.state=copy.deepcopy(state['rng']);self.maps={};self.projections={}


def cache_geometry(record):
    """Cell geometry is immutable until the next write replaces its sources."""
    sources=record['sources'];record['pixel_count']=sum(len(source['pixels']) for source in sources)
    record['world_mean']=np.concatenate([source['world'] for source in sources]).mean(0)
    record['origin_mean']=sum(source['origin']*len(source['pixels']) for source in sources)/record['pixel_count']
    record['depth_mean']=np.concatenate([source['depth'] for source in sources]).mean()


def memory_geometry(cells,keys,pose,clock,selection):
    """Read cached cell geometry without expanding historical source pixels."""
    records=[cells[key] for key in keys]
    return dict(contract=MEMORY_CONTRACT,keys=np.asarray(keys,dtype=np.int64).reshape(-1,3),
        lengths=np.asarray([r['pixel_count'] for r in records],dtype=np.int64),
        points=np.asarray([(r['world_mean']-pose[:3,3])@pose[:3,:3] for r in records],dtype=np.float32).reshape(-1,3),
        origins=np.asarray([(r['origin_mean']-pose[:3,3])@pose[:3,:3] for r in records],dtype=np.float32).reshape(-1,3),
        depth=np.asarray([r['depth_mean'] for r in records],dtype=np.float32),
        times=np.asarray([r['time'] for r in records],dtype=np.float64),pose=pose.copy(),clock=clock,selection=copy.deepcopy(selection))


def fusion_plan(cells,keys,pose,clock,selection):
    """Compact, immutable replay of the exact source membership used at each write."""
    frame_ids=[];pixels=[];offsets=[];depth=[];kind=[];age=[]
    for key in keys:
        record=cells[key];sources=record['sources']
        for source in sources:
            count=len(source['pixels']);frame_ids.extend([source['frame']]*count);pixels.extend(source['pixels'])
            offsets.extend(source['offsets']);depth.extend(source['depth']);kind.extend([source['kind']]*count)
            age.extend([record['write_time']-source['time']]*count)
    return dict(memory_geometry(cells,keys,pose,clock,selection),frame_ids=np.asarray(frame_ids,dtype=np.int64),
        pixels=np.asarray(pixels,dtype=np.int64),offsets=np.asarray(offsets,dtype=np.float32).reshape(-1,3),
        depths=np.asarray(depth,dtype=np.float32),kinds=np.asarray(kind,dtype=np.float32),ages=np.asarray(age,dtype=np.float32))
