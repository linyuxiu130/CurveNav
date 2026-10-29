"""Original D2 observation/knowledge/scoring geometry without a timed-policy import.

KnowledgeAudit, audit_branch, branch_at and return_budget retain the original
scoring equations; only their model-loading imports are removed.
"""
from curvenav.experiment_io import sha
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np

from curvenav.data.depth import BENCHMARK_INTRINSICS,preprocess_depth
from curvenav.data_generation.generate import create_simulator,set_pose
from curvenav.physical import ROBOT_BASE_HEIGHT_ABOVE_GROUND_M,ROBOT_FOOTPRINT_RADIUS_M

RUNTIME_ASSETS_PATH = None
RUNTIME_ASSETS_SHA256 = None
ROBOT_RADIUS = ROBOT_FOOTPRINT_RADIUS_M


def configure(runtime_assets):
    global RUNTIME_ASSETS_PATH, RUNTIME_ASSETS_SHA256
    RUNTIME_ASSETS_PATH = Path(runtime_assets).resolve()
    RUNTIME_ASSETS_SHA256 = sha(RUNTIME_ASSETS_PATH)
    sensor_assets.cache_clear()
    sensor_assets()

MAX_DEPTH_M=5.0


@lru_cache(maxsize=1)
def sensor_assets():
    data=json.loads(RUNTIME_ASSETS_PATH.read_text())
    assert data['complete']
    for path,digest in data['sources'].items():assert sha(path)==digest
    sources=list(data['sources'])
    config_path=next(Path(path) for path in sources if Path(path).name=='config.json')
    intrinsic_path=next(Path(path) for path in sources if Path(path).name=='camera_intrinsics.npy')
    camera_path=next(Path(path) for path in sources if Path(path).name=='camera_to_body.npy')
    config=json.loads(config_path.read_text())
    assert config['camera']==data['camera']
    intrinsic=np.load(intrinsic_path,mmap_mode='r')[0,-1].astype(np.float32)
    camera_to_body=np.load(camera_path,mmap_mode='r')[0,-1].astype(np.float32)
    np.testing.assert_allclose(intrinsic,np.asarray(data['intrinsics'],dtype=np.float32),rtol=0,atol=1e-6)
    np.testing.assert_allclose(camera_to_body,np.asarray(data['camera_to_body'],dtype=np.float32),rtol=0,atol=1e-6)
    assert abs(float(data['body_radius_m'])-ROBOT_FOOTPRINT_RADIUS_M)<1e-8
    assert abs(float(data['body_base_height_m'])-ROBOT_BASE_HEIGHT_ABOVE_GROUND_M)<1e-8
    return data,intrinsic,camera_to_body


def clearance(paths,boxes,radius=ROBOT_RADIUS):
    """Original pure segment-to-box clearance used by the D2 scorer."""
    paths=np.asarray(paths,dtype=np.float64)
    if paths.ndim==2:paths=paths[None]
    boxes=np.asarray(boxes,dtype=np.float64)[None]
    a,b=paths[None,:,:-1,None,:],paths[None,:,1:,None,:]
    lo,hi=boxes[:,None,None,:,:2],boxes[:,None,None,:,2:]
    delta=b-a
    near=np.divide(lo-a,delta,out=np.full_like(lo-a,-np.inf),where=delta!=0)
    far=np.divide(hi-a,delta,out=np.full_like(hi-a,np.inf),where=delta!=0)
    enter=np.maximum(np.minimum(near,far).max(-1),0)
    leave=np.minimum(np.maximum(near,far).min(-1),1)
    intersects=(enter<=leave)&~(((delta==0)&((a<lo)|(a>hi))).any(-1))
    endpoint=np.minimum(np.linalg.norm(np.maximum(np.maximum(lo-a,a-hi),0),axis=-1),
        np.linalg.norm(np.maximum(np.maximum(lo-b,b-hi),0),axis=-1))
    corners=np.stack((lo,np.stack((lo[...,0],hi[...,1]),-1),hi,
        np.stack((hi[...,0],lo[...,1]),-1)),-2)
    fraction=(((corners-a[...,None,:])*delta[...,None,:]).sum(-1)/
        np.maximum((delta**2).sum(-1)[...,None],1e-20)).clip(0,1)
    corner_distance=np.linalg.norm(corners-(a[...,None,:]+fraction[...,None]*delta[...,None,:]),axis=-1).min(-1)
    return np.where(intersects,0,np.minimum(endpoint,corner_distance)).min((2,3))[0]-radius


def body_pose(view):
    x,y,angle=np.asarray(view,dtype=np.float64)
    pose=np.eye(4);c,s=math.cos(angle),math.sin(angle)
    pose[:2,:2]=[[c,-s],[s,c]]
    pose[:3,3]=[x,y,float(sensor_assets()[0]['body_base_height_m'])]
    return pose.astype(np.float32)


def observation(sim,view,k):
    set_pose(sim,np.asarray([view[0],0.,-view[1]],dtype=np.float32),-float(view[2]))
    raw=np.asarray(sim.get_sensor_observations()['depth'])
    if raw.dtype!=np.float32 or raw.shape!=(BENCHMARK_INTRINSICS.height,BENCHMARK_INTRINSICS.width):
        raise RuntimeError('Habitat returned an unexpected D2 depth frame')
    depth,measured_k=preprocess_depth(raw,source_intrinsics=BENCHMARK_INTRINSICS,
        maximum_m=MAX_DEPTH_M,height=126,width=224)
    np.testing.assert_allclose(measured_k,k,rtol=0,atol=1e-5)
    return depth.astype(np.float16)

def margin(points,boxes,tolerance=.0075):
    return float(clearance(np.asarray(points)[None],np.asarray(boxes))[0]-tolerance)

def step(view,control,dt):
    v,w=control;angle=w*dt;distance=v*dt*np.sinc(angle/(2*np.pi));mid=view[2]+angle/2
    return np.asarray(view)+[distance*math.cos(mid),distance*math.sin(mid),angle]

def calibration():
    _,k,camera=sensor_assets()
    return k.copy(),camera.copy()

class KnowledgeAudit:
    """Evaluation-only visibility from measured depth, never passed to policy."""
    def __init__(self,case,k,camera):
        self.case=case;self.k=k;self.camera=camera
        x=np.arange(case['floor_bounds'][0]+.1,case['floor_bounds'][2],.2);y=np.arange(case['floor_bounds'][1]+.1,case['floor_bounds'][3],.2)
        xx,yy=np.meshgrid(x,y);xy=np.stack((xx.ravel(),yy.ravel()),1)
        safe=clearance(np.repeat(xy[:,None],2,axis=1),np.asarray(case['boxes']),radius=ROBOT_RADIUS)>.03
        self.xy=xy[safe];self.known=np.zeros(len(self.xy),bool)
        self.wall_xy=[]
        for x0,y0,x1,y1 in case['boxes']:
            for aa,zz in [([x0,y0],[x1,y0]),([x1,y0],[x1,y1]),([x1,y1],[x0,y1]),([x0,y1],[x0,y0])]:
                length=np.linalg.norm(np.array(zz)-aa)
                if length>.01:self.wall_xy.extend(np.linspace(aa,zz,max(2,int(length/.2)+1)).tolist())
        self.wall_xy=np.array(self.wall_xy);self.wall_seen=np.zeros(len(self.wall_xy),bool)
    def visible(self,xy,image,pose,wall=False):
        camera=pose@self.camera;points=np.column_stack((xy,np.full(len(xy),.09)))
        optical=(points-camera[:3,3])@camera[:3,:3];z=optical[:,2];uv=optical[:,:2]/np.maximum(z[:,None],1e-9)
        uv=uv*np.array([self.k[0,0],self.k[1,1]])+self.k[:2,2];ij=np.floor(uv).astype(int)
        valid=(z>0)&(z<5)&(ij[:,0]>=0)&(ij[:,0]<image.shape[1])&(ij[:,1]>=0)&(ij[:,1]<image.shape[0]);result=np.zeros(len(xy),bool)
        measured=image[ij[valid,1],ij[valid,0]].astype(float)*5
        result[valid]=abs(measured-z[valid])<.08 if wall else (measured-z[valid]>.02)
        return result
    def update(self,image,view):
        pose=body_pose(view);self.known|=self.visible(self.xy,image,pose)
        self.known|=np.linalg.norm(self.xy-view[:2],axis=1)<ROBOT_RADIUS
        self.wall_seen|=self.visible(self.wall_xy,image,pose,wall=True)
    def fraction(self):return float(self.known.mean())

def audit_branch(case, arm_index, prefix_views, obs_cache, k, camera):
    """Audit one axis-aligned D2 arm from cached real depth observations.

    Cache keys follow Replay.observation: tuple(np.round(view, 7)); values are
    normalized depth arrays, not the older (depth, feature) cache tuples.
    Missing observations fail visibly. No graph edges or returned flags are read.
    """

    route = np.asarray(case['deadend_arms'][arm_index]['route'], dtype=float)
    delta = np.diff(route, axis=0)
    if not np.all((abs(delta) > 1e-8).sum(-1) == 1):
        raise ValueError('D2 branch audit requires nonzero axis-aligned segments')
    half, spacing, eps = float(case['hall']) / 2, .1, 1e-5
    low = np.minimum(route[:-1], route[1:]) - half
    high = np.maximum(route[:-1], route[1:]) + half

    def inside(points):
        return ((points[:, None] >= low) & (points[:, None] <= high)).all(-1).any(-1)

    boundary = []
    for lo, hi in zip(low, high):
        for axis in (0, 1):
            other = 1 - axis
            samples = np.linspace(lo[other], hi[other],
                                  int(np.ceil((hi[other] - lo[other]) / spacing)) + 1)
            for sign in (-1, 1):
                points = np.empty((len(samples), 2))
                points[:, other] = samples
                points[:, axis] = lo[axis] if sign == -1 else hi[axis]
                normal = np.eye(2)[axis] * sign
                keep = inside(points - eps * normal) & ~inside(points + eps * normal)
                boundary.extend(points[keep])
    boundary = np.unique(np.round(boundary, 9), axis=0)
    # The fork square is an intentional opening into other corridors.
    boundary = boundary[(abs(boundary - route[0]).max(-1) > half + eps)]
    direction = delta[-1] / np.linalg.norm(delta[-1])
    normal = np.array([-direction[1], direction[0]])
    lateral = np.linspace(-half + .05, half - .05,
                          int(np.ceil((2 * half - .1) / spacing)) + 1)
    terminal = route[-1] + direction * half + lateral[:, None] * normal
    terminal_face = abs((boundary - route[-1]) @ direction - half) < eps
    sides = boundary[~terminal_face]
    assert len(sides) > 0 and len(terminal) > 0
    audit = KnowledgeAudit(case, k, camera)
    terminal_seen = np.zeros(len(terminal), dtype=bool)
    side_seen = np.zeros(len(sides), dtype=bool)
    for view in np.asarray(prefix_views):
        depth = obs_cache[tuple(np.round(view, 7))]
        pose = body_pose(view)
        audit.update(depth, view)
        terminal_seen |= audit.visible(terminal, depth, pose, wall=True)
        side_seen |= audit.visible(sides, depth, pose, wall=True)
    region = inside(audit.xy) & (abs(audit.xy - route[0]).max(-1) > half + eps)
    free_count = int(region.sum())
    assert free_count > 0
    unknown = int((region & ~audit.known).sum())
    terminal_fraction = float(terminal_seen.mean())
    side_fraction = float(side_seen.mean())
    # Strict side/free gates prevent a visible end wall concealing a side exit.
    confirmed = terminal_fraction >= .9 and bool(side_seen.all()) and unknown == 0
    return dict(arm_index=int(arm_index), confirmed=confirmed,
                terminal_wall_fraction=terminal_fraction,
                terminal_samples=len(terminal), terminal_seen=int(terminal_seen.sum()),
                side_boundary_fraction=side_fraction, side_boundary_confirmed=bool(side_seen.all()),
                side_samples=len(sides), side_seen=int(side_seen.sum()),
                free_points=free_count, unknown_free_points=unknown,
                observed_free_fraction=1 - unknown / free_count,
                unseen_side_xy=sides[~side_seen].tolist(),
                scope='Conservative sampled-depth audit, not a continuous free-space certificate; '
                      'unconfirmed means insufficient evidence, not an open branch.')

def return_budget(row, fork):
    if row['stage'] == 'returned':
        return 0, 64
    views = np.asarray(row['views'])
    last_fork = np.flatnonzero(np.linalg.norm(views[:, :2]-fork, axis=1) < .15)[-1]
    moves = np.diff(views[last_fork:, :2], axis=0)
    moves = moves[np.linalg.norm(moves, axis=1) > 1e-6]
    yaw = np.arctan2(moves[:, 1], moves[:, 0])
    turns = abs(np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw)))).sum()
    count = int(np.ceil(np.linalg.norm(moves, axis=1).sum()/.125)+np.ceil((turns+np.pi)/.3))
    return count, 2*(count+32)

def branch_at(view, case, confirmed):
    for index in confirmed:
        route = np.asarray(case['deadend_arms'][index]['route'])
        if abs(view[:2]-route[0]).max() <= .8:
            continue
        low = np.minimum(route[:-1], route[1:])-case['hall']/2
        high = np.maximum(route[:-1], route[1:])+case['hall']/2
        if ((view[:2] >= low) & (view[:2] <= high)).all(-1).any():
            return index
    return None

class Environment:
    def __init__(self,actor,case,gpu=0,*,timestamp_origin=.5):
        self.case=case
        runtime_assets,self.k,self.camera=sensor_assets()
        self.sim=create_simulator(Path(case['assets']),'maze',gpu,runtime_assets['camera'],'maze.scene_dataset_config.json')
        self.memory=actor.new_memory()
        self.knowledge=KnowledgeAudit(case,self.k,self.camera)
        self.clock=timestamp_origin-.5;self.path=0.;self.cache={};self.view=np.asarray(case['start'],dtype=np.float64)
        for view in np.asarray(case['prefix_views'],dtype=np.float64):
            self.write(view)
            self.memory.write()  # Recorded prefix acquisition events prefill the same writer.

    def write(self,view):
        key=tuple(np.round(view,7))
        if key not in self.cache:self.cache[key]=observation(self.sim,view,self.k)
        image=self.cache[key];self.clock+=.5
        self.memory.update(image,self.k,self.camera,body_pose(view),self.clock)
        self.knowledge.update(image,view);self.view=np.asarray(view,dtype=np.float64).copy()
