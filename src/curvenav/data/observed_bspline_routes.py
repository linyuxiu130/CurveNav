"""Original V4 sensor routes and actual-hit history, without a timed actor/codec import.

Only unused timed plan/_controls/_path and codec initialization are removed.
Sensor update, refutation, footprint, occupied clearance and route search are retained.
"""
from collections import deque
from itertools import product
import math
import sys
import numpy as np
from scipy import ndimage
from curvenav.data.obstacle_memory import VOXEL_M, volumes_observed_clear
from curvenav.physical import ROBOT_BASE_HEIGHT_ABOVE_GROUND_M


def body_pose(view):
    c,s=math.cos(float(view[2])),math.sin(float(view[2]))
    return np.asarray([[c,-s,0.,view[0]],[s,c,0.,view[1]],
        [0.,0.,1.,ROBOT_BASE_HEIGHT_ABOVE_GROUND_M],[0.,0.,0.,1.]],dtype=np.float32)

ROUTE_SOURCE_SHA256 = '6c3f4df9f6e69636defca8c45b9ce38c234d9a1c55574bcc0d8bd571c2b8d4a6'
HIT_SOURCE_SHA256 = '37b391a3eada8dc681f21e69989017ba9f766fe4972c5c2d6767121180508239'

class ObservedBSplineRoutes:
    grid = .05
    radius = .167584539
    margin = .10
    waypoint_tolerance = .005
    forecast_guard = .0075
    maximum = 5.
    band = (.05, .161981515)

    def __init__(self, k, camera):
        self.k = np.asarray(k, dtype=float).copy()
        self.camera = np.asarray(camera, dtype=float).copy()
        self.cells = {}
        self.occupied = set()
        self.corners = np.asarray(list(product((0., 1.), repeat=3)))
        self.view = None
        self.updates = 0
        self._rays = None

    def update(self, depth, view, executed=True, executed_sweep=None):
        """Consume one actually acquired normalized optical-Z image and its pose.

        A supplied sweep contains actual SE2 poses, not virtual observations.
        Only the physical radius is used; curved segments shrink their chord
        capsule by the arc sagitta. Without a sweep only this pose is certified.
        """
        depth = np.asarray(depth, dtype=np.float32)
        view = np.asarray(view, dtype=float)
        if depth.ndim != 2 or not np.isfinite(depth).all() or ((depth < 0) | (depth > 1)).any():
            raise ValueError('depth must be finite normalized optical-Z in [0,1]')
        if view.shape != (3,) or not np.isfinite(view).all():
            raise ValueError('view must be finite metric x,y,yaw')
        sweep = view[None] if executed_sweep is None else np.asarray(executed_sweep, dtype=float)
        if sweep.ndim != 2 or sweep.shape[1] != 3 or not len(sweep) or not np.isfinite(sweep).all():
            raise ValueError('executed_sweep must contain finite actual SE2 poses')
        if not np.allclose(sweep[-1], view, rtol=0, atol=1e-7):
            raise ValueError('executed_sweep must end at the acquired pose')
        if executed_sweep is not None:
            if not executed or self.view is None and len(sweep) != 1:
                raise ValueError('a sweep requires a preceding actual acquired pose')
            if self.view is not None and not np.allclose(sweep[0], self.view, rtol=0, atol=1e-7):
                raise ValueError('executed_sweep must begin at the previous acquired pose')
        if self._rays is None:
            y, x = np.indices(depth.shape, dtype=float)
            self._rays = np.stack(((x+.5-self.k[0, 2])/self.k[0, 0],
                                  (y+.5-self.k[1, 2])/self.k[1, 1], np.ones_like(x)), -1)
        if self._rays.shape[:2] != depth.shape:
            raise ValueError('observation resolution changed without new calibration')
        pose = body_pose(view)
        camera_world = pose @ self.camera
        world = (self._rays*(depth*self.maximum)[..., None]) @ camera_world[:3, :3].T + camera_world[:3, 3]
        observed = depth > 0
        bounds = np.concatenate((world[observed, :2], sweep[:, :2]-self.radius, sweep[:, :2]+self.radius))
        lo = np.floor(bounds.min(0)/self.grid).astype(int)-1
        hi = np.floor(bounds.max(0)/self.grid).astype(int)+1
        x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        keys = np.column_stack((x.ravel(), y.ravel()))
        masks = np.array([self.cells.get(tuple(key), 0) for key in keys], dtype=np.uint8)
        pending = masks != 255
        heights = np.linspace(*self.band, 9)
        lower = np.column_stack((np.repeat(keys[pending]*self.grid, 8, axis=0),
                                 np.tile(heights[:-1], int(pending.sum()))))
        size = np.array([self.grid, self.grid, heights[1]-heights[0]])
        boxes = lower[:, None]+self.corners*size
        clear = volumes_observed_clear(boxes, depth, self.k, camera_world, self.maximum).reshape(-1, 8)
        bits = (clear.astype(np.uint16) * (1 << np.arange(8))).sum(1).astype(np.uint8)
        masks[pending] |= bits
        if executed:
            near = (((keys+.5)*self.grid >= sweep[:, :2].min(0)-self.radius-self.grid) &
                    ((keys+.5)*self.grid <= sweep[:, :2].max(0)+self.radius+self.grid)).all(-1)
            corners = (keys[near, None]+np.asarray(list(product((0., 1.), repeat=2))))*self.grid
            certified = np.zeros(int(near.sum()), bool)
            segments = zip(sweep[:-1], sweep[1:]) if len(sweep) > 1 else [(view, view)]
            for a, b in segments:
                delta = b[:2]-a[:2]
                length = np.linalg.norm(delta)
                angle = abs(np.arctan2(np.sin(b[2]-a[2]), np.cos(b[2]-a[2])))
                sagitta = length*.5*np.tan(angle*.25)
                if length:
                    along = np.clip(((corners-a[:2])*delta).sum(-1)/(length*length), 0, 1)
                    nearest = a[:2]+along[..., None]*delta
                else:
                    nearest = a[:2]
                certified |= np.linalg.norm(corners-nearest, axis=-1).max(-1) <= self.radius-sagitta
            masks[np.flatnonzero(near)[certified]] = 255
        for key, mask in zip(keys, masks):
            if mask:
                self.cells[tuple(key)] = int(mask)
        if self.occupied:
            voxels = np.asarray(sorted(self.occupied), dtype=np.int64)
            boxes = (voxels[:, None]+self.corners)*VOXEL_M
            cleared = volumes_observed_clear(boxes, depth, self.k, camera_world, self.maximum)
            self.occupied.difference_update(map(tuple, voxels[cleared]))
        hit = observed & (depth < 1) & (world[..., 2] >= self.band[0]) & (world[..., 2] <= self.band[1])
        self.occupied.update(map(tuple, np.unique(np.floor(world[hit]/VOXEL_M).astype(np.int64), axis=0)))
        for key in self._occupied_cells():
            self.cells.pop(key, None)
        self.view = view.copy()
        self.updates += 1

    def _occupied_cells(self):
        if not self.occupied:
            return set()
        voxels = np.asarray(list(self.occupied))[:, :2]
        lo = np.floor(voxels*VOXEL_M/self.grid).astype(int)
        hi = np.floor(np.nextafter((voxels+1)*VOXEL_M, -np.inf)/self.grid).astype(int)
        return {tuple(cell) for dx, dy in product((0, 1), repeat=2)
                for cell in (lo+[dx, dy])[((lo+[dx, dy]) <= hi).all(-1)]}

    def _raster(self, view):
        occupied = self._occupied_cells()
        keys = list(self.cells) + list(occupied) + [tuple(np.floor(view[:2]/self.grid).astype(int))]
        keys = np.asarray(keys)
        lo, hi = keys.min(0)-5, keys.max(0)+5
        shape = tuple((hi-lo+1)[::-1])
        free, blocked = np.zeros(shape, bool), np.zeros(shape, bool)
        for key, value in self.cells.items():
            x, y = np.asarray(key)-lo
            free[y, x] = value == 255
        for key in occupied:
            x, y = np.asarray(key)-lo
            blocked[y, x] = True
        free &= ~blocked
        return free, blocked, lo

    def footprint_observed(self, points, radius=None):
        """Conservative disk coverage by fully observed body-height grid columns."""
        radius = self.radius if radius is None else radius
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        reach = int(np.ceil(radius/self.grid))+1
        offsets = np.asarray(list(product(range(-reach, reach+1), repeat=2)))
        keys = np.floor(points[:, None]/self.grid).astype(int)+offsets
        distance = np.maximum(abs((keys+.5)*self.grid-points[:, None])-self.grid/2, 0)
        overlap = np.linalg.norm(distance, axis=-1) <= radius
        needed = np.unique(keys[overlap], axis=0)
        return all(self.cells.get(tuple(key), 0) == 255 for key in needed)


    def occupied_clearance(self, points):
        """Clearance to retained measured occupied voxels, never scene boxes."""
        if not self.occupied:
            return float('inf')
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        lower = np.unique(np.asarray(list(self.occupied))[:, :2], axis=0)*VOXEL_M
        upper = lower+VOXEL_M
        reach = self.radius+self.margin+self.forecast_guard
        nearby = ((upper >= points.min(0)-reach) & (lower <= points.max(0)+reach)).all(-1)
        if not nearby.any():
            return float('inf')
        distances = np.maximum(np.maximum(lower[None, nearby]-points[:, None],
                                           points[:, None]-upper[None, nearby]), 0)
        return float(np.linalg.norm(distances, axis=-1).min()-self.radius)

    def _remaining_route(self, view, route):
        index = 1
        while index < len(route)-1 and np.linalg.norm(route[index]-view[:2]) <= np.sqrt(2)*self.grid:
            # Remove a tiny grid attachment only when the direct remaining
            # segment is itself observed and keeps the occupied margin.
            distance = np.linalg.norm(route[index+1]-view[:2])
            points = np.linspace(view[:2], route[index+1], max(2, int(np.ceil(distance/.01))+1))
            if (not self.footprint_observed(points, self.radius+.005) or
                    self.occupied_clearance(points) < self.margin+self.forecast_guard):
                break
            index += 1
        return np.concatenate((view[None, :2], route[index:]))


    def source_frontier_target(self, view, source_distance):
        """Choose an observed frontier using the same source C-space as the expert."""
        free, blocked, lo = self._raster(view)
        distance = ndimage.distance_transform_edt(free)*self.grid-self.grid/np.sqrt(2)
        obstacle_distance = ndimage.distance_transform_edt(~blocked)*self.grid if blocked.any() else np.full(free.shape, np.inf)
        safe = (distance >= self.radius) & (obstacle_distance-self.grid/np.sqrt(2) >=
            self.radius+self.margin+self.waypoint_tolerance+self.forecast_guard)
        # Reject the narrow unobserved fringes directly in front of measured walls.
        unknown = ~free & ~blocked
        unknown_core = ndimage.distance_transform_edt(unknown)*self.grid-self.grid/np.sqrt(2) >= self.radius
        near_core = ndimage.distance_transform_edt(~unknown_core)*self.grid <= self.radius+self.margin+2*self.grid
        frontier = free & ndimage.binary_dilation(unknown) & near_core & (obstacle_distance > self.radius+self.margin+self.grid)
        if not unknown_core.any():
            frontier[:] = False
        labels, count = ndimage.label(frontier, structure=np.ones((3, 3), bool))
        sizes = np.bincount(labels.ravel())
        frontier &= sizes[labels] >= 3
        if not frontier.any():
            return None, dict(reason='no_frontier', frontier_cells=0, safe_cells=int(safe.sum()))
        sites=np.argwhere(safe)
        xy=(sites[:,::-1]+lo+.5)*self.grid
        source_grid=source_distance.source.grids[0]
        current_cell=np.floor((view[:2]-source_grid.origin_xy)/source_grid.cell_size_m).astype(int)
        cells=np.floor((xy-source_grid.origin_xy)/source_grid.cell_size_m).astype(int)
        travel=source_distance.query(0,current_cell,cells)
        source_cost=np.full(free.shape,np.inf)
        source_cost[sites[:,0],sites[:,1]]=travel
        candidates=[]
        for label in np.unique(labels[frontier]):
            frontier_distance=np.full(free.shape,-1,dtype=np.int32)
            queue=deque(map(tuple,np.argwhere(frontier&(labels==label))))
            for node in queue:frontier_distance[node]=0
            while queue:
                node=queue.popleft()
                for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
                    other=node[0]+dy,node[1]+dx
                    if (0<=other[0]<free.shape[0] and 0<=other[1]<free.shape[1]
                            and free[other] and frontier_distance[other]<0):
                        frontier_distance[other]=frontier_distance[node]+1;queue.append(other)
            reachable=list(map(tuple,np.argwhere(safe&(frontier_distance>=0)&np.isfinite(source_cost))))
            if not reachable:continue
            approach=min(reachable,key=lambda node:(frontier_distance[node],source_cost[node],node))
            point=(np.asarray(approach)[::-1]+lo+.5)*self.grid
            if np.linalg.norm(point-view[:2])<=self.waypoint_tolerance:continue
            candidates.append((float(source_cost[approach]),approach,int(label),point))
        if not candidates:
            return None,dict(reason='no_reachable_frontier',frontier_cells=int(frontier.sum()),safe_cells=int(safe.sum()))
        distance,target,component,point=min(candidates,key=lambda c:c[:3])
        return point,dict(reason='source_connected_frontier',target_world=point.tolist(),
            source_distance_m=distance,
            frontier_cells=int(frontier.sum()),safe_cells=int(safe.sum()),
            reachable_frontier_components=len(candidates),frontier_component=component)



RECORD=np.dtype([('world',np.float64,(3,)),('frame',np.uint32),('pixel',np.uint32)])

class ObservedHitHistory:
    maximum=ObservedBSplineRoutes.maximum
    band=ObservedBSplineRoutes.band

    def __init__(self,k,camera):
        self.k=np.asarray(k,dtype=float).copy();self.camera=np.asarray(camera,dtype=float).copy()
        self.live={};self.corners=np.asarray(list(product((0.,1.),repeat=3)))
        self.rays=None;self.updates=0;self.view=None;self.error_bound=None

    def update(self,depth,view):
        depth=np.asarray(depth,dtype=np.float32);view=np.asarray(view,dtype=float)
        if depth.ndim!=2 or not np.isfinite(depth).all() or ((depth<0)|(depth>1)).any():
            raise ValueError('depth must be finite normalized optical-Z in[0,1]')
        if view.shape!=(3,) or not np.isfinite(view).all():raise ValueError('view must be finite x,y,yaw')
        if self.rays is None:
            y,x=np.indices(depth.shape,dtype=float)
            self.rays=np.stack(((x+.5-self.k[0,2])/self.k[0,0],(y+.5-self.k[1,2])/self.k[1,1],np.ones_like(x)),-1)
            self.error_bound=float(5*np.finfo(np.float16).eps*np.linalg.norm(self.rays,axis=-1).max())
        if self.rays.shape[:2]!=depth.shape:raise ValueError('observation resolution changed')
        camera_world=body_pose(view)@self.camera
        # Match V4 exactly: depth is FP32 and depth*5 happens before FP64 rays.
        world=(self.rays*(depth*self.maximum)[...,None])@camera_world[:3,:3].T+camera_world[:3,3]
        cleared_keys=set();removed_points=0
        if self.live:
            voxels=np.asarray(sorted(self.live),dtype=np.int64)
            cleared=volumes_observed_clear((voxels[:,None]+self.corners)*VOXEL_M,depth,self.k,camera_world,self.maximum)
            cleared_keys=set(map(tuple,voxels[cleared]))
            for key in cleared_keys:
                removed_points+=sum(len(chunk) for chunk in self.live.pop(key))
        hit=(depth>0)&(depth<1)&(world[...,2]>=self.band[0])&(world[...,2]<=self.band[1])
        pixels=np.flatnonzero(hit);points=world.reshape(-1,3)[pixels]
        keys,inverse=np.unique(np.floor(points/VOXEL_M).astype(np.int64),axis=0,return_inverse=True)
        records=np.empty(len(points),dtype=RECORD)
        records['world']=points;records['frame']=self.updates;records['pixel']=pixels
        order=np.argsort(inverse,kind='stable');ends=np.cumsum(np.bincount(inverse,minlength=len(keys)))
        start=0
        for key,end in zip(keys,ends,strict=True):
            self.live.setdefault(tuple(key),[]).append(records[order[start:end]].copy());start=end
        assert start==len(records)
        self.view=view.copy();self.updates+=1
        return dict(frame=self.updates-1,hits=len(points),cleared_voxels=len(cleared_keys),
            removed_points=removed_points,cleared_then_rehit=sum(tuple(key) in cleared_keys for key in keys))

    def records(self):
        chunks=[chunk for key in sorted(self.live) for chunk in self.live[key]]
        return np.concatenate(chunks) if chunks else np.empty(0,dtype=RECORD)

    def memory(self):
        chunks=[chunk for values in self.live.values() for chunk in values]
        return dict(points=sum(len(chunk) for chunk in chunks),voxels=len(self.live),chunks=len(chunks),
            array_bytes=sum(chunk.nbytes for chunk in chunks),
            approximate_python_and_arrays_bytes=sys.getsizeof(self.live)+sum(sys.getsizeof(k)+sys.getsizeof(v)
                for k,v in self.live.items())+sum(sys.getsizeof(chunk) for chunk in chunks))
