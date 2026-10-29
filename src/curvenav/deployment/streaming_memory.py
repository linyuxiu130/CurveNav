"""Six-frame fusion writes descriptors; one spatial latent feeds geometry and environment tokens."""
from dataclasses import replace
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from curvenav.data.history import inverse_rigid
from curvenav.encoders.configuration import memory_splat_stencil,apply_memory_splat


def source_groups(plan):
    """The replay plan is immutable; group its pixels once, in original order."""
    if '_source_groups' not in plan:
        ids,counts=np.unique(plan['frame_ids'],return_counts=True)
        order=np.argsort(plan['frame_ids'],kind='stable')
        positions=np.split(order,np.cumsum(counts)[:-1])
        plan['_source_groups']=list(zip(ids,positions))
    return plan['_source_groups']


def source_sample_inputs(plan,maps,widths):
    """Cache each immutable plan's calibrated pixels and ragged bucket order."""
    groups=source_groups(plan)
    first=maps[int(groups[0][0])];device=first.device
    cache=plan.setdefault('_source_sample_inputs',{})
    key=(device,tuple(first.shape))
    if key not in cache:
        buckets={}
        for frame,selected in groups:
            frame=int(frame);size=1<<(len(selected)-1).bit_length()
            buckets.setdefault((tuple(maps[frame].shape[1:]),widths[frame],size),[]).append((frame,selected))
        prepared=[];destinations=[]
        for (_,image_width,size),members in buckets.items():
            pixels=np.zeros((len(members),1,size),dtype=np.int64);valid=[]
            for row,(_,selected) in enumerate(members):
                pixels[row,0,:len(selected)]=plan['pixels'][selected]
                valid.append(row*size+np.arange(len(selected)))
                destinations.append(selected)
            prepared.append(([frame for frame,_ in members],image_width,
                torch.as_tensor(pixels,device=device),torch.as_tensor(np.concatenate(valid),device=device)))
        cache[key]=(prepared,torch.as_tensor(np.argsort(np.concatenate(destinations)),device=device))
    return cache[key]


def source_features_batch(encoder,plans,maps,widths):
    """Batch the original sampler across independent states with matching buckets."""
    inputs=[source_sample_inputs(p,m,w) for p,m,w in zip(plans,maps,widths,strict=True)]
    groups={};pieces=[[None]*len(prepared) for prepared,_ in inputs]
    for row,(prepared,_) in enumerate(inputs):
        for slot,(frames,image_width,pixels,valid) in enumerate(prepared):
            shape=tuple(maps[row][frames[0]].shape[1:])
            groups.setdefault((shape,image_width,pixels.shape[-1]),[]).append((row,slot,frames,pixels,valid))
    for (_,image_width,_),members in groups.items():
        features=torch.cat([maps[row][frame] for row,_,frames,_,_ in members for frame in frames])
        pixels=torch.cat([pixels for _,_,_,pixels,_ in members])
        sampled=encoder.sample_pixel_features(features,pixels,image_width)[:,0]
        for values,(row,slot,_,_,valid) in zip(sampled.split([len(frames) for _,_,frames,_,_ in members]),members,strict=True):
            pieces[row][slot]=values.reshape(-1,features.shape[1]).index_select(0,valid)
    return [torch.cat(parts).index_select(0,order).float() for parts,(_,order) in zip(pieces,inputs,strict=True)]


def source_features(encoder,plan,maps,widths):
    return source_features_batch(encoder,[plan],[maps],[widths])[0]


def fuse_sources(policy,plan,maps,widths):
    """Identical differentiable write fusion for online memory and training replay."""
    return fuse_sources_batch(policy,[plan],[maps],[widths])[0]


def unique_fusion_cells(plans,maps,widths):
    """Identical cell writes share current-weight pixel embeddings within this call."""
    if len(plans)<2:return None
    seen={};inverse=[];lengths=[];selections=[]
    for plan,features,image_widths in zip(plans,maps,widths,strict=True):
        boundaries=np.r_[0,np.cumsum(plan['lengths'])]
        if '_fusion_cells' not in plan:
            cells=[]
            for start,end in zip(boundaries[:-1],boundaries[1:],strict=True):
                digest=hashlib.sha256()
                for name in ('frame_ids','pixels','offsets','depths','kinds','ages'):
                    digest.update(plan[name][start:end].tobytes())
                cells.append((digest.digest(),tuple(map(int,np.unique(plan['frame_ids'][start:end])))))
            plan['_fusion_cells']=cells
        selected=[]
        for cell,(signature,frames) in enumerate(plan['_fusion_cells']):
            key=(signature,tuple((id(features[frame]),image_widths[frame]) for frame in frames))
            if key not in seen:
                seen[key]=len(lengths);start,end=boundaries[cell:cell+2]
                selected.append(np.arange(start,end));lengths.append(end-start)
            inverse.append(seen[key])
        selections.append(np.concatenate(selected) if selected else np.empty(0,dtype=np.int64))
    if len(seen)==len(inverse):return None
    device=next(iter(maps[0].values())).device
    return torch.as_tensor(inverse,device=device),torch.as_tensor(lengths,device=device),selections


def fuse_sources_batch(policy,plans,maps,widths):
    """Fuse independent writes together; segment lengths keep each cell separate."""
    encoder=policy.depth_encoder;device=next(policy.parameters()).device
    sizes=[len(p['lengths']) for p in plans];active=[i for i,size in enumerate(sizes) if size]
    if not active:return [torch.empty(0,encoder.spatial_projection.out_channels,device=device) for _ in plans]
    inputs=[]
    for index in active:
        plan=plans[index];cache=plan.setdefault('_fusion_inputs',{});key=(device,encoder.max_depth_m)
        if key not in cache:
            tensor=lambda value:torch.as_tensor(value,device=device,dtype=torch.float32)
            geometry=torch.cat((tensor(plan['offsets'])/encoder.max_depth_m,
                (tensor(plan['depths'])/encoder.max_depth_m)[:,None],
                (tensor(plan['depths'])<encoder.max_depth_m)[:,None],tensor(plan['kinds'])[:,None]),-1)
            ages,inverse=np.unique(plan['ages'],return_inverse=True)
            cache[key]=(geometry,tensor(ages)[:,None],torch.as_tensor(inverse,device=device),torch.as_tensor(plan['lengths'],device=device))
        inputs.append(cache[key])
    ages=torch.cat([values[1] for values in inputs])
    lengths=torch.cat([values[3] for values in inputs])
    unique=unique_fusion_cells([plans[i] for i in active],[maps[i] for i in active],[widths[i] for i in active])
    source_plans=[plans[i] for i in active];selected=None;used=list(range(len(active)))
    if unique is not None:
        cells,lengths,selections=unique;source_plans=[]
        used=[i for i,indices in enumerate(selections) if len(indices)]
        selected=[torch.as_tensor(selections[i],device=device) for i in used]
        for i in used:
            plan=plans[active[i]];indices=selections[i];key=indices.tobytes()
            cached=plan.get('_selected_source_plan')
            if cached is None or cached[0]!=key:
                plan['_selected_source_plan']=(key,{name:plan[name][indices] for name in ('frame_ids','pixels')})
            source_plans.append(plan['_selected_source_plan'][1])
    geometry=torch.cat([values[0] for values in inputs] if selected is None else
        [inputs[i][0].index_select(0,indices) for i,indices in zip(used,selected,strict=True)])
    visual=torch.cat(source_features_batch(encoder,source_plans,[maps[active[i]] for i in used],[widths[active[i]] for i in used]))
    # FP32 fusion avoids a second BF16 rounding boundary dependent on the number of written cells.
    with torch.autocast(device_type=device.type,enabled=False):
        # Every pixel from one write shares an age; compute its current embedding once.
        age_features=encoder.time_projection(ages).split([len(values[1]) for values in inputs])
        time=torch.cat([F.embedding(values[2],features) for features,values in zip(age_features,inputs,strict=True)] if selected is None else
            [F.embedding(inputs[i][2].index_select(0,indices),age_features[i]) for i,indices in zip(used,selected,strict=True)])
        tokens=encoder.output_norm(visual+encoder.geometry_projection(geometry)+time)
        fused=torch.segment_reduce(tokens,reduce='mean',lengths=lengths)
        if unique is not None:fused=fused.index_select(0,cells)
        return fused.split(sizes)


def memory_raster_inputs(policy,head,plan,visual):
    """Reuse each immutable memory's geometry and calibrated splat stencil."""
    device=visual.device;tensor=lambda value:torch.as_tensor(value,device=device)
    maximum=policy.depth_encoder.max_depth_m
    cache=plan.setdefault('_encoding_inputs',{})
    key=(device,maximum,head.horizon,head.raster_padding)
    if key not in cache:
        points=tensor(plan['points'])[None];valid=torch.ones(points.shape[:2],device=device,dtype=torch.bool)
        suffix=torch.cat((points[0]/maximum,tensor(plan['depth'])[:,None]/maximum,
            tensor(plan['keys'][:,2])[:,None].float(),tensor(plan['origins'])/maximum),-1)
        selection=plan['selection'];transforms=inverse_rigid(plan['pose'])@selection['poses']
        stencil=memory_splat_stencil(points,valid,head.horizon,64,head.raster_padding)
        cache[key]=(suffix,tensor(plan['pose'])[None],tensor(selection['valid'])[None],
            tensor(transforms).float()[None],tensor(selection['ages'])[None],stencil)
    return cache[key]


def memory_rasters(policy,head,plans,visuals):
    """Splat independent memories in one batch, padding only their sparse cells."""
    inputs=[memory_raster_inputs(policy,head,p,v) for p,v in zip(plans,visuals,strict=True)]
    count=max(len(v) for v in visuals)
    raw=torch.stack([F.pad(torch.cat((visual,values[0]),-1),(0,0,0,count-len(visual)))
        for visual,values in zip(visuals,inputs,strict=True)])
    entries=[]
    for slot in range(4):
        indices,weights=zip(*(values[-1][0][slot] for values in inputs),strict=True)
        entries.append((torch.cat([F.pad(x,(0,0,0,count-x.shape[1])) for x in indices]),
            torch.cat([F.pad(w,(0,0,0,count-w.shape[1])) for w in weights])))
    mass=torch.cat([values[-1][1] for values in inputs])
    raster,mass=apply_memory_splat(raw,(entries,mass,head.raster_size))
    size=head.raster_size
    pose,selection_valid,transforms,ages=(torch.cat(values) for values in zip(*(x[1:-1] for x in inputs),strict=True))
    return raster.transpose(1,2).reshape(len(plans),-1,size,size),mass.transpose(1,2).reshape(len(plans),1,size,size),pose,selection_valid,transforms,ages


def memory_raster(policy,head,plan,visual):
    return memory_rasters(policy,head,[plan],[visual])


def encode_memories(policy,head,plans,visuals,goals,goal_present):
    """Same independent memory encodings, batched through the geometry network."""
    raster,mass,poses,valid,transforms,ages=memory_rasters(policy,head,plans,visuals)
    field,logits,configuration=head(raster,poses,mass)
    goal=torch.stack([torch.as_tensor(g,device=raster.device,dtype=torch.float32).reshape(2) for g in goals])
    present=torch.as_tensor(goal_present,device=raster.device,dtype=torch.bool)[:,None]
    goal=torch.where(present,goal,torch.zeros_like(goal))
    condition=policy.condition_encoder.encode_configuration(configuration,field,goal,valid,transforms,ages)
    return replace(condition,goal_present=present.float()),(field,logits)


def encode_memory(policy,head,plan,visual,goal,goal_present):
    """Online and replay use the same encoder; online supplies one memory."""
    return encode_memories(policy,head,[plan],[visual],[goal],[goal_present])
