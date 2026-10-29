"""Deduplicate full-image inputs, then encode same-shaped frames in GPU batches."""
import numpy as np
import torch
from itertools import groupby
from curvenav.deployment.streaming_memory import source_groups


def step_feature_maps(actor,records,cache=None,*,cnn_batch_size=64):
    if cnn_batch_size<1:
        raise ValueError('cnn_batch_size must be positive')
    device=next(actor.policy.parameters()).device
    if cache is not None:
        keys={(item['raw_sha256'],int(index)) for item,*_ in records for index,_ in source_groups(item['memory_plan'])}
        return {key:cache[key].to(device) for key in keys}
    inputs={}
    for item,raw,_,_ in records:
        key=item['raw_sha256']
        if key not in inputs:
            depths=raw.setdefault('_device_depths',{})
            if device not in depths:
                depths[device]=torch.as_tensor(np.asarray(raw['depths'],dtype=np.float32),device=device)[:,None]
            inputs[key]=(depths[device],{})
        inputs[key][1].update((int(index),None) for index,_ in source_groups(item['memory_plan']))
    groups={}
    for key,(depth,indices) in inputs.items():
        groups.setdefault(tuple(depth.shape[1:]),[]).extend((key,index) for index in indices)
    maps={}
    for keys in groups.values():
        for start in range(0,len(keys),cnn_batch_size):
            chunk=keys[start:start+cnn_batch_size]
            parts=[]
            for key,members in groupby(chunk,key=lambda pair:pair[0]):
                indices=torch.tensor([index for _,index in members],device=device)
                parts.append(inputs[key][0].index_select(0,indices))
            depth=torch.cat(parts)
            if len(chunk)<cnn_batch_size:
                depth=torch.cat((depth,depth[:1].expand(cnn_batch_size-len(chunk),-1,-1,-1)))
            features=actor.policy.depth_encoder.image_features(depth)[:len(chunk)]
            maps.update(zip(chunk,features.split(1),strict=True))
    return maps
