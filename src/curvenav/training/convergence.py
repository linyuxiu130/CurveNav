"""Deterministic, equal-task monitoring of training and held-out states."""
from collections import Counter,defaultdict
import torch
import torch.distributed as dist
from curvenav.training.bspline_objective import batched_losses


@torch.no_grad()
def validation_losses(actor,records,rank,world,device,control,cnn_cache=None):
    totals=torch.zeros(6,dtype=torch.float64,device=device)
    local=records[rank::world]
    counts=Counter(record[0]['trial_id'] for record in records)
    tasks=defaultdict(list)
    for record in local:tasks[record[0]['trial_id']].append(record)
    # Validation must not advance the training flow/noise RNG stream.
    with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
        torch.manual_seed(20261028+rank)
        if cnn_cache is None:
            from curvenav.training.bspline_image_batch import step_feature_maps
            cnn_cache=step_feature_maps(actor,local)
        for trial,rows in tasks.items():
            for start in range(0,len(rows),2):
                batch=rows[start:start+2]
                terms,_=batched_losses(actor,batch,cnn_cache=cnn_cache)
                weight=len(batch)/(len(counts)*counts[trial])
                totals[0]+=weight
                totals[1:]+=torch.stack([terms[k].detach().double() for k in ('CFM','TRAJ','GF','LH','LQ')])*weight
    totals=totals.cpu()
    dist.all_reduce(totals,group=control)
    return dict(zip(('CFM','TRAJ','GF','LH','LQ'),(totals[1:]/totals[0]).tolist(),strict=True))


@torch.no_grad()
def comparison_losses(actor,training,validation,rank,world,device,control,cnn_cache=None):
    """Reuse current-weight image maps for both splits in this one comparison."""
    if cnn_cache is None:
        from curvenav.training.bspline_image_batch import step_feature_maps
        cnn_cache=step_feature_maps(actor,training[rank::world]+validation[rank::world])
    return {name:validation_losses(actor,records,rank,world,device,control,cnn_cache)
        for name,records in (('train',training),('validation',validation))}


def cnn_plateau(history):
    """Require learning, then five epochs without a 1% LH improvement after epoch ten."""
    if len(history)<10:return False
    losses=[row['LH'] for row in history]
    best=losses[0];last_improvement=0
    for index,value in enumerate(losses[1:],1):
        if value<best*.99:best=value;last_improvement=index
    return best<losses[0]*.8 and len(losses)-1-last_improvement>=5 and losses[-1]<=best*1.05
