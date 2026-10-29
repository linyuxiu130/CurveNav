from curvenav.training.convergence import cnn_plateau
import pytest


def test_freeze_requires_actual_learning_and_a_stable_validation_plateau():
    rows=lambda values:[dict(LH=value) for value in values]
    assert not cnn_plateau(rows([4.]*20))
    assert not cnn_plateau(rows([4.,3.,2.]+[2.]*6))
    assert cnn_plateau(rows([4.,3.,2.]+[2.]*7))
    assert not cnn_plateau(rows([4.,3.,2.]+[2.]*6+[3.]))
    assert not cnn_plateau(rows([4.,3.,2.]+[2.]*6+[1.8]))


def test_validation_reuses_current_features_only_inside_one_call_and_preserves_rng(monkeypatch):
    from types import SimpleNamespace
    import torch
    from curvenav.training import convergence as module,bspline_image_batch as images

    actor=SimpleNamespace(weight=torch.nn.Parameter(torch.tensor(2.)))
    calls=[]
    def features(actor,records):
        values=[r[0]['value'] for r in records];calls.append(values)
        return {value:actor.weight*value for value in values}
    def losses(actor,batch,*,cnn_cache):
        values=[r[0]['value'] for r in batch]
        assert all(not cnn_cache[value].requires_grad for value in values)
        torch.rand(1)
        value=torch.stack([cnn_cache[v] for v in values]).mean()
        return {k:value*(i+1) for i,k in enumerate(('CFM','TRAJ','GF','LH','LQ'))},None
    monkeypatch.setattr(module.dist,'all_reduce',lambda tensor,group:None)
    monkeypatch.setattr(images,'step_feature_maps',features)
    monkeypatch.setattr(module,'batched_losses',losses)
    rng=torch.get_rng_state()
    records=[(dict(trial_id=str(i%2),value=i),) for i in range(1,7)]
    first=module.validation_losses(actor,records,0,2,torch.device('cpu'),None)
    assert first==pytest.approx(dict(CFM=6.,TRAJ=12.,GF=18.,LH=24.,LQ=30.))
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():actor.weight.add_(1)
    second=module.validation_losses(actor,records,0,2,torch.device('cpu'),None)
    assert second['CFM']==pytest.approx(9.) and calls==[[1,3,5],[1,3,5]]


def test_paired_metrics_equalize_tasks_preserve_rng_and_share_fresh_maps(monkeypatch):
    from types import SimpleNamespace
    import torch
    from curvenav.training import convergence as module,bspline_image_batch as images
    actor=SimpleNamespace(weight=torch.nn.Parameter(torch.tensor(2.)))
    records=lambda trial,values:[(dict(trial_id=trial,value=v),) for v in values]
    train=records('a',[1])+records('b',[2,4,6])
    validation=records('a',[3,5])+records('b',[8])
    calls=[]
    def features(actor,rows):
        calls.append([r[0]['value'] for r in rows])
        return {r[0]['value']:actor.weight*r[0]['value'] for r in rows}
    def losses(actor,batch,*,cnn_cache):
        torch.randn(3)
        value=torch.stack([cnn_cache[r[0]['value']] for r in batch]).mean()
        return {k:value for k in ('CFM','TRAJ','GF','LH','LQ')},None
    monkeypatch.setattr(images,'step_feature_maps',features)
    monkeypatch.setattr(module,'batched_losses',losses)
    monkeypatch.setattr(module.dist,'all_reduce',lambda tensor,group:None)
    rng=torch.get_rng_state().clone()
    metrics=module.comparison_losses(actor,train,validation,0,1,torch.device('cpu'),None)
    assert metrics['train']['LH']==pytest.approx(5.) and metrics['validation']['LH']==pytest.approx(12.)
    assert torch.equal(rng,torch.get_rng_state()) and calls==[[1,2,4,6,3,5,8]]
    with torch.no_grad():actor.weight.add_(1)
    next_metrics=module.comparison_losses(actor,train,validation,0,1,torch.device('cpu'),None)
    assert next_metrics['train']['LH']==pytest.approx(7.5) and len(calls)==2


def test_uneven_rank_reduction_uses_global_task_row_counts(monkeypatch):
    from types import SimpleNamespace
    import torch
    from curvenav.training import convergence as module
    records=[(dict(trial_id=t,value=v),) for t,v in [('a',1),('b',2),('b',4),('b',6),('c',8)]]
    def losses(actor,batch,*,cnn_cache):
        value=torch.tensor([r[0]['value'] for r in batch],dtype=torch.float64).mean()
        return {k:value for k in ('CFM','TRAJ','GF','LH','LQ')},None
    partial=[]
    monkeypatch.setattr(module,'batched_losses',losses)
    monkeypatch.setattr(module.dist,'all_reduce',lambda tensor,group:partial.append(tensor.clone()))
    for rank in range(3):module.validation_losses(SimpleNamespace(),records,rank,3,torch.device('cpu'),None,{})
    reduced=torch.stack(partial).sum(0)
    torch.testing.assert_close(reduced[1:]/reduced[0],torch.full((5,),13/3,dtype=torch.float64))
