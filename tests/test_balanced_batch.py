from collections import Counter
import random

import numpy as np
import pytest
import torch

from curvenav.training.bspline_replay import balanced_batch_indices,batch_indices,rank_batch_sizes,epoch_budget


@pytest.mark.parametrize('world', [7, 8])
@pytest.mark.parametrize('batch', [2, 6, 22])
def test_fixed_uniform_task_multiset_rng_and_rank_capacity(batch,world):
    half = batch // 2
    goals = list(range(world * half))
    memories = list(range(world * half, 2 * world * half))
    goal_cases = [[[index]] for index in goals]
    memory_cases = [[[index]] for index in memories]
    source_keys = [frozenset({('raw', index * 3 + frame)
                              for frame in range((index % 7) + 1)})
                   for index in range(2 * len(goals))]

    original = [index for rank in range(world)
                for index in batch_indices(goal_cases,memory_cases,3,5,rank,world,batch)]
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    assigned = [balanced_batch_indices(
        goal_cases, memory_cases, 3, 5, rank, world, batch, source_keys,
    ) for rank in range(world)]
    assert random.getstate() == py_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert Counter(index for rank_batch in assigned for index in rank_batch) == Counter(original)
    assert all(len(rank_batch) == batch for rank_batch in assigned)
    assert len(set(original))==len(original)


def test_seven_ranks_preserve_global176_samples_and_mean_gradient():
    goals=[[[i,i+20]] for i in range(20)]
    memories=[[[40+i,60+i,80+i]] for i in range(20)]
    keys=[frozenset(('raw',j) for j in range(i%13+1)) for i in range(100)]
    old=[i for rank in range(8) for i in batch_indices(goals,memories,3,5,rank,8,22)]
    assigned=[balanced_batch_indices(goals,memories,3,5,rank,7,26,keys,global_batch=176) for rank in range(7)]
    assert rank_batch_sizes(7,26,176)==[26,25,25,25,25,25,25]
    assert Counter(i for batch in assigned for i in batch)==Counter(old)
    for batch,size in zip(assigned,rank_batch_sizes(7,26,176),strict=True):
        assert len(batch)==size
    weight=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    expected=torch.autograd.grad((weight*torch.tensor(old,dtype=torch.float64)-1).square().mean(),weight)[0]
    gradients=[]
    for batch in assigned:
        loss=(weight*torch.tensor(batch,dtype=torch.float64)-1).square().mean()*(7*len(batch)/176)
        gradients.append(torch.autograd.grad(loss,weight)[0])
    torch.testing.assert_close(torch.stack(gradients).mean(),expected,rtol=1e-12,atol=1e-12)


def test_global_batch_keeps_epoch_and_lr_budget_when_gpu_count_changes():
    from test_policy import tiny_config
    cfg=tiny_config().training
    goals=[[[0]*15] for _ in range(16)];memories=[[[0]*43] for _ in range(24)]
    assert epoch_budget(goals,memories,7,26,cfg,1e-5,global_batch=176)==epoch_budget(goals,memories,8,22,cfg,1e-5)


def test_source_assignment_reuses_overlapping_histories_before_adding_frames():
    rng=np.random.default_rng(49)
    starts=rng.integers(0,12,24);sizes=rng.integers(2,10,24)
    keys=[frozenset(range(start,start+size)) for start,size in zip(starts,sizes,strict=True)]
    goals=[[[i]] for i in range(12)];memories=[[[12+i]] for i in range(12)]
    batches=[balanced_batch_indices(goals,memories,0,0,rank,3,8,keys) for rank in range(3)]
    assert Counter(i for batch in batches for i in batch)==Counter(range(24))
    work=[len(set().union(*(keys[i] for i in batch))) for batch in batches]
    assert sum(work)<=33 and max(work)<=15


def test_forty_tasks_give_goal40_memory60_and_cover_longest_history():
    from test_policy import tiny_config
    goals=[[[i]] for i in range(16)]
    memories=[[[16+i]] for i in range(24)]
    selected=[i for step in range(5) for rank in range(8)
        for i in batch_indices(goals,memories,step,0,rank,8,22)]
    assert Counter(selected)==Counter({i:22 for i in range(40)})
    assert sum(i<16 for i in selected)/len(selected)==.4
    memories[-1]=[list(range(40,522))]
    budget=epoch_budget(goals,memories,8,22,tiny_config().training,1e-5)
    assert budget['steps_per_epoch']==110 and budget['total_steps']==5500
    visited={i for step in range(110) for rank in range(8)
        for i in batch_indices(goals,memories,step,0,rank,8,22)}
    assert visited==set(range(39))|set(range(40,522))
