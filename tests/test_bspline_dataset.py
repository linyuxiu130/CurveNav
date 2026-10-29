"""Preserved real data/geometry/availability boundaries for native joint training."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from curvenav.data.bspline_replay import dense_curve
from curvenav.data.observed_geometry import clearance
from curvenav import build_policy
from test_policy import tiny_config


def test_expert_admission_batches_every_eligible_row_and_identifies_failures():
    from types import SimpleNamespace
    from curvenav.data.bspline_dataset import validate_expert_labels
    policy=torch.nn.Linear(1,1)
    policy.curve_codec=SimpleNamespace(decode_values=lambda values:(values[:,:2,None].expand(-1,-1,2),None))
    calls=[]
    class Utility:
        def __call__(self,paths,batch):
            assert not torch.is_grad_enabled()
            calls.append((len(paths),batch['task_states']))
            return SimpleNamespace(failed=paths[:,:,0,0]<0)
    utility,other=Utility(),Utility()
    def item(index):
        return dict(teacher_values=torch.ones(14),q_supervision_available=True,task_state=index,
            view=np.array([index,1.,0.]),goal=torch.tensor([1.,0.]),goal_present=True,
            target_field=torch.zeros(6,2,2),trial_id='K0',step=index)
    rows=[item(i) for i in range(65)]
    skipped=item(100);skipped['teacher_values']=None
    unavailable=item(101);unavailable['q_supervision_available']=False
    records=[(row,None,None,utility) for row in rows+[skipped,unavailable]]+[(item(66),None,None,other)]
    validate_expert_labels(SimpleNamespace(policy=policy),records)
    assert [count for count,_ in calls]==[32,32,1,1]
    assert [state for _,states in calls for state in states]==list(range(65))+[66]
    rows[37]['teacher_values'][0]=-1
    with pytest.raises(ValueError,match='K0 step 37'):
        validate_expert_labels(SimpleNamespace(policy=policy),records)

def test_dense_native_curve_bound_and_exact_segment_box_contacts():
    codec = build_policy(tiny_config()).curve_codec
    values = torch.zeros(1, 14)
    values[0, 0::2] = torch.linspace(.1, .7, 7)
    path, pad = dense_curve(values)
    assert path.shape == (1, 257, 2)
    assert torch.equal(path[:, 0], torch.zeros(1, 2))
    torch.testing.assert_close(path[:, -1], values.reshape(1, 7, 2)[:, -1], rtol=0, atol=0)
    assert pad.item() > 0
    boxes = np.array([[0., 0., 1., 1.]], dtype=np.float64)
    crossing = np.array([[[-1., .5], [2., .5]]])
    separated = np.array([[[-1., 2.], [2., 2.]]])
    assert clearance(crossing, boxes, radius=.2)[0] == -.2
    assert clearance(separated, boxes, radius=.2)[0] == .8

def test_expert_pool_exact_pose_source_and_float32_deduplication():
    from curvenav.data.bspline_dataset import attach_q_expert_pools
    def item(value, source='grid', x=0.):
        return dict(teacher_values=torch.full((14,),value),source_grid_sha256=source,
            view=np.array([x,2.,3.],dtype=np.float64),raw_sha256='raw',acquisitions=10,step=0)
    a,b,duplicate,c,d = item(.1),item(.2),item(.1),item(.3,source='other'),item(.4,x=np.nextafter(0.,1.))
    records=[(x,None,None,None) for x in (a,b,duplicate,c,d)]
    proof=attach_q_expert_pools(records)
    assert len(proof)==3 and len(proof[0]['teacher_float32_sha256'])==2
    torch.testing.assert_close(a['q_expert_values'],torch.stack((a['teacher_values'],b['teacher_values'])),rtol=0,atol=0)
    torch.testing.assert_close(b['q_expert_values'],torch.stack((b['teacher_values'],a['teacher_values'])),rtol=0,atol=0)
    assert c['q_expert_values'].shape==d['q_expert_values'].shape==(1,14)

def test_sampler_cycles_every_state_without_history_weight_and_keeps_case_balance():
    from collections import Counter
    from curvenav.data.bspline_dataset import case_groups, sample_index, SAMPLING
    items=[]
    for trial,history,count in [('D2','old',5),('D2','new',18),('K','other',3)]:
        items.extend((dict(trial_id=trial,raw_sha256=history),{},None,None) for _ in range(count))
    groups=case_groups(items,range(len(items)))
    selected=[sample_index(groups,p) for p in range(46)]
    assert selected[::2]==list(range(23))
    assert set(selected)==set(range(len(items)))
    assert Counter(items[i][0]['trial_id'] for i in selected)=={'D2':23,'K':23}


def test_partial_q_pool_may_use_compatible_expert_but_not_for_cfm():
    from curvenav.data.bspline_dataset import attach_q_expert_pools
    def item(teacher,source):
        return dict(teacher_values=teacher,source_grid_sha256=source,view=[0.,1.,2.],raw_sha256='raw',step=0,acquisitions=10)
    partial=item(None,'a');expert=item(torch.ones(14),'a');isolated=item(None,'b')
    attach_q_expert_pools([(x,None,None,None) for x in (partial,expert,isolated)])
    assert partial['teacher_values'] is None and partial['q_expert_values'].shape==(1,14)
    assert isolated['teacher_values'] is None and isolated['q_expert_values'].shape==(0,14)

def test_source_utility_availability_uses_real_cells_and_disconnected_start():
    from curvenav.data.privileged import SourceConfigurationGrid,SourceConfigurationSpaceQuery
    from curvenav.data.bspline_teachers import source_utility_availability
    from curvenav.data.goal_distance import GoalDistanceQuery
    grid=np.ones((5,5),dtype=np.float32);grid[1,1]=-.05
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(grid,np.zeros(2),1.,'canonical_world_xy'),))
    distance=GoalDistanceQuery(query)
    defined,evidence=source_utility_availability(distance,np.array([1.5,1.5,0.]),torch.tensor([2.,2.]))
    assert not defined and evidence['start_cell']==[1,1] and evidence['fixed_distances_m'][0] is None
    graph=distance.graphs[0];field=next(iter(distance.fields.values()))
    assert source_utility_availability(distance,np.array([1.5,1.5,0.]),torch.tensor([2.,2.]))==(defined,evidence)
    assert distance.graphs[0] is graph and next(iter(distance.fields.values())) is field
    defined,evidence=source_utility_availability(distance,np.array([.5,.5,0.]),torch.tensor([3.,3.]))
    assert defined and evidence['fixed_distances_m'][0]>0

def test_native_geometry_cpu_and_missing_contract_rejected_at_boundary(tmp_path):
    from curvenav.data.bspline_replay import require_native_geometry
    from curvenav.data.bspline_dataset import load_items
    from types import SimpleNamespace
    import json
    with pytest.raises(ValueError,match='requires CUDA native projection'):require_native_geometry('cpu')
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(complete=True,split='development')))
    with pytest.raises(ValueError,match='rebuild replay memory'):
        load_items([path],None)
    from curvenav.data.six_frame_memory import MEMORY_CONTRACT
    path.write_text(json.dumps(dict(complete=True,split='development',input_contract=MEMORY_CONTRACT)))
    with pytest.raises(ValueError,match='requires geometry_input_contract'):
        load_items([path],None)
    path.write_text(json.dumps(dict(complete=True,split='development',input_contract=MEMORY_CONTRACT,
        geometry_input_contract='cuda_native_projection_v1',geometry_label_contract='observed_distance_raw_spatial_gradient_v1')))
    with pytest.raises(ValueError,match='rebuild geometry labels'):
        load_items([path],None)


def test_parallel_reading_checks_row_digest_before_deserialization(tmp_path):
    import json
    from curvenav.data.bspline_dataset import load_items
    from curvenav.data.six_frame_memory import MEMORY_CONTRACT
    from curvenav.data.observed_geometry import GEOMETRY_LABEL_CONTRACT
    row=tmp_path/'row.pt';row.write_bytes(b'invalid pickle must never be deserialized')
    manifest=dict(complete=True,split='train',input_contract=MEMORY_CONTRACT,
        geometry_input_contract='cuda_native_projection_v1',geometry_label_contract=GEOMETRY_LABEL_CONTRACT,
        quality=dict(passed=True),source_kind='omni_teacher40_continuous_geometry_v3',trial_id='K0',
        items=[dict(path=str(row),sha256='0'*64)])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):load_items([path],None)

def test_parallel_tree_reads_verify_index_digest_before_unpickling(tmp_path,monkeypatch):
    import json
    from curvenav.data import bspline_dataset as dataset
    from curvenav.data.six_frame_memory import MEMORY_CONTRACT
    from curvenav.data.observed_geometry import GEOMETRY_LABEL_CONTRACT
    dep=tmp_path/'dependency';dep.write_bytes(b'invalid pickle')
    item={key:str(dep) for key in ('raw_path','index_path','memory_snapshot_path','source_grid')}
    item.update(index_sha256='0'*64,critic_replay_provenance=dict(result_path=str(dep),candidates_path=str(dep)))
    row=tmp_path/'row.pt';torch.save(item,row)
    manifest=dict(complete=True,split='train',input_contract=MEMORY_CONTRACT,
        geometry_input_contract='cuda_native_projection_v1',geometry_label_contract=GEOMETRY_LABEL_CONTRACT,
        quality=dict(passed=True),source_kind='omni_teacher40_continuous_geometry_v3',trial_id='K0',
        items=[dict(path=str(row),sha256=dataset.sha(row))])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    def forbidden(_):raise RuntimeError('deserialization ran before integrity admission')
    monkeypatch.setattr(dataset,'_load_tree',forbidden)
    with pytest.raises(AssertionError):dataset.load_items([path],None)

def test_teacher_availability_schema_rejects_unknown_and_preserves_partial():
    from curvenav.data.bspline_dataset import validate_teacher_schema
    qualified=dict(teacher_values=torch.ones(14),teacher_checks={'safe':True},teacher_certificate={'qualified':True})
    validate_teacher_schema(qualified)
    with pytest.raises(ValueError,match='Unsupported replay label_schema'):
        validate_teacher_schema(dict(qualified,label_schema='unknown'))
    partial=dict(label_schema='loss_availability_v1',loss_availability=dict(CFM=False,GF=True,LH=True,LQ=True,own_expert=False),teacher_values=None,teacher_checks=None,teacher_certificate=None,teacher_rejection={'reason':'mpc'})
    validate_teacher_schema(partial)
    bad=deepcopy(partial);bad['loss_availability']['GF']=False
    with pytest.raises(ValueError,match='requires GF/LH'):validate_teacher_schema(bad)
    undefined=deepcopy(partial);undefined['loss_availability']['LQ']=False;undefined['q_availability']=dict(defined=False,reason='source_start_occupied')
    validate_teacher_schema(undefined)


def test_recorded_bank_rejects_shape_source_and_float32_sha_changes():
    import hashlib
    from curvenav.data.bspline_dataset import validate_recorded_bank
    paths=torch.zeros(16,64,2);view=np.array([1.,2.,3.],dtype=np.float64)
    item=dict(critic_replay_paths=paths,raw_sha256='raw',step=4,acquisitions=50,view=view,critic_replay_provenance=dict(raw_sha256='raw',decision_step=4,acquisitions=50,view_float64_hex=view.tobytes().hex(),native_paths_float32_sha256=hashlib.sha256(paths.numpy().tobytes()).hexdigest()))
    validate_recorded_bank(item)
    for changed in (dict(item,critic_replay_paths=paths.double()),dict(item,step=5),dict(item,critic_replay_paths=paths+1)):
        with pytest.raises(ValueError):validate_recorded_bank(changed)


def test_generated_teacher_pool_requires_same_history_and_goal():
    from curvenav.data.bspline_dataset import attach_trajectory_teacher_pools
    def item(value,raw='a',goal_present=False,goal=0.,plan='p'):
        return dict(raw_sha256=raw,acquisitions=10,memory_plan_sha256=plan,view=np.array([1.,2.,0.]),
            goal_present=goal_present,goal=torch.tensor([goal,0.]),
            teacher_values=None if value is None else torch.full((14,),value))
    rows=[item(1.),item(2.),item(3.,raw='b'),item(4.,goal_present=True,goal=3.),item(None),item(5.,plan='different')]
    attach_trajectory_teacher_pools([(row,) for row in rows])
    assert rows[0]['trajectory_teacher_values'][:,0].tolist()==[1.,2.]
    assert rows[1]['trajectory_teacher_values'][:,0].tolist()==[1.,2.]
    assert rows[2]['trajectory_teacher_values'][:,0].tolist()==[3.]
    assert rows[3]['trajectory_teacher_values'][:,0].tolist()==[4.]
    assert rows[4]['trajectory_teacher_values'].shape==(0,14)
    assert rows[5]['trajectory_teacher_values'][:,0].tolist()==[5.]


def test_reference_fit_preserves_backwards_tangent_and_endpoint():
    from curvenav.trajectory.control_points import project_reference_tangent
    codec = build_policy(tiny_config()).curve_codec
    reference = torch.stack((-torch.linspace(0, 2, 64), torch.zeros(64)), -1)[None]
    values, path = project_reference_tangent(codec, reference)
    assert values[0, 0] < 0
    torch.testing.assert_close(path, reference, atol=1e-5, rtol=1e-5)


def test_dataset_entry_binds_admission_and_original_validation(tmp_path):
    import json
    from curvenav.data.bspline_dataset import dataset_manifests
    from curvenav.data.bspline_replay import sha
    admission=tmp_path/'admission.json';admission.write_text('{"complete":true}')
    validation=tmp_path/'validation.json';validation.write_text('{"items":[]}')
    frozen=tmp_path/'frozen.json';frozen.write_text(json.dumps(dict(manifests=[str(validation)],manifest_sha256={str(validation):sha(validation)})))
    summary=dict(complete=True,admission=str(admission),admission_sha256=sha(admission),
        frozen_validation=str(frozen),frozen_validation_sha256=sha(frozen),
        manifests=dict(train=['train.json'],validation=[str(validation)]),rows=dict(train=3019,validation=194))
    entry=tmp_path/'dataset_summary.json';entry.write_text(json.dumps(summary))
    paths,rows=dataset_manifests(entry)
    assert paths['validation']==[validation] and rows==summary['rows']
    validation.write_text('{"items":["changed"]}')
    with pytest.raises(ValueError,match='Original validation manifest SHA'):dataset_manifests(entry)
    validation.write_text('{"items":[]}');summary['manifests']['validation']=[];entry.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='differ from the original'):dataset_manifests(entry)
