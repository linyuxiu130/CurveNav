"""Native B-spline replay admission, expert pools and task-balanced state sampling."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,pickle
import numpy as np
import torch
from curvenav.data.bspline_replay import sha
from curvenav.data.observed_geometry import GEOMETRY_LABEL_CONTRACT
from curvenav.data.fixed40_targets import task_state_from_history
from curvenav.data.six_frame_memory import MEMORY_CONTRACT
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.training.critic import RouteUtilityTeacher
from curvenav.data.bspline_teachers import source_utility_availability

SAMPLING = 'uniform_trial_all_state_cycle_v2'


def dataset_manifests(path):
    """Read the admitted dataset and its immutable original validation binding."""
    summary=json.loads(Path(path).read_text())
    if not summary['complete'] or sha(summary['admission'])!=summary['admission_sha256']:
        raise ValueError('Dataset admission is incomplete or its SHA differs')
    if sha(summary['frozen_validation'])!=summary['frozen_validation_sha256']:
        raise ValueError('Frozen validation binding SHA differs')
    frozen=json.loads(Path(summary['frozen_validation']).read_text())
    if summary['manifests']['validation']!=frozen['manifests']:
        raise ValueError('Dataset validation manifests differ from the original validation')
    for filename,digest in frozen['manifest_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Original validation manifest SHA differs')
    return {part:[Path(p) for p in paths] for part,paths in summary['manifests'].items()},summary['rows']

def case_groups(items, indices):
    """Equal task exposure, cycling all states in the original raw-history insertion order."""
    cases = {}
    for index in indices:
        item, raw, *_ = items[index]
        key=item['trial_id']
        cases.setdefault(key, {}).setdefault(item['raw_sha256'], []).append(index)
    return [[[i for states in histories.values() for i in states]] for histories in cases.values()]

def sample_index(groups, position):
    states = groups[position % len(groups)][0]
    visit = position // len(groups)
    return states[visit % len(states)]

def attach_q_expert_pools(items):
    """Group admitted experts by source geometry and exact physical capture pose."""
    groups = {}
    for item, *_ in items:
        key = (item['source_grid_sha256'], np.asarray(item['view'], dtype=np.float64).tobytes())
        groups.setdefault(key, []).append(item)
    provenance = []
    for (grid_sha, view_bytes), members in groups.items():
        experts = {}
        for item in members:
            if item['teacher_values'] is None:
                continue
            values = item['teacher_values'].float().contiguous()
            experts.setdefault(values.numpy().tobytes(), values)
        for item in members:
            own = None if item['teacher_values'] is None else item['teacher_values'].float().contiguous().numpy().tobytes()
            ordered = ([own] if own is not None else []) + [key for key in experts if key != own]
            item['q_expert_values'] = torch.stack([experts[key] for key in ordered]) if ordered else torch.empty(0, 14)
        provenance.append(dict(source_grid_sha256=grid_sha, view_float64_hex=view_bytes.hex(), teacher_float32_sha256=[hashlib.sha256(key).hexdigest() for key in experts], members=[dict(trial_id=item.get('trial_id'), raw_sha256=item['raw_sha256'], acquisitions=item['acquisitions'], step=item['step']) for item in members]))
    return provenance

def attach_trajectory_teacher_pools(items):
    """Only identical causal history, pose and policy goal may share trajectory teachers.

    Q's source/pose pool can contain different histories or missions and is not a G target.
    """
    groups = {}
    for item, *_ in items:
        key = (item['raw_sha256'], item['acquisitions'], item['memory_plan_sha256'],
               np.asarray(item['view'], dtype=np.float64).tobytes(), item['goal_present'],
               item['goal'].float().contiguous().numpy().tobytes())
        groups.setdefault(key, []).append(item)
    for members in groups.values():
        teachers = {}
        for item in members:
            if item['teacher_values'] is not None:
                value = item['teacher_values'].float().contiguous()
                teachers.setdefault(value.numpy().tobytes(), value)
        for item in members:
            # Missing/failed teacher qualification must not be bypassed by a different row.
            item['trajectory_teacher_values'] = (torch.stack(list(teachers.values()))
                if item['teacher_values'] is not None else torch.empty(0, 14))


def validate_teacher_schema(item):
    schema = item.get('label_schema', 'qualified_only')
    if schema not in ('qualified_only', 'loss_availability_v1'):
        raise ValueError(f'Unsupported replay label_schema: {schema}')
    if schema == 'loss_availability_v1':
        availability = item['loss_availability']
        if not isinstance(availability['LQ'], bool) or availability != dict(CFM=False, GF=True, LH=True, LQ=availability['LQ'], own_expert=False):
            raise ValueError('loss_availability_v1 requires GF/LH; only CFM/own expert missing and source-defined LQ boolean are supported')
        if not availability['LQ']:
            assert not item['q_availability']['defined'] and item['q_availability']['reason']
        assert item['teacher_values'] is None and item['teacher_checks'] is None and (item['teacher_certificate'] is None)
        assert item['teacher_rejection']
    else:
        assert item['teacher_values'].shape == (14,) and all(item['teacher_checks'].values())
        assert item['teacher_certificate']['qualified']

def validate_recorded_bank(item):
    paths = item['critic_replay_paths']
    if paths.dtype != torch.float32 or (paths.ndim != 3 or paths.shape[0] != 16 or paths.shape[1:] != (64, 2)) or (not torch.isfinite(paths).all()):
        raise ValueError('Native replay requires finite original FP32 native16 paths')
    binding = item['critic_replay_provenance']
    if binding['raw_sha256'] != item['raw_sha256'] or binding['decision_step'] != item['step'] or binding['acquisitions'] != item['acquisitions'] or (binding['view_float64_hex'] != np.asarray(item['view'], dtype=np.float64).tobytes().hex()):
        raise ValueError('Recorded bank source/step differs from replay state')
    if binding['native_paths_float32_sha256'] != hashlib.sha256(paths.contiguous().numpy().tobytes()).hexdigest():
        raise ValueError('Recorded bank path SHA differs from bound original')

def _load_tree(path):
    with Path(path).open('rb') as stream:return pickle.load(stream)['tree']


@torch.no_grad()
def validate_expert_labels(actor,records):
    """Check every admitted expert, batching states that share one source oracle."""
    device=next(actor.policy.parameters()).device;groups={}
    for item,_,_,utility in records:
        if item['teacher_values'] is not None and item['q_supervision_available']:
            groups.setdefault(utility,[]).append(item)
    for utility,items in groups.items():
        for start in range(0,len(items),32):
            rows=items[start:start+32];count=len(rows)
            expert,_=actor.policy.curve_codec.decode_values(torch.stack([item['teacher_values'] for item in rows]).to(device))
            label=utility(expert[:,None],dict(task_states=[item['task_state'] for item in rows],
                source_grid_index=torch.zeros(count,dtype=torch.long,device=device),
                source_origin_xy=torch.as_tensor(np.stack([item['view'][:2] for item in rows]),dtype=torch.float64,device=device),
                source_yaw_rad=torch.tensor([item['view'][2] for item in rows],dtype=torch.float64,device=device),
                point_goal=torch.stack([item['goal'] for item in rows]).to(device),
                goal_present=torch.tensor([item['goal_present'] for item in rows],device=device),
                observed_field=torch.stack([item['target_field'] for item in rows]).to(device)))
            for item,failed in zip(rows,label.failed[:,0].cpu().tolist(),strict=True):
                if failed:raise ValueError(f"Expert contradicts Q physical/task labels: {item['trial_id']} step {item['step']}")


def load_items(manifest_paths, actor):
    raw_cache, utility_cache, files, items = ({}, {}, {}, [])

    def bind(path):
        key = str(Path(path).resolve())
        if key not in files:
            files[key] = sha(path)
        return files[key]
    manifests, case_rows = ([], {})
    for path in manifest_paths:
        manifest = json.loads(path.read_text())
        assert manifest['complete'] and manifest['split'] in ('train', 'development')
        if manifest.get('input_contract') != MEMORY_CONTRACT:
            raise ValueError('rebuild replay memory with curvenav.data.prepare_six_frame_replay before training')
        if manifest.get('geometry_input_contract') != 'cuda_native_projection_v1':
            raise ValueError('Native replay requires geometry_input_contract=cuda_native_projection_v1')
        if manifest.get('geometry_label_contract') != GEOMETRY_LABEL_CONTRACT:
            raise ValueError('rebuild geometry labels from recorded observed hits before training')
        if not manifest['quality']['passed']:raise ValueError('Teacher execution or planning-state coverage did not pass admission')
        manifests.append(bind(path))
        if manifest['source_kind'] == 'omni_teacher40_continuous_geometry_v3':
            case_rows.setdefault(manifest['trial_id'], 0)
        entries=[];dependencies=set()
        for record in manifest['items']:
            assert bind(record['path']) == record['sha256']
            item = torch.load(record['path'], map_location='cpu', weights_only=False)
            entries.append(item)
            dependencies.update(item[key] for key in ('raw_path','index_path','memory_snapshot_path','source_grid'))
            binding=item['critic_replay_provenance']
            dependencies.update(binding[key] for key in ('result_path','candidates_path'))
        pending=sorted({str(Path(p).resolve()) for p in dependencies}-files.keys())
        with ThreadPoolExecutor(max_workers=4) as pool:
            files.update(zip(pending,pool.map(sha,pending),strict=True))
            for item in entries:assert bind(item['index_path'])==item['index_sha256']
            trees=list(pool.map(_load_tree,[item['index_path'] for item in entries]))
        for item,tree in zip(entries,trees,strict=True):
            assert item['geometry_label_contract']==manifest['geometry_label_contract']
            validate_teacher_schema(item)
            validate_recorded_bank(item)
            if not item['goal_present']:
                assert not item['goal'].any()
            if 'source_grid_frame' not in item:
                raise ValueError('replay item must declare source_grid_frame; regenerate old unmarked XY grids')
            if manifest['source_kind'] == 'omni_teacher40_continuous_geometry_v3':
                assert item['trial_id'] == manifest['trial_id'] and 'no_goal_teacher' in item
                case_rows[item['trial_id']] += 1
            else:
                raise ValueError('Native replay requires reissued task labels or native goal retention with correct grid metadata')
            sidecar = item['no_goal_teacher']
            if sidecar is not None:
                assert not sidecar['goal_present'] and (not sidecar['goal'].any())
                assert sidecar['teacher_values'].shape == (14,) and all(sidecar['teacher_checks'].values())
                assert sidecar['teacher_certificate']['qualified']
            assert bind(item['raw_path']) == item['raw_sha256']
            binding = item['critic_replay_provenance']
            for file_key, hash_key in (('result_path', 'result_sha256'), ('candidates_path', 'candidates_sha256')):
                assert bind(binding[file_key]) == binding[hash_key]
            if bind(item['memory_snapshot_path'])!=item['memory_plan_sha256']:
                raise ValueError('memory plan SHA mismatch')
            assert bind(item['source_grid']) == item['source_grid_sha256']
            if item['raw_path'] not in raw_cache:
                raw_cache[item['raw_path']] = torch.load(item['raw_path'], map_location='cpu', weights_only=False)
            raw=raw_cache[item['raw_path']]
            if raw['task']['panel']=='D2' and manifest['quality'].get('executed_steps')!=raw['task']['budget']:
                raise ValueError('D2 training requires teacher execution through the original full evaluation budget')
            if raw['task']['panel']=='D2':
                evidence=raw['task_evidence']
                assert bind(evidence['reference_path'])==evidence['reference_sha256']
            item['task_state']=task_state_from_history(raw,item['acquisitions'])
            item['memory_plan']=torch.load(item['memory_snapshot_path'],map_location='cpu',weights_only=False)
            plan=item['memory_plan']
            if plan['contract']!=MEMORY_CONTRACT or plan['raw_sha256']!=item['raw_sha256'] or plan['selection']['indices'][-1]!=item['acquisitions']-1:
                raise ValueError('memory plan contract or causal cutoff differs from replay row')
            if (plan['frame_ids']<0).any() or (plan['frame_ids']>=item['acquisitions']).any():
                raise ValueError('memory plan refers to an unavailable or future observation')
            raw_provenance = raw_cache[item['raw_path']]['provenance']
            assert raw_provenance['split']==manifest['split'] and raw_provenance['source_kind']==manifest['source_kind']
            if item['source_grid'] not in utility_cache:
                utility_cache[item['source_grid']] = RouteUtilityTeacher(SourceConfigurationSpaceQuery.from_paths((Path(item['source_grid']),)), actor.policy.planning_horizon_m, actor.head.radius)
            query = utility_cache[item['source_grid']].source_query
            item['q_supervision_available'], item['current_q_availability'] = source_utility_availability(
                utility_cache[item['source_grid']].goal_distance, item['view'], item['goal'])
            grid_frame = query.grids[0].coordinate_frame
            if item['source_grid_frame'] != grid_frame:
                raise ValueError('replay item source_grid_frame differs from loaded source grid')
            if grid_frame != 'canonical_world_xy':
                raise ValueError('Native replay requires explicit canonical_world_xy source grids')
            items.append((item, raw_cache[item['raw_path']], tree, utility_cache[item['source_grid']]))
    validate_expert_labels(actor,items)
    q_expert_pools = attach_q_expert_pools(items)
    attach_trajectory_teacher_pools(items)
    device=next(actor.policy.parameters()).device
    for item,*_ in items:item['target_field']=item['target_field'].to(device)
    goal = [i for i, (item, *_) in enumerate(items) if item['goal_present']]
    memory = [i for i, (item, *_) in enumerate(items) if not item['goal_present']]
    assert goal and memory, 'Require native14 goal retention and memory labels, never timed24 targets'
    return (items, goal, memory, dict(geometry_input_contract='cuda_native_projection_v1', manifests=manifests, files=files, case_rows=case_rows, q_expert_pools=q_expert_pools, critic_replay_contract='native_recorded_paths_fp32_v2', label_availability_schema='loss_availability_v1' if any((item.get('label_schema') == 'loss_availability_v1' for item, *_ in items)) else 'qualified_only'))
