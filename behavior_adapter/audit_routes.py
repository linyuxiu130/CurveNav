"""Independently verify saved BEHAVIOR expert routes in 3D and task space."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


TASK_TEMPLATES = Path('/mnt/dataset/Benchmark/Behavior2026/2026-challenge-demos/primitive_task_templates_context_v3_20260807')


def audit_record(task_root, record):
    errors = []
    directory = task_root / record['route_directory']
    with np.load(directory/'trajectory.npz') as route, np.load(directory/'map.npz') as grid:
        positions = route['position_world_m']
        rotations = Rotation.from_quat(route['quaternion_xyzw']).as_euler('xyz')
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            errors.append('nonfinite_pose')
        expected_z = float(grid['expected_base_z_m'])
        if np.max(np.abs(positions[:, 2] - expected_z)) > .08 + 1e-4:
            errors.append('unsupported_height')
        if np.max(np.abs(rotations[:, :2])) > np.deg2rad(10) + 1e-4:
            errors.append('tipped_posture')
        pixels = np.rint((positions[:, :2]-grid['origin_xy'])/float(grid['resolution_m'])).astype(int)
        inside = ((pixels >= 0) & (pixels < np.asarray(grid['free'].shape[::-1]))).all(axis=1)
        if not inside.all():
            errors.append('outside_map')
        else:
            y, x = pixels[:, 1], pixels[:, 0]
            if not grid['physical_support'][y, x].all():
                errors.append('unsupported_xy')
            if np.min(grid['clearance_m'][y, x]) < .10 - 1e-5:
                errors.append('insufficient_clearance')
        if record['goal_type'] == 'task_target':
            target = record.get('target')
            if not target or not target.get('scope_name') or not target.get('stage_id'):
                errors.append('missing_task_anchor')
            else:
                lo, hi = np.asarray(target['object_aabb_m'])[:, :2]
                xy = positions[-1, :2]
                edge = np.linalg.norm(np.maximum(np.maximum(lo-xy, xy-hi), 0))
                # Some long-running collectors loaded the newer goal sampler
                # after their older record serializer. The dataset-wide
                # contract permits the official 1.5 m bound even when that
                # optional per-record field is absent.
                if edge > float(record.get('task_approach_max_m') or 1.5) + .01:
                    errors.append('goal_too_far_from_task_object')
                center = (lo+hi)/2
                facing = np.arctan2(center[1]-xy[1], center[0]-xy[0])
                yaw = rotations[-1, 2]
                if abs(np.arctan2(np.sin(facing-yaw), np.cos(facing-yaw))) > np.deg2rad(3):
                    errors.append('goal_not_facing_task_object')
        elif record['goal_type'] != 'scene_random' or record.get('target') is not None:
            errors.append('bad_random_goal_label')
        if not record['result']['success']:
            errors.append('reported_failure')
    return errors


def audit_root(root):
    counts = Counter()
    violations = []
    for task_root in sorted((root/'routes').glob('task_*')):
        journal = task_root/'routes.jsonl'
        if not journal.exists():
            continue
        task_index = int(task_root.name.rsplit('_', 1)[1])
        template = json.loads((TASK_TEMPLATES/f'task-{task_index:04d}.json').read_text())
        canonical = {stage: node for stage in template['canonical_sequence']
                     for node in template['nodes'] if node['template_stage_id'] == stage}
        pairs = defaultdict(list)
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            counts['routes'] += 1
            counts[record['goal_type']] += 1
            errors = []
            try:
                errors.extend(audit_record(task_root, record))
            except (OSError, KeyError, ValueError) as error:
                errors.append(f'unreadable_record:{type(error).__name__}')
            if record.get('task') != template['task_name'].replace(' ', '_'):
                errors.append('task_template_mismatch')
            if record['goal_type'] == 'task_target' and record.get('target'):
                target = record['target']
                node = canonical.get(target.get('stage_id'))
                category = node['object_context'][0] if node and node['object_context'] else None
                if (category is None or target.get('template_category') != category
                        or not (target.get('object_category') == category
                                or target.get('object_category', '').startswith(category + '_'))):
                    errors.append('target_not_canonical_task_object')
            endpoints = np.asarray(record.get('planned_endpoints_xy_m', []))
            if endpoints.shape == (2, 2):
                minimum = .25 if record['goal_type'] == 'task_target' else .5
                if np.linalg.norm(endpoints[1]-endpoints[0]) < minimum - 1e-6:
                    errors.append('route_below_planned_minimum')
                instance_key = (record['scene_name'], record['instance'])
                for prior in pairs[instance_key]:
                    forward = np.linalg.norm(endpoints-prior, axis=1).max()
                    reverse = np.linalg.norm(endpoints-prior[::-1], axis=1).max()
                    if min(forward, reverse) < .25 - 1e-6:
                        errors.append('near_duplicate')
                        break
                pairs[instance_key].append(endpoints)
            else:
                errors.append('missing_endpoints')
            if errors:
                violations.append({'route': record['route_id'], 'errors': errors})
    return {'counts': dict(counts), 'violation_count': len(violations),
            'violations': violations[:100]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_root(args.root)
    output = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(output)
    print(output)
    raise SystemExit(bool(report['violation_count']))
