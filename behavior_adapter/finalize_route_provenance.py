"""Normalize mixed-import goal metadata after a task journal is closed.

Long-running scene processes can load a newer goal sampler after an older
collector serializer. Their successful routes remain valid, but a v1 sampling
label may omit the effective 1.5 m bound or 0.25 m short-route fallback.
Only completed 200-route journals are rewritten, atomically and on request.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def repair_record(record):
    if record.get('goal_type') != 'task_target' or record.get('sampling') != 'task_object_approach_v1':
        return False
    points = np.asarray(record['planned_endpoints_xy_m'])
    lo, hi = np.asarray(record['target']['object_aabb_m'])[:, :2]
    edge = float(np.linalg.norm(np.maximum(np.maximum(lo-points[1], points[1]-hi), 0)))
    distance = float(np.linalg.norm(points[1]-points[0]))
    if edge <= 1.05 + .01 and distance >= .5 - 1e-6:
        return False
    if edge > 1.5 + .01 or distance < .25 - 1e-6:
        raise ValueError(f'Out-of-contract task route: {record["route_id"]}')
    record['sampling'] = 'task_object_approach_mixed_import'
    record['task_approach_max_m'] = 1.5
    record['task_route_min_m'] = .25
    record['provenance_note'] = 'v1 collector serializer loaded updated task-goal sampler'
    return True


def finalize(root, apply):
    report = {'eligible_tasks': 0, 'skipped_open_tasks': 0, 'records_to_update': 0,
              'updated_tasks': []}
    for task_root in sorted((root/'routes').glob('task_*')):
        journal = task_root/'routes.jsonl'
        if not journal.exists():
            continue
        complete = task_root/'complete.json'
        if not complete.exists():
            report['skipped_open_tasks'] += 1
            continue
        records = [json.loads(line) for line in journal.open()]
        if len(records) != 200 or json.loads(complete.read_text()).get('routes') != 200:
            raise ValueError(f'Invalid completed journal: {task_root}')
        report['eligible_tasks'] += 1
        changed = sum(repair_record(record) for record in records)
        report['records_to_update'] += changed
        if not changed or not apply:
            continue
        temporary = journal.with_suffix('.jsonl.provenance_tmp')
        with temporary.open('w') as output:
            for record in records:
                output.write(json.dumps(record) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, journal)
        report['updated_tasks'].append(task_root.name)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Rewrite eligible closed journals')
    args = parser.parse_args()
    print(json.dumps(finalize(args.root, args.apply), indent=2))
