"""Ground BEHAVIOR navigation goals in a task stage and a loaded object instance."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label


TEMPLATES = Path('/mnt/dataset/Benchmark/Behavior2026/2026-challenge-demos/primitive_task_templates_context_v3_20260807')
GROUND = {'floors', 'lawn', 'driveway', 'carpet'}
APPROACH_MIN_M = .35
# OmniGibson's BASE_POSE_SAMPLING_UPPER_BOUND is 1.5 m.
APPROACH_MAX_M = 1.5


class TaskGoalUnavailable(ValueError):
    """The loaded task instance has no safe approach to any canonical target."""


def task_targets(task, task_index):
    """Return ordered, task-scoped physical targets from the canonical action sequence."""
    from route import RoutePlanningError

    template = json.loads((TEMPLATES/f'task-{task_index:04d}.json').read_text())
    if template['task_name'].replace(' ', '_') != task:
        raise RoutePlanningError(f'Task template does not match loaded task: {task}')
    nodes = {node['template_stage_id']: node for node in template['nodes']}
    return [(stage, nodes[stage]) for stage in template['canonical_sequence']]


def iter_targets(evaluator, task, task_index, rng):
    """Yield task-scoped physical objects in canonical stage order."""
    scope = [(name, obj) for name, obj in evaluator.env.task.object_scope.items()
             if obj is not None and hasattr(obj, 'aabb') and hasattr(obj, 'category')]
    for stage_id, node in task_targets(task, task_index):
        # The first context item is the acted-on object. Later items describe
        # supports or destinations and need an instance relation to be valid.
        # Never navigate to an arbitrary same-category support instead.
        for category in node['object_context'][:1]:
            if category in GROUND:
                continue
            exact = [(name, obj) for name, obj in scope if obj.category == category]
            # Some template labels abbreviate a category, e.g. digital_camera.
            matches = exact or [(name, obj) for name, obj in scope
                                if obj.category.startswith(category + '_')]
            for index in rng.permutation(len(matches)):
                name, obj = matches[int(index)]
                lo, hi = (bound.cpu().numpy() for bound in obj.aabb)
                yield {'stage_id': stage_id, 'stage_label': node['label'],
                       'template_category': category, 'scope_name': name,
                       'object_name': obj.name, 'object_category': obj.category,
                       'object_aabb_m': [lo.tolist(), hi.tolist()]}


def resolve_and_sample_task_goal(evaluator, task, task_index, grid, rng, band='any'):
    """Advance to the next canonical task stage only if an earlier one is unreachable."""
    from route import RoutePlanningError

    reasons = {}
    for target in iter_targets(evaluator, task, task_index, rng):
        try:
            endpoints, final_yaw = sample_task_endpoints(grid, target, rng, band)
        except RoutePlanningError as error:
            reasons.setdefault(target['object_category'],
                               f'{target["scope_name"]}: {error}')
            continue
        return target, endpoints, final_yaw
    detail = ('; '.join(f'{category}/{reason}' for category, reason in reasons.items())
              if reasons else 'no canonical object resolved in this instance')
    raise TaskGoalUnavailable(f'No physically reachable canonical task target for {task}: {detail}')


def sample_task_endpoints(grid, target, rng, band='any'):
    """Choose a collision-free approach pose near the target and a connected start."""
    from route import RESOLUTION, RoutePlanningError

    free = grid['free']
    components, _ = label(free)
    cells = np.argwhere(free)
    if not len(cells):
        raise RoutePlanningError('No collision-free cells in this scene')
    xy = grid['origin_xy'] + cells[:, ::-1] * RESOLUTION
    lo, hi = np.asarray(target['object_aabb_m'])[:, :2]
    edge_delta = np.maximum(np.maximum(lo - xy, xy - hi), 0)
    approach_distance = np.linalg.norm(edge_delta, axis=1)
    center = (lo + hi) / 2
    center_distance = np.linalg.norm(xy - center, axis=1)
    limits = {'near': (.5, 1.5), 'medium': (1.5, 5.), 'long': (5., 10.),
              'narrow': (1.5, 5.), 'any': (.5, 10.)}
    low, high = limits[band]
    # Prefer the original close interaction area; use the official 1.5 m bound
    # only when no connected, collision-free start exists nearer to the object.
    # Some official robot starts are already beside the task object. Prefer a
    # normal route; accept a 25 cm approach only when the connected safe area
    # cannot support the usual 50 cm minimum.
    start_minimums = (low, .25) if band == 'any' else (low,)
    goal_counts = []
    for minimum in start_minimums:
        for upper in (1.05, APPROACH_MAX_M):
            center_limit = max(1.6, np.linalg.norm((hi-lo)/2) + upper)
            goals = cells[(approach_distance >= APPROACH_MIN_M)
                          & (approach_distance <= upper)
                          & (center_distance <= center_limit)]
            goal_counts.append(len(goals))
            for goal in goals[rng.permutation(len(goals))[:128]]:
                starts = cells[components[cells[:, 0], cells[:, 1]] == components[tuple(goal)]]
                distances = np.linalg.norm(starts - goal, axis=1) * RESOLUTION
                eligible = (distances >= minimum) & (distances <= high)
                if band == 'narrow':
                    eligible &= grid['clearance_m'][starts[:, 0], starts[:, 1]] < .30
                starts = starts[eligible]
                if len(starts):
                    start = starts[int(rng.integers(len(starts)))]
                    return (grid['origin_xy'] + np.stack([start[::-1], goal[::-1]]) * RESOLUTION,
                            float(np.arctan2(center[1] - (grid['origin_xy'][1]+goal[0]*RESOLUTION),
                                             center[0] - (grid['origin_xy'][0]+goal[1]*RESOLUTION))))
    raise RoutePlanningError(f'No connected {band} start near {target["object_name"]}; '
                             f'nearest safe edge={approach_distance.min():.3f} m, '
                             f'candidate goals={goal_counts}')
