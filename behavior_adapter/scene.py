"""Load a BEHAVIOR training instance through the official simulator API."""
import argparse
import json
from pathlib import Path

import numpy as np


def training_instance(value):
    value = int(value)
    if not 0 <= value <= 300:
        raise ValueError('Training instance must be in [0, 300]; test instances are excluded')
    return value


def load_scene(task, instance):
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import Evaluator, DEFAULT_ROBOT_CONFIG_PATH
    from collect import DATA

    robot_config = OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH)
    robot_config.disable_grasp_handling = True
    robot_config.sensor_config.VisionSensor.modalities = ["rgb", "depth_linear"]
    robot_config.sensor_config.VisionSensor.sensor_kwargs.image_height = DATA.image_height
    robot_config.sensor_config.VisionSensor.sensor_kwargs.image_width = DATA.image_width

    cfg = OmegaConf.create({
        'env_wrapper': {'_target_': 'omnigibson.envs.EnvironmentWrapper'},
        'policy_name': 'local',
        'model': {'_target_': 'omnigibson.eval.policies.LocalPolicy', 'action_dim': None},
        'headless': True, 'partial_scene_load': False, 'max_steps': 10000,
        'write_video': False, 'mode': 'train', 'seed': 0,
        'task': {'name': task}, 'robot': robot_config,
    })
    evaluator = Evaluator(cfg)
    evaluator.reset()
    evaluator.load_task_instance(instance)
    return evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='turning_on_radio')
    parser.add_argument('--instance', type=training_instance, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-motion', action='store_true')
    parser.add_argument('--benchmark-step', action='store_true')
    parser.add_argument('--generate-route', action='store_true')
    parser.add_argument('--probe-map', action='store_true')
    parser.add_argument('--collect-plan', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import omnigibson as og
    from omnigibson.macros import gm
    from PIL import Image
    import torch

    gm.HEADLESS = True
    np.random.seed(0)
    torch.manual_seed(0)
    with load_scene(args.task, args.instance) as evaluator:
        from pxr import UsdPhysics
        head = evaluator.robot_camera_names['head']
        camera = evaluator.robot.sensors[head.split('::', 1)[1]]
        if args.collect_plan or args.generate_route:
            from robot import configure_navigation_camera
            from collect import DATA
            camera = configure_navigation_camera(evaluator, DATA.image_width, DATA.image_width)
        for _ in range(3):
            og.sim.render()
        obs = camera.get_obs()[0]
        rgb = obs['rgb'].cpu().numpy()
        depth = obs['depth_linear'].cpu().numpy()
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise RuntimeError('Head camera has no valid depth')
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(args.output / 'head_rgb.png')
        np.save(args.output / 'head_depth_m.npy', depth)
        objects = []
        for obj in evaluator.env.scene.objects:
            position, orientation = obj.get_position_orientation()
            lo, hi = obj.aabb
            objects.append({'name': obj.name, 'category': obj.category,
                            'prim_path': obj.prim_path,
                            'position_m': position.tolist(), 'orientation_xyzw': orientation.tolist(),
                            'aabb_m': [lo.tolist(), hi.tolist()]})
        collisions = [str(p.GetPath()) for p in og.sim.stage.Traverse()
                      if p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI)
                      and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        if not objects or not collisions:
            raise RuntimeError('Loaded scene lacks objects or enabled collision geometry')
        report = {'task': args.task, 'instance': args.instance, 'split': 'train',
                  'scene': evaluator.env.task.scene_name, 'full_scene': True,
                  'robot_model': evaluator.robot.model,
                  'head_camera': head, 'depth_shape': list(depth.shape),
                  'valid_depth_fraction': float(valid.mean()),
                  'depth_range_m': [float(depth[valid].min()), float(depth[valid].max())],
                  'objects': objects, 'enabled_collision_prims': collisions,
                  'scope': 'Scene loading audit only; collision prims include the robot. AABBs are not a navigation map.'}
        from robot import robot_contract
        body = robot_contract(evaluator, args.output)
        (args.output / 'robot.json').write_text(json.dumps(body, indent=2) + '\n')
        (args.output / 'scene.json').write_text(json.dumps(report, indent=2) + '\n')
        if args.probe_map:
            from route import build_map
            scope = {name: {'name': obj.name, 'category': obj.category,
                            'aabb_m': [bound.tolist() for bound in obj.aabb]}
                     for name, obj in evaluator.env.task.object_scope.items()
                     if obj is not None and hasattr(obj, 'aabb')}
            (args.output/'object_scope.json').write_text(json.dumps(scope, indent=2)+'\n')
            grid = build_map(evaluator, body, full_scene=True)
            np.savez_compressed(args.output/'map.npz', **grid)
            print(json.dumps({'event': 'map_probe', 'task': args.task, 'instance': args.instance,
                              'layout_cells': int(grid['layout_support'].sum()),
                              'physical_cells': int(grid['physical_support'].sum()),
                              'free_cells': int(grid['free'].sum()),
                              'support_meshes': int(grid['support_mesh_count'])}), flush=True)
        if args.benchmark_step:
            from test_scene import benchmark_step
            measured = benchmark_step(evaluator)
            (args.output/'step_benchmark.json').write_text(json.dumps(measured, indent=2)+'\n')
            print(json.dumps({'event': 'step_benchmark', **measured}), flush=True)
        if args.check_motion:
            from test_scene import check_motion
            motion = check_motion(evaluator)
            (args.output / 'motion.json').write_text(json.dumps(motion, indent=2) + '\n')
        if args.collect_plan or args.generate_route:
            camera.remove_modality('rgb')
            og.sim.update_handles()
        if args.collect_plan:
            from collect import collect_task
            collect_task(evaluator, args.collect_plan, args.output)
        if args.generate_route:
            from route import generate_route
            result = generate_route(evaluator, body, args.output)
            if not result["success"]:
                raise RuntimeError(result["failure"])
        print(json.dumps({'event': 'scene_adapter_complete',
                          **{k: v for k, v in report.items()
                             if k not in ('objects', 'enabled_collision_prims')}}), flush=True)


if __name__ == '__main__':
    main()
