#!/usr/bin/env python3
"""
SanD-planner 简易推理服务器 / Simple SanD-planner inference server.

无频率限制，每次请求都重新规划，只支持 point goal（点目标）导航。
No rate limit; every request triggers a fresh planning pass, and only point-goal navigation is supported.
"""

from flask import Flask, request, jsonify
from sand_planner.config import InferenceConfig
from sand_planner.agent.sand_planner_agent import SandPlannerAgent
from navbench.protocol import inference_context, policy_response, read_depth, read_rgb, seed_process
import numpy as np
import json

# 设置 matplotlib 使用非交互式后端，避免 GUI 相关的错误
# Configure matplotlib to use a non-interactive backend to avoid GUI-related errors.
import matplotlib
# 使用 Anti-Grain Geometry 后端，无需 X11 或其他 GUI
# Use the Anti-Grain Geometry backend, which requires neither X11 nor any other GUI.
matplotlib.use('Agg')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8890, help="服务器端口")
parser.add_argument("--checkpoint", type=str, default=None, help="模型checkpoint路径，默认使用InferenceConfig中的配置")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()
app = Flask(__name__)
navigator = None


@app.route("/navigator_reset", methods=['POST'])
def navigator_reset():
    """重置导航器 / Reset the navigator."""
    global navigator

    seed_process(request.get_json().get('seed'))
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))

    print('相机内参:', intrinsic)
    print('停止阈值:', threshold)
    print('批处理大小:', batchsize)

    if navigator is None:
        print("🚀 创建SanD-planner Agent...")
        config_kwargs = dict(
            device=args.device,
            save_visualizations=False,
            save_data=False,
            show_verbose=False,
        )
        if args.checkpoint is not None:
            config_kwargs['checkpoint_path'] = args.checkpoint
        config = InferenceConfig(**config_kwargs)
        navigator = SandPlannerAgent(
            image_intrinsic=intrinsic,
            config=config,
            verbose=False,
        )
        navigator.reset(batchsize, threshold)
        # 预热：触发 torch.compile 编译，避免首次推理 timeout
        # Warm-up: trigger torch.compile so the first real inference does not time out.
        warmup_batch = int(batchsize)
        dummy_img = np.random.randint(
            0, 255, (warmup_batch, 480, 640, 3), dtype=np.uint8
        )
        dummy_dep = np.random.rand(
            warmup_batch, 480, 640, 1
        ).astype(np.float32) * 5.0
        dummy_goal = np.tile(
            np.array([[3.0, 0.0, 0.0]]), (warmup_batch, 1)
        )
        with inference_context():
            navigator.step_pointgoal(dummy_goal, dummy_img, dummy_dep)
        navigator.reset(batchsize, threshold)
    else:
        print("🔄 重置现有SanD-planner Agent...")
        # 更新相机内参 / Update the camera intrinsics.
        if not np.array_equal(navigator.image_intrinsic, intrinsic):
            print("📷 更新相机内参...")
            navigator.image_intrinsic = intrinsic
            navigator.update_camera_config(intrinsic)
        navigator.reset(batchsize, threshold)

    return jsonify({"algo": "sand_planner_simple"})

@app.route("/navigator_reset_env", methods=['POST'])
def navigator_reset_env():
    """重置特定环境 / Reset a specific environment."""
    env_id = int(request.get_json().get('env_id'))
    navigator.reset_env(env_id)

    return jsonify({"algo": "sand_planner_simple"})


@app.post("/shutdown")
def navigator_shutdown():
    return jsonify({"status": "ok"})

@app.route("/pointgoal_step", methods=['POST'])
def pointgoal_step():
    """点目标导航步进（无频率限制，每次请求都重新规划）/ Point-goal navigation step (no rate limit; re-plans on every request)."""
    # 解析输入数据（图像、深度图、目标点）/ Parse the input data (image, depth, goal).
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = navigator.batch_size

    # 处理图像数据 / Process the image data.
    image = read_rgb("image", batch_size)

    # 处理深度图数据 / Process the depth image data.
    depth = read_depth(batch_size)

    # 执行导航推理——每次都重新规划，无频率限制。
    # Run navigation inference: re-plan on every call, with no rate limit.
    with inference_context():
        execute_trajectory, _, _, _ = navigator.step_pointgoal(goal, image, depth)
    return policy_response(execute_trajectory)


@app.route("/health", methods=['GET'])
def health_check():
    """健康检查接口 / Health-check endpoint."""
    return jsonify({
        "status": "healthy",
        "agent_loaded": navigator is not None,
        "algorithm": "sand_planner_simple",
        "checkpoint": args.checkpoint
    })

if __name__ == "__main__":
    preview_config = InferenceConfig() if args.checkpoint is None else InferenceConfig(checkpoint_path=args.checkpoint)
    print("🚀 启动Simple SanD-planner服务器...")
    print(f"📍 端口: {args.port}")
    print(f"🤖 模型: {preview_config.checkpoint_path}")
    print(f"✨ 特性: 无频率限制，每次都重新规划")
    print(f"🔗 仅支持点目标（point-goal）导航")
    print(f"🌐 服务器运行在 http://127.0.0.1:{args.port}")

    app.run(host='127.0.0.1', port=args.port, threaded=False)
