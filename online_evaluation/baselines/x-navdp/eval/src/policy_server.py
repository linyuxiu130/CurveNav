"""
X-NavDP policy server for navigation policy inference.

This module provides a server that accepts RGB-D images and goals via HTTP POST,
runs the X-NavDP policy to generate navigation trajectories, and returns the results.
"""

from flask import Flask, request, jsonify
from navbench.protocol import gradient_inference_context, health_payload, policy_response, read_depth, read_rgb, seed_process
import numpy as np
import json
import threading
from functools import wraps

from .policy_agent import NavDP_Agent

app = Flask(__name__)
navdp_navigator = None
navdp_state_lock = threading.RLock()
embodiment_idx = 2  # Default: unitree_go2 (0=wheeled, 1=humanoid, 2=quadruped)
policy_device = "cuda:0"
policy_checkpoint = None


@app.route("/health", methods=["GET"])
def health():
    return health_payload()


def synchronized_navdp_route(func):
    """Serialize policy server access to mutable navigator state."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with navdp_state_lock:
            return func(*args, **kwargs)
    return wrapper


def init_app(
    embodiment,
    device="cuda:0",
    checkpoint=None,
):
    """Initialize the global app configuration."""
    global embodiment_idx, policy_device
    global policy_checkpoint

    EMBODIMENT_NAME_TO_IDX = {
        "quadruped": 2,
        "humanoid": 1,
        "wheeled": 0,
    }
    if embodiment not in EMBODIMENT_NAME_TO_IDX:
        raise ValueError(
            f"embodiment must be one of {list(EMBODIMENT_NAME_TO_IDX)}, got {embodiment!r}"
        )
    embodiment_idx = EMBODIMENT_NAME_TO_IDX[embodiment]
    policy_device = device
    policy_checkpoint = checkpoint

    return app


@app.route("/navigator_reset", methods=['POST'])
@synchronized_navdp_route
def navdp_reset():
    """Reset the navigator with initial camera intrinsics and batch size."""
    global navdp_navigator

    seed_process(request.get_json().get('seed'))
    intrinsic = np.array(request.get_json().get('intrinsic'))
    batchsize = np.array(request.get_json().get('batch_size'))

    if navdp_navigator is None:
        navdp_navigator = NavDP_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=policy_checkpoint,
            device=policy_device,
            embodiment=embodiment_idx,
        )
        navdp_navigator.reset(batchsize)
    else:
        navdp_navigator.reset(batchsize)

    return jsonify({"algo": "navdp-rl"})


@app.route("/navigator_reset_env", methods=['POST'])
@synchronized_navdp_route
def navdp_reset_env():
    """Reset a specific environment in the navigator."""
    env_id = int(request.get_json().get('env_id'))
    navdp_navigator.reset_env(env_id)

    return jsonify({"algo": "navdp-rl"})


@app.route("/pointgoal_step", methods=['POST'])
@synchronized_navdp_route
def navdp_step_xy():
    """Process a point goal navigation step (with robot state guidance)."""
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = navdp_navigator.batch_size

    state_data = json.loads(request.form['state_data'])
    robot_pos = np.array(state_data['robot_pos'])
    robot_quat = np.array(state_data['robot_quat'])
    image = read_rgb("image", batch_size)
    depth = read_depth(batch_size)

    with gradient_inference_context():
        execute_trajectory, _, _, _ = \
            navdp_navigator.step_pointgoal_with_guidance(goal, image, depth, robot_pos, robot_quat)
    return policy_response(execute_trajectory)


@app.route("/shutdown", methods=["POST"])
def shutdown_server():
    """Shutdown the server gracefully."""
    shutdown = request.environ.get("werkzeug.server.shutdown")
    if shutdown is not None:
        threading.Thread(target=shutdown, daemon=True).start()
    return jsonify({"status": "ok", "message": "server shutting down"})


def run_server(port=8888, host='127.0.0.1'):
    """Run the Flask server."""
    app.run(host=host, port=port, threaded=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--embodiment",
        type=str,
        required=True,
        choices=("wheeled", "humanoid", "quadruped"),
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    init_app(
        args.embodiment,
        args.device,
        checkpoint=args.checkpoint,
    )
    run_server(port=args.port)
