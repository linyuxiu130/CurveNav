from flask import Flask, request, jsonify
from policy_agent import NavDP_Agent
from navbench.protocol import health_payload, inference_context, policy_response, read_depth, read_rgb, seed_process
import numpy as np
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(ROOT_DIR / "checkpoints/navdp/navdp_pretrain.ckpt"),
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

app = Flask(__name__)
navdp_navigator = None


@app.route("/health", methods=['GET'])
def health():
    return health_payload()

@app.route("/navigator_reset",methods=['POST'])
def navdp_reset():
    global navdp_navigator
    seed_process(request.get_json().get('seed'))
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))
    global_batch_size = int(request.get_json().get('global_batch_size', batchsize))
    batch_start = int(request.get_json().get('batch_start', 0))
    if navdp_navigator is None:
        navdp_navigator = NavDP_Agent(intrinsic,
                                image_size=224,
                                memory_size=8,
                                predict_size=24,
                                temporal_depth=16,
                                heads=8,
                                token_dim=384,
                                navi_model=args.checkpoint,
                                device=args.device,
                                cache_rgb_tokens=True)
        navdp_navigator.reset(batchsize, threshold, global_batch_size, batch_start)
    else:
        navdp_navigator.reset(batchsize, threshold, global_batch_size, batch_start)

    return jsonify({"algo":"navdp"})

@app.route("/navigator_reset_env",methods=['POST'])
def navdp_reset_env():
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"navdp"})

@app.route("/pointgoal_step",methods=['POST'])
def navdp_step_xy():
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    batch_size = navdp_navigator.batch_size

    image = read_rgb('image', batch_size)
    depth = read_depth(batch_size)
    with inference_context():
        execute_trajectory, _, _, _ = navdp_navigator.step_pointgoal(goal,image,depth)
    return policy_response(execute_trajectory)


@app.post("/shutdown")
def navdp_shutdown():
    """Acknowledge evaluator teardown and release temporal policy state."""
    global navdp_navigator
    navdp_navigator = None
    return jsonify({"ok": True})


if __name__ == "__main__":
    # The agent owns temporal memory and is intentionally single-requested.
    app.run(host='127.0.0.1', port=args.port, threaded=False)
