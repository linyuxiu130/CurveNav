from flask import Flask, request, jsonify
from iplanner_agent import IPlannerAgent
from navbench.protocol import health_payload, inference_context, policy_response, read_depth, seed_process
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--config",type=str,default="./configs/iplanner.yaml")
parser.add_argument("--checkpoint",type=str,default="./checkpoints/iplanner.pth")
parser.add_argument("--device",type=str,default="cuda:0")
args = parser.parse_args()

app = Flask(__name__)
iplanner_navigator = None

@app.route("/health", methods=["GET"])
def health():
    return health_payload()

@app.route("/navigator_reset",methods=['POST'])
def iplanner_reset():
    global iplanner_navigator
    seed_process(request.get_json().get('seed'))
    intrinsic = np.array(request.get_json().get('intrinsic'))
    if iplanner_navigator is None:
        iplanner_navigator = IPlannerAgent(intrinsic,
                                           model_path=args.checkpoint,
                                           model_config_path=args.config,
                                           device=args.device)
    return jsonify({"algo":"iplanner"})

@app.route("/navigator_reset_env",methods=['POST'])
def iplanner_reset_env():
    return jsonify({"algo":"iplanner"})

@app.post("/shutdown")
def iplanner_shutdown():
    return jsonify({"status": "ok"})

def process_goal(goal,range=5.0):
    return_goal = np.clip(goal,-range,range)
    return return_goal

@app.route("/pointgoal_step",methods=['POST'])
def iplanner_step_pointgoal():
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    goal = process_goal(goal)
    batch_size = goal.shape[0]

    depth = read_depth(batch_size)

    with inference_context():
        _,trajectory,_ = iplanner_navigator.step_pointgoal(depth,goal)
    return policy_response(trajectory)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=args.port,threaded=False)
