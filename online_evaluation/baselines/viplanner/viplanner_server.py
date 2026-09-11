from flask import Flask, request, jsonify
from viplanner_agent import VIPlannerAgent
from navbench.protocol import health_payload, inference_context, policy_response, read_depth, read_rgb, seed_process
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--config",type=str,default="./configs/viplanner.yaml")
parser.add_argument("--checkpoint",type=str,default="./checkpoints/viplanner.pt")
parser.add_argument("--m2f_config",type=str,default="~/miniconda3/envs/habitat/lib/python3.9/site-packages/mmdet/.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py")
parser.add_argument("--m2f_checkpoint",type=str,default="./checkpoints/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth")
parser.add_argument("--device",type=str,default="cuda:0")
args = parser.parse_args()

app = Flask(__name__)
viplanner_navigator = None

@app.route("/health", methods=["GET"])
def health():
    return health_payload()

@app.route("/navigator_reset",methods=['POST'])
def viplanner_reset():
    global viplanner_navigator
    seed_process(request.get_json().get('seed'))
    intrinsic = np.array(request.get_json().get('intrinsic'))
    if viplanner_navigator is None:
        viplanner_navigator = VIPlannerAgent(intrinsic,
                                            m2f_path=args.m2f_checkpoint,
                                            m2f_config_path=args.m2f_config,
                                            model_path=args.checkpoint,
                                            model_config_path=args.config,
                                            device=args.device)
    return jsonify({"algo":"viplanner"})

@app.route("/navigator_reset_env",methods=['POST'])
def viplanner_reset_env():
    return jsonify({"algo":"viplanner"})

@app.post("/shutdown")
def viplanner_shutdown():
    return jsonify({"status": "ok"})

def process_goal(goal,range=10.0):
    return_goal = np.clip(goal,-range,range)
    return return_goal

@app.route("/pointgoal_step",methods=['POST'])
def viplanner_step_pointgoal():
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    goal = process_goal(goal)
    batch_size = goal.shape[0]

    image = read_rgb("image", batch_size)
    depth = read_depth(batch_size)

    with inference_context():
        _,trajectory,_ = viplanner_navigator.step_pointgoal(image,depth,goal)
    return policy_response(trajectory)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=args.port,threaded=False)
