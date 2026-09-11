#!/usr/bin/env python3
"""Serve CurveNav through the common NavBench PointGoal protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from flask import Flask, jsonify, request
import numpy as np

from navbench.protocol import (
    health_payload,
    policy_response,
    read_raw,
    seed_process,
)


def create_app(runtime: CurveNavRuntime) -> Flask:
    from curvenav.data.observation import DEPTH_CONTEXT_FIELDS
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return health_payload()

    @app.post("/navigator_reset")
    def navigator_reset():
        payload = request.get_json()
        seed_process(payload.get("seed"))
        runtime.reset(
            int(payload["batch_size"]),
        )
        return jsonify(
            {
                "algo": "curvenav",
                "ema": True,
                "observation_config": asdict(runtime.config.data),
            }
        )

    @app.post("/navigator_reset_env")
    def navigator_reset_env():
        # Sensor history is reset by the simulator before creating a snapshot.
        return jsonify({"algo": "curvenav"})

    @app.post("/pointgoal_step")
    def pointgoal_step():
        state = json.loads(request.form["state_data"])
        trajectory = runtime.step(
            np.asarray(state["planning_goal"], dtype=np.float32),
            {name: read_raw("depth_context_" + name) for name in DEPTH_CONTEXT_FIELDS},
        )
        return policy_response(trajectory.path)

    @app.post("/shutdown")
    def shutdown():
        """Acknowledge evaluator teardown without changing model state."""
        return jsonify({"ok": True})

    return app


def main() -> None:
    from curvenav.deployment import CurveNavRuntime, load_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, policy = load_policy(args.checkpoint, args.config, args.device)
    runtime = CurveNavRuntime(config, policy, args.device)
    create_app(runtime).run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
