# BEHAVIOR Live Viewer

Persistent, browser-controlled R1Pro navigation using the trained CurveNav policy.
The default `turning_on_radio` task is BEHAVIOR task `000`, training instance `000`.
The scene is loaded once; robot randomization keeps the simulator running.

On the H800 host:

```bash
cd /shibo_huang/CurveNav
bash behavior_adapter/live_viewer.sh start
bash behavior_adapter/live_viewer.sh status
```

The first start loads Isaac Sim and the task assets and can take a few minutes.
The server listens only on `127.0.0.1:8765` and persists in a tmux session. Connect
from the client machine with:

```bash
ssh -N -L 8765:127.0.0.1:8765 -p 6135 root@120.48.58.51
```

Then open <http://127.0.0.1:8765>. Use `bash behavior_adapter/live_viewer.sh logs`
for simulator startup output and `bash behavior_adapter/live_viewer.sh stop` to shut
down the viewer and simulator.

Choose another training task or instance at startup without changing the code:

```bash
CURVENAV_LIVE_TASK=turning_on_radio CURVENAV_LIVE_INSTANCE=12 \
  bash behavior_adapter/live_viewer.sh start
```

The default map is an actual top camera render with ceiling and roof objects hidden.
The scene view takes most of the browser width; “放大俯视图” expands it further.
Choose “指定放置机器人”, optionally set heading in degrees, then click the map.
Placement requires a real floor and no physical collision, and resets the sensor
history before the next goal.
Use “显示参考栅格” to inspect the floor and collision map for context. Clicked targets
go directly to the model without A* or grid gating. The blue line is the current
learned curve used for control; cyan is measured motion. Only actual simulator
collision contact triggers an automatic safety stop during navigation.
Head RGB and linear depth come from the same 224 × 224 camera observation. The
status panel checks online depth range, intrinsics, temporal history, and optical
camera pose against the model input contract. The viewer supports navigation
integration; it does not perform BEHAVIOR task interactions or claim task success.
