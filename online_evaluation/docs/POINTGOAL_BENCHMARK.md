# PointGoal B16 benchmark contract

## Frozen inputs

`suites/pointgoal-v2.json` pins the X-NavDP paper, official code commit, X-NavDP asset
revision, Scene-N1 revision, scene split digest, Dingo digest, every episode-file digest,
and the aggregate 40-file digest. The executable split is exactly 20 Home plus 20
Commercial scenes in `scene_split.json` order.

Each official episode file is a `(100, 5)` little-endian float64 NumPy array:

```text
start_x_m, start_y_m, goal_x_m, goal_y_m, start_yaw_rad
```

Coordinates are simulator-world planar coordinates and the fifth column is used verbatim for
the initial robot yaw. Episode IDs are the official row indices `0..99`. No route is regenerated,
reoriented, filtered, assigned a difficulty, or supplied with an expert path.

## Simulator and controller

- Dingo wheel radius `0.06125 m`, wheel base `0.22616 m`.
- RGB/depth camera `640×360`, period `0.05 s`, pose and pinhole parameters exactly as frozen
  in the suite manifest.
- Differential-drive unicycle MPC: 30 nodes, `0.1 s` step, `|v|≤0.5 m/s`, `|ω|≤0.5 rad/s`,
  `Q=diag(10,10,0)`, `R=diag(0.05,0.05)`.
- Policy input depth remains the single raw transport: `[B,H,W,1]` float32 metres, with NaN
  denoting invalid depth. There is no uint16 encoding or compatibility fallback.
- The policy response contains only the selected execution trajectory; candidate trajectories and
  critic values remain server-internal because the released evaluator does not consume them.
- Seed is `1234` for every scene. The project result protocol uses 16 simulator environments.
- Preserve the released domain configs: Home uses `rgb_num_samples=7` and
  `height_offset=0.1`; Commercial uses `rgb_num_samples=8` and `height_offset=0.5`.

The project entry point runs the released evaluator at commit
`878740a2011856d0e3782dd6ccd880fd2eccd70f` with Isaac Sim 5.0, Isaac Lab 0.46.2 and
`acados_template`. The former `omni.isaac.lab`/Ipopt backport has been removed, so a successful
run cannot silently use a different simulator API or optimizer. The pinned runtime installs the
sole raw-tensor client, raw X-NavDP policy server and metric-only policy agent, while the evaluator
omits the bird-eye sensor and per-step MP4 writer. No PIL/quantized-depth or visualization server
path remains; those streams do not feed policy observations, control, termination, or metrics.

## Termination and metrics

Arrival starts when planar goal distance is below `0.5 m` and the sum of absolute world linear
velocity components is below `0.25 m/s`; the latched stop timer then runs for `4 s`. The only
other evaluator termination is the `122 s` timeout. The project result records no custom collision,
stuck, selector, or latency metric.

For each episode, the released evaluator defines `d` as initial Euclidean start-goal distance,
`p` as executed planar path length, and computes:

```text
success = 1 for arrival termination, else 0
SPL = success * d / max(d, p)
```

This is the X-NavDP release's metric, even though other PointGoal benchmarks sometimes use a
geodesic shortest-path numerator. Report arithmetic mean SR and SPL over the same episode set.

## Evaluation volume

The benchmark has one official project volume: 100 rows per scene, 4,000 episodes per model.
The upstream-style `--episodes-per-scene` option exists only for import/runtime diagnostics; any
value other than 100 makes the run ineligible for project aggregation.

## Outputs

Every scene writes `metric.csv` with unique contiguous official episode indices. Its exact fields
are `success`, `spl`, `distance`, and `episode_idx`. `episodes.csv` merges all scenes;
`summary.csv` contains the separate Home and Commercial means. The launcher records the suite digest,
source revisions, model checkpoint digest, command, seed, and runtime settings.

For closed-loop diagnosis, each completed episode also writes
`trajectory_traces/episode-NNN.npz` beside `metric.csv`. The numeric, compressed NPZ separates
simulation-rate execution from policy-rate replanning. Step arrays contain simulation time, world
robot pose, robot-frame PointGoal, measured planar speed, the consumed MPC command, low-level
action, plan version and control index. Plan arrays contain the selected local trajectory padded
only along its variable point axis, its exact per-plan point count, MPC controls and predicted
states, adaptive desired speed, reference curvature, and policy/MPC latency. Scalar fields contain
termination, initial goal distance, executed path length and exact
simulation elapsed time. The
terminal pose and goal fields are explicitly named `terminal_pre_step_*` because the vectorized
environment may reset before returning the terminal observation. No RGB/depth frames, candidates,
critic outputs or Python objects are stored. Tracing observes the existing tensors after they are
computed and does not enter policy, MPC, termination or metric calculations.

After each checkpoint finishes, `navbench.trajectory_metrics` reconciles every trace against its
official metric row before writing `trajectory_episodes.csv` and `trajectory_summary.csv` at the
session root. Its geometry-only diagnostics are final goal distance, signed goal progress,
progress per executed metre, path-length efficiency, detour overhead
`max(path + final_distance - initial_distance, 0)`, goal-distance backtracking and monotonicity,
world-path tortuosity, speed, goal bearing, MPC command variation, policy/MPC latency, local-plan
arc length, endpoint error, discrete curvature peak and curvature energy. These diagnostics do not
change termination or SPL and do not infer collision from low motion; contact is not part of the
released wheeled evaluator signal.

Each GPU executes one resident scene job with 16 vector environments. Policy server placement is
recorded explicitly, and a server is restarted between scene jobs.

## Distributed execution and resumption

The launcher uses one independent worker per selected GPU and a dynamic scene queue. On multiple
hosts, `--shard-index N --shard-count K` assigns the same frozen scene order by `jobs[N::K]`;
each shard therefore owns complete scenes and never splits or duplicates an episode. Shards must
use the same manifest digest, upstream runtime revision, seed, model checkpoint and simulator
configuration. `navbench.metrics` accepts repeated `--root` arguments and rejects duplicate
`(domain, scene, episode_idx)` identities before writing a merged result.

`--resume-root` is an explicit, identity-checked continuation path. It skips only a scene whose
local `metric.csv` contains exactly the expected contiguous official episode IDs; partial or
malformed outputs remain pending. This avoids rerunning completed scenes after a host interruption
without accepting stale results from another model, suite, runtime or shard.

These features change only scheduling and result assembly around the frozen B16 execution. They do
not enable mixed precision, altered observations, protocol fallbacks or cross-scene server reuse.
