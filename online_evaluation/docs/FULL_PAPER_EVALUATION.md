# Full project evaluation plan: accelerated X-NavDP PointGoal workload

## Experimental unit

Evaluate every model on the same immutable 40 scenes and the same 100 released episodes per
scene: 20 Home plus 20 Commercial, for 4,000 episodes per model. Do not introduce Easy/Hard
subsets, generated endpoints, expert routes, or alternate success/SPL definitions.

The X-NavDP paper's simulation table contains iPlanner, ViPlanner, NavDP, NavOL, SIDP,
NavDP-RL and X-NavDP; NavOL/SIDP/NavDP-RL report only some wheeled cells. This repository
currently has runnable adapters for iPlanner, ViPlanner, NavDP and X-NavDP. CurveNav is the new
method evaluated under the identical protocol. SanD-Planner may be retained as an additional
baseline, but it must not be presented as an original X-NavDP paper row. NavOL, SIDP and NavDP-RL
remain explicitly missing until their exact released source, checkpoint and inference contracts
are supplied; paper numbers must not be copied into locally measured columns.

## Controlled variables

For every model freeze:

- suite-definition digest and all upstream revisions;
- episode IDs `0..99`, scene order, and seed `1234`;
- Dingo USD, scale, camera intrinsics/extrinsics, depth transport, simulation timing;
- the released Home/Commercial config difference (`7/0.1` versus `8/0.5` for RGB samples and
  height offset);
- official arrival rule, timeout, MPC horizon/weights/limits, and evaluator precision;
- model source revision, checkpoint SHA-256, and model-specific strict loading contract.

Use 16 simulator environments for every reported model. Different models may run on different
but equivalent GPUs or servers only when software, scenes, settings, episode identities and batch
size are identical. The resulting table is the project's accelerated benchmark and must not be
presented as a numerically exact reproduction of upstream single-environment execution.

## Procedure

1. Run `scripts/prepare_scenes.sh check` and preserve its successful asset validation.
2. Run dependency/contract tests and the raw-depth boundary round trip.
3. Run `scripts/check_xnavdp_runtime.py "$NAVBENCH_XNAVDP_ROOT" --python
   "$NAVBENCH_EVAL_PYTHON"`; any revision, source-file, Isaac Lab, acados, or raw-bridge mismatch
   is fatal.
4. Run a one-episode smoke for each adapter; discard smoke outputs from official aggregates.
5. Run the default 100 episodes per scene once per model. Multiple hosts may use deterministic
   `--shard-index/--shard-count` scene shards; a resumed shard must pass the launcher identity
   check and may queue only scenes whose complete official metric file is missing.
6. Require exactly 4,000 unique `(domain, scene, episode_idx)` rows per model and the identical
   identity set across models.
7. Run the released statistics procedure separately over 2,000 Home episodes and 2,000
   Commercial episodes.

Report Home SR/SPL and Commercial SR/SPL exactly as the released evaluator's statistics script.
Do not add a new primary aggregate, difficulty split, oracle analysis, collision metric, latency
metric, or custom confidence-interval procedure to the project result table.

## Acceptance checklist

- 40 official scenes, exactly 20 per domain.
- 100 rows per scene and 4,000 unique rows per model.
- Episode and navigation inputs match pinned SHA-256 values.
- No Easy/Hard, regenerated episode, sidecar, geodesic numerator, stuck termination, matched
  trace, oracle selector, protocol fallback, MPS, AMP, or altered camera observation.
- Success threshold `0.5 m`, velocity threshold `0.25 m/s`, hold `4 s`, timeout `122 s`.
- SPL recomputes exactly as `success*d/max(d,p)` using initial Euclidean distance.
- Commands, environment versions, GPU type, runtime, and checkpoint SHA are archived.
- Each scene is one resident job with 16 environments, one evaluator and its declared server mapping.
- The launcher may use a dynamic scene queue and host-only CPU/resource-poll tuning; these do not
  alter scene order in the manifest, episode IDs, simulator settings, policy inputs, or metrics.
- Scene sharding across hosts and strict multi-root metric merging are allowed; duplicate
  `(domain, scene, episode_idx)` identities are fatal.
- The fixed launch gate is at least 8,500 MiB free on the assigned GPU. Server restart between
  scenes remains required for project comparability.

The upstream-compatible `--episodes-per-scene` diagnostic option may shorten a debugging run, but
it is not a benchmark budget and its outputs are not official project results.
