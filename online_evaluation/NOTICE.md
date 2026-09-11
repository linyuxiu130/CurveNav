# Upstream code and assets

General Navigation Benchmark contains an evaluation integration and project-native research
code. The active PointGoal contract, the externally staged evaluator, and the vendored X-NavDP
policy-server components are pinned to
[NavDP](https://github.com/InternRobotics/NavDP) commit
`878740a2011856d0e3782dd6ccd880fd2eccd70f`. It also contains adapted components
from NavDP and SanD-Planner. It does not grant new rights over upstream code,
model checkpoints, Isaac Sim/Lab, robot assets, or scene assets.

The canonical CurveNav source is `../src/curvenav` in this same
[linyuxiu130/CurveNav](https://github.com/linyuxiu130/CurveNav) repository.
`baselines/curvenav/` contains only the benchmark protocol adapter, not a second model
implementation. Runs record the explicit config bundle and checkpoint identity.
Generated datasets and model checkpoints are managed separately by SHA-256 and are not
distributed through Git.

SanD-Planner attribution and license are retained in `baselines/sandplanner/`.
X-NavDP attribution, MIT license, citation, and third-party notices are retained
in `baselines/x-navdp/`. Depth Anything V2's license is retained alongside its
vendored inference source.
