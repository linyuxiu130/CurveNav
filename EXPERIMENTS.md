# CurveNav experiments

This is the sole experiment record. Each entry contains only the hypothesis,
authoritative run, result, root conclusion, and next decision. Architecture
details belong in `ARCHITECTURE.md`.

## E001 — separated PointGoal with compressed scene memory

- Hypothesis: four learned metric depth frames, target-independent scene memory,
  and PointGoal used only as a cross-attention query are sufficient for one
  deterministic MeanFlow B-spline to imitate safe experts.
- Model/data: 41.5M parameters; 14 scalar Cartesian-control tokens; 64 learned
  scene slots; sole improved-MeanFlow loss; source-C-space-certified HSSD data
  with 25,777 train and 6,064 validation samples.
- Training: 4x RTX 4090, global batch 1024, 8,000 steps / 200 epochs; stable
  throughput about 5.1k samples/s; final loss 0.00873; no OOM, NaN, or restart.
- Artifact: `outputs/offline-evaluation-geometry-first-step8000-20260901` on
  `go-4x4090`, checkpoint step 8000, EMA weights.
- Strict offline result: ADE 0.03393 m; source collision 14.97%; margin violation
  19.90%; forward-direct collision 3.60%; forward-detour 34.56%; rear-goal
  71.90%; expert-moves-away 73.99%. Predicted curvature variation is 191.64
  1/m2 versus 7.27 for the expert. Batch-1 inference is 45.18 ms P50.
- Root conclusion: training converged, but the architecture did not establish a
  metric correspondence between a 2-D control point and raw depth surfaces.
  Independent scalar control tokens plus positionless learned scene compression
  fit common goal-directed paths while underfitting obstacle-conditioned modes.
  Cartesian B-splines are not by themselves the root cause: SanD uses the same
  representation but explicitly samples and geometrically selects candidates.
- Decision: reject this checkpoint for online scoring. Preserve the one-loss,
  one-step MeanFlow contract; replace scalar trajectory tokens and positionless
  scene compression with a direct metric depth-to-2-D-control interaction, then
  run a short source-truth gate before another full training.

## E002 — direct metric depth-to-plan interaction

- Hypothesis: retain all four frames as metric-positioned visual tokens and use
  shared 2-D control tokens so every predicted control point can directly attend
  to obstacle geometry, while PointGoal remains a separate intent query.
- Model/data: 33.8M parameters; seven 2-D Cartesian B-spline control tokens;
  384 depth tokens plus three history-state tokens; one deterministic improved
  MeanFlow step and MeanFlow imitation as the only training objective. Same
  25,777/6,064 source-certified train/validation data as E001.
- Training: 4x RTX 4090, global batch 1024, 8,000 steps / 200 epochs; stable
  throughput about 4.05k samples/s; final loss 0.01508; no OOM, NaN, restart, or
  skipped update.
- Final artifact: `/home/hsb/navigation_three_projects/curvenav/outputs/offline-e002-step8000`
  on `go-4x4090`, checkpoint step 8,000, EMA weights. Full 6,064-sample audit
  took 35.32 s including three diagnostic interventions; base-policy batch-32
  throughput was 920.7 observations/s and batch-1 inference was 29.28 ms P50.
- Final source-truth result: ADE 0.03514 m; full-path collision 13.98%; first
  0.5/1.0 m collision 0.066%/0.676%; mean first-collision distance 2.740 m.
  Forward-direct collision is 2.79%, forward-detour 33.26%, rear-goal 70.0%,
  and expert-moves-away 73.99%. Fixed-MPC mean desired speed is 0.443 m/s and
  curvature limits 5.80% of plans.
- Causal audit: swapping the complete four-frame depth condition between paired
  real samples changes the path by 0.1225 m and raises full/first-1m collision
  to 32.49%/3.68%; swapping only real PointGoals changes it by 0.1373 m and
  raises collision to 31.13%/1.81%. The policy uses both inputs; depth removal
  is more damaging to immediate safety, so “depth is ignored” and “PointGoal is
  the sole override” are both rejected.
- Failure localization: 848 predicted curves collide in source truth, but only
  528 (62.26%) contain a collision point that the four-frame raw depth field
  itself recognizes; recognized points are 38.30% of all source-collision
  points. Together with 0.676% first-metre collision, this separates visible
  generation errors from an unobservable distant tail. Full-path collision is
  a strict risk metric, not a prediction that the receding-horizon robot will
  immediately collide.
- Visibility/terminal audit on the same 6,064 outputs: among the 848 colliding
  curves, 457 (53.89%) hit an obstacle already recognized by the current frame,
  71 (8.37%) are recognized only after adding history, and 320 (37.74%) remain
  unrecognized by all four frames. The physical endpoint collides in 458
  curves; at the endpoint itself 125/22/311 are
  current-visible/history-only/unrecognized.
  The final 0.25 m contains a collision in 544 curves, but only 110 collisions
  are confined to that terminal window. History is causally active but weak:
  it changes the path by 0.0413 m on average, lowers collision from 15.09% to
  13.98%, and makes 22/74 (29.73%) current-only trajectories safe when their
  collision obstacle is visible only through history.
- Root conclusion: training after step 3,200 reduced loss from 0.08861 to
  0.01508 but changed full collision only from 14.07% to 13.98%. The remaining
  failure is therefore not insufficient optimization. Low ADE also hides a
  derivative error: predicted curvature variation is 235.01 1/m2 versus 7.27
  for experts, and full-path P95 maximum curvature is 21.74 1/m versus 2.53.
  Absolute Cartesian control regression learns position closely but does not
  preserve the strong neighbouring-control correlations that define a stable
  tangent field; detour/rear modes remain underfit. The fixed MPC sees only the
  first 12 points, where P95 curvature is 1.22 1/m and only 5.80% of plans are
  curvature-limited. Thus the derivative defect is real but is not yet proven
  to lie in the executed prefix; only a real closed loop can determine whether
  distant errors repeatedly enter it.
- Decision: do not add a clearance loss, scorer, or hard curvature projection.
  Keep E002 frozen until a fixed online trace is available. If online confirms
  replanning instability, the next single architectural experiment is an
  invertible Euclidean control-increment Flow coordinate, whose linear
  cumulative decode directly supervises B-spline tangent geometry without a
  constraint or second objective.
- Online status: not yet measured. On 2026-09-01 the 3090 endpoint was
  unreachable from both the restored local aTrust route and `go-4x4090`; this
  is infrastructure state, not a CurveNav result. Do not transfer any older
  checkpoint's closed-loop score to E002.

## E003 — geometry-first path-relative incremental MeanFlow

- Hypothesis: E002's remaining visible collisions and derivative error come
  from two coupled representation defects: absolute controls are poorly
  conditioned, and PointGoal changes scene retrieval before obstacle geometry
  is grounded. Replacing both boundaries inside the one generator should lower
  visible/history-only collision without a safety loss or selector.
- Pre-implementation evidence: standardized absolute-control covariance has
  condition number `23800.8`; standardized increments reduce it to `278.2`
  (`85.6x`). E002 has 528/848 colliding curves with raw observable collision
  evidence, so partial observability alone cannot explain its failure.
- Architecture: standardized unconstrained control increments; goal-independent
  depth cross-attention; multiplicative PointGoal modulation only after geometry;
  the exact MeanFlow estimate `e_hat=z_t-t*v_theta` supplies physical control
  anchors for path-relative attention in the interval-average field. One
  improved-MeanFlow loss and one deterministic inference call remain.
- Rejected alternatives: DPNet dead-end head/virtual wall/primitive library;
  SanD/NavDP candidate scorers; clearance penalties; hard horizon, curvature,
  or collision projection. They add a second decision mechanism without fixing
  the generator's representation boundary.
- Acceptance gates: full mathematical and gradient suite; deterministic compile
  smoke; at least `3000 samples/s` steady 4x4090 throughput; then the complete
  6064-sample source-truth audit. Primary comparisons are current-visible,
  history-only, first-1m collision, detour collision, derivative quality, ADE,
  and arc-length/progress. Fixed 10-episode online evaluation remains mandatory.
- E003a rejection before full training: `100 passed, 4 skipped`; remote compiled
  CUDA smoke passed and four-RTX-4090 training was numerically stable at
  approximately `3.65-3.88k samples/s`. A step-2400 audit exposed a violated
  intent contract before spending all 8,000 steps. PointGoal distance is greater
  than the `3.6m` local horizon in 60.26% of train samples and reaches `23.31m`.
  The raw Cartesian goal MLP produced validation intent norms up to `56.08`.
  Across all intent channels/blocks, factors `1+scale(intent)` were negative for
  0.22% / 8.25% / 18.49% of samples in distance bins `(0,3.6]`, `(3.6,7.2]`,
  and `(7.2,inf)`. Thus distant mission scale could flip or amplify a local
  grounded update. The step-2400 checkpoint is retained as a rejected numerical
  artifact and is not a navigation result.
- Root correction for E003b: encode PointGoal as unit Cartesian direction plus
  `log1p(distance/3.6m)`, then RMS-normalize the intent embedding. This preserves
  direction and continuous distance without clipping or a trajectory-length
  constraint, while removing distance-proportional feature gain. No loss, head,
  inference branch, or model selection mechanism is added.
- E003b validation/start: full CPU regression is `101 passed, 4 skipped`; the
  remote compiled CUDA forward/backward/JVP/optimizer/deployment test is
  `1 passed in 32.71s`. Four-RTX-4090 training restarted from step zero with the
  same global batch 1024. At step 80 the loss is `0.91346`; post-compile
  throughput is `3.57-3.71k samples/s`, all four cards hold about `18.31GiB`,
  and no NaN, OOM, skipped update, or restart has occurred. The complete
  step-8000 source-truth and fixed online evaluations remain pending.
- E003b step-800 intent audit on all 6,064 validation goals passes the root
  numerical gate. Intent-norm min/median/max is `19.699/19.705/19.726`. For
  goal-distance bins `(0,3.6]`, `(3.6,7.2]`, `(7.2,inf)`, median absolute
  modulation is `0.532/0.518/0.480`, P99 is `1.966/1.777/1.651`, and negative
  `1+scale` fractions are `10.32%/8.92%/6.31%`. Unlike E003a, modulation no
  longer grows with global goal distance. Negative learned feature modulation
  still exists, but it is bounded by normalized intent and is not a far-goal
  numerical shortcut; navigation safety remains an empirical evaluation gate.
- The same audit at step 2,400 remains stable: intent norm is approximately
  `19.72`; near/mid/far median absolute modulation is
  `0.683/0.651/0.582`, P99 is `2.614/2.147/1.932`, and negative-factor fractions
  are `16.85%/14.85%/9.72%`. Learned modulation strengthens during training but
  still decreases rather than explodes with mission-goal distance.
- Design limitation to test rather than conceal: the same-block cross-attention
  is goal-independent before its PointGoal modulation, but a previous block's
  modulated residual is the next block's query. E003 is therefore a local
  geometry-before-intent ordering bias, not a formal non-interference theorem.
  Persistent current-visible collisions would directly reject this factorization.
- E003b completed all `8,000` updates on four RTX 4090s with global batch 1024,
  final loss approximately `0.01465`, steady-state throughput about
  `3.66-3.73k samples/s`, and no NaN, OOM, skipped update, or restart. The EMA
  checkpoint is `541,696,099` bytes and the strict evaluator independently
  reports `checkpoint_step=8000`.
- Complete 6,064-sample source-C-space evaluation rejects the safety part of
  the hypothesis. E003b versus E002 is: ADE `0.03580` versus `0.03514 m`, full
  collision `14.17%` versus `13.98%`, detour collision `32.92%` versus
  `33.26%`, rear-goal collision `77.62%` versus `70.00%`, and moves-away
  collision `78.03%` versus `73.99%`. The `0.18 pp` aggregate change is below
  one binomial standard error and is not an improvement. Current-visible
  colliding trajectories remain `450`; history-only collisions fall from 71
  to 46 while unrecognized collisions rise from 320 to 363. PointGoal and
  depth swaps still change paths by `0.1359 m` and `0.1230 m`, so neither input
  is disconnected.
- The incremental coordinate is useful but insufficient: curvature variation
  falls `235.01 -> 159.71 1/m2`, P95 maximum curvature `21.74 -> 15.29 1/m`,
  and first `0.5/1.0 m` collision falls `0.066/0.676% -> 0.016/0.511%`.
  Endpoint collision nevertheless rises `7.55% -> 7.97%`. This local-prefix
  improvement may matter under receding-horizon execution, so fixed online
  evaluation is still required, but it cannot be reported as full-path safety.
- A separate read-only geometry audit rules out the seven control anchors as
  the dominant error. On the validation expert distribution, control-to-curve
  nearest distance is `0.016 m` mean, `0.049 m` P95, and `0.096 m` P99, below
  the model's local metric-token scale. Densifying this attention would add
  substantial cost without addressing the observed failure.
- Root conclusion: marginally normalizing PointGoal and ordering one block's
  depth read before its goal modulation fixes the far-goal numerical shortcut,
  but does not create a target-independent traversability decision. Goal-
  modulated residuals become the next block's scene query, and the goal-shaped
  instantaneous proposal also defines the path-relative lookup. The surviving
  current-visible shortcuts and severe rear/moves-away failures are direct
  evidence that the generator still fits progress more easily than rare safe
  detours. Do not add DPNet virtual walls, a scorer, more candidates, a
  clearance penalty, or dense path stages in response. First obtain the fixed
  closed-loop trace; any E004 must replace this factorization as one coherent
  architecture rather than layer another safety mechanism on top.
- Final artifacts are `outputs/offline-e003b-final/offline-metrics.json`,
  `offline-cases.json`, `offline-cases.svg`, and `offline-cases.png`. Fixed
  online evaluation is still unavailable: the 3090 has no route from the 4090,
  and the local aTrust process is not running. No older checkpoint's online
  result is assigned to E003b.

## E004 — metric goal-reference path interaction

- Hypothesis: E003b fails because a learned PointGoal modulation repeatedly
  biases the hidden state and later scene queries. Replacing that semantic
  branch with a physical goal-reference corridor should reduce already-visible
  collisions while retaining the well-conditioned incremental B-spline and the
  sole improved-MeanFlow objective.
- Architecture delta: remove the PointGoal MLP and every per-block FiLM. The
  seven Greville controls of the exact straight B-spline from the origin to
  `min(||g||,3.6m)` provide the first field's metric path-relative queries. Flow
  transports standardized expert-minus-reference control increments, so its
  learned signal is route-shape deviation rather than common goal progress. The
  instantaneous field's exact clean estimate provides the second field's
  queries. The affine residual coordinate is unbounded and does not constrain
  output length or heading. No loss, head, candidate, scorer, projection, or
  inference pass is added.
- Final pre-run audit also removed a provisional learned embedding of the
  reference controls before it produced any optimizer step. That embedding
  would have recreated a goal-only hidden-state path. In the accepted graph,
  PointGoal affects learned computation only through scene-relative metric
  query geometry; its other role is the analytic affine coordinate origin.
- Pre-training identifiability audit on all 6,064 validation samples: the exact
  goal-reference spline violates the `0.10 m` source margin in 45.66% of cases.
  Safe-reference versus blocked-reference expert residual RMS is `0.329` versus
  `1.027` (`3.13x`); lateral residual RMS is `0.171` versus `1.083` (`6.35x`).
  Although blocked references are 45.66% of samples, they carry 91.26% of the
  standardized residual-coordinate squared energy. Thus the sole Flow target
  now gives obstacle-conditioned detours the dominant learning signal rather
  than letting common straight progress swamp them. This is evidence of
  identifiability, not a claim of guaranteed collision freedom.
- Pre-training verification: source dataset recompilation preserves all expert,
  depth, goal, and provenance arrays while replacing only the coordinate
  contract; `102 passed, 4 skipped` locally. A compiled RTX 4090 BF16
  forward/backward/JVP/optimizer/deployment smoke passes with finite loss and a
  finite `[4,64,2]` path.
- Run status: E003b's completed checkpoint was preserved as
  `outputs/train_policy-e003b-step8000` on `go-4x4090`. E004 started from zero
  on four RTX 4090s with global batch 1024. The first checkpoint is step 800;
  training will be stopped there for the complete source-truth gate before any
  continuation to step 2,400 or 8,000.
- Evidence gate: before a full run, tests must prove that scene memory is target
  independent, the target enters only through metric references, both fields
  are path-relative, and one-step/JVP gradients are finite. A step-800 then
  step-2400 complete source-truth audit must show a substantial reduction in
  current-visible, rear-goal, and moves-away collision; otherwise reject E004
  without spending 8,000 steps. The final unchanged physical target is at most
  1% full source-C-space collision, reported without excluding OOB, endpoints,
  unobserved regions, or short trajectories.
- Rejection: the step-800 gate had loss `0.486` and source collision `33.53%`
  (first 0.5/1.0 m `4.39%/12.48%`, detour `70.93%`, rear-goal `92.38%`).
  This checkpoint was under-trained and is not used as a performance verdict.
  The architecture was rejected before continuation for a structural reason:
  its codec always added the straight PointGoal reference back to generated
  residuals. That makes model error physically default toward the exact unsafe
  corridor the model is supposed to reject, contradicting the intended
  geometry-before-goal factorization.

## E005 — observed C-space BEV with goal-indexed physical increments

- Hypothesis: a single expert-imitation generator can learn visible avoidance
  only if robot-footprint-aware geometry is explicit, target independent, and
  upstream of goal selection, while its output coordinates contain no analytic
  goal trajectory.
- Architecture: all four frames share one ResNet-18-style encoder and calibrated
  SE(2) projection. Learned visual evidence is metric-splatted and fused with a
  64x64 observed Dingo C-space into one 16x16 BEV. Unknown EDT extrapolation is
  masked. Seven PointGoal Greville anchors are relative-attention queries only.
  Flow generates standardized physical B-spline increments. A four-block
  shared trunk feeds parallel four-block instantaneous and average iMF fields.
- Objective/inference: unchanged sole improved-MeanFlow imitation objective,
  exact deployment-boundary quarter and fixed typical latent; one deterministic
  call. No clearance loss, candidate, critic, scorer, projection or fallback.
- Root bug fixed during preflight: normalized max-range depth was previously
  excluded from visibility even though it is calibrated negative ray evidence.
  It is now observed-free up to 5 m while still excluded from obstacle/surface
  classification. Validation expert-point support rises to `81.86%`, fully
  observed expert curves to `51.10%`, and median contiguous observed prefix to
  `2.31 m`.
- Pre-training gate: the complete local suite passes `104 passed, 4 skipped`
  (two optional plotting imports and two local-CUDA tests). The RTX 4090
  compiled BF16 forward/backward/JVP/optimizer/deployment test passes in
  `153.61 s`. The prepared manifest is rebuilt with the exact physical-increment
  statistics. The final 4x4090 run started from step zero with global batch
  1024; steps 20/40/60 report `4.079/4.265/4.278k samples/s`, about
  `16.7 GiB/GPU`, finite loss and no skipped update. Its first physical
  checkpoint is step 800.
- Acceptance: first-1m and current/history-visible source collision below 1%,
  with ADE, arc length, progress and fixed-MPC curvature non-regressing. Full
  3.6m collision remains reported, but local depth alone cannot guarantee its
  unobserved tail; no score may hide OOB, endpoints or unrecognized collisions.
- Final run: step 8,000 completed without a skipped update. Final loss is
  `0.02051`; the last logged four-GPU throughput is `4.054k samples/s`.
  Batch-one model latency on one RTX 4090 is `38.66 ms` median / `38.96 ms`
  p95 (`25.86/25.67 Hz`), and batch-32 policy throughput is `710.90
  observations/s`.
- Final source-truth result: ADE is `0.03395 m`; collision in the first
  `0.5/1.0 m` is `0.033%/0.297%`. Forward-detour first-1m collision is
  `0.907%`. These execution-prefix gates pass and are materially better than
  E003. Full 3.6m collision is still `12.24%` (`27.99%` forward-detour,
  `66.67%` rear-goal), so the complete architecture is not accepted as a
  sub-1% full-horizon generator.
- Failure location is now explicit: mean distance to first collision is
  `2.76 m`; endpoint and terminal-0.25m collision are `7.40%` and `8.58%`.
  Of 742 colliding trajectories, 399 are unrecognized by all four depth frames,
  328 have current-frame evidence and 15 only historical evidence. Depth and
  PointGoal swaps change the path by `0.128 m` and `0.126 m`, respectively;
  the model no longer exhibits a simple PointGoal-only shortcut. The remaining
  problem is predominantly far-tail observability and long-horizon shape
  accuracy, not failure to generate a safe immediate prefix.
- Executability audit: the fixed controller's first-12-point curvature p95 is
  `1.153 m^-1` and only `4.90%` of plans are speed-limited, but full-curve max
  curvature p95 is `17.48 m^-1` versus `2.53 m^-1` for experts. This tail-shape
  gap must be distinguished from immediate collision before changing the
  trajectory coordinate system. The next evidence gate is the unchanged
  fixed online 10-episode protocol; no safety loss, selector or hard curvature
  constraint is added from this offline result alone.
- Fixed online result: checkpoint step 8,000, Home scene, seed 1234 and ten
  episodes produced `SR=1/10` and mean `SPL=0.092984`. Nine failures timed out.
  The policy and MPC did not output zero: plans averaged about `2.82 m`, MPC
  desired speed averaged about `0.26 m/s`, and curvature limiting was active on
  91% of plans. Yet actual motion repeatedly stalled. Closed-loop plans had
  p95 peak curvature about `310 m^-1`, far beyond both teacher-forced prediction
  (`17.48 m^-1`) and expert (`2.53 m^-1`), while adjacent replans disagreed by
  only `5.3 mm` over the first metre. The model was stably generating an unsafe
  out-of-distribution shape, not randomly jittering or being actively stopped.
- Evaluation audit: official success, timeout, SPL, PointGoal transform, output
  origin insertion and MPC axes are internally consistent and cannot turn a
  successful run into this 1/10 result. Two bugs existed only in the added map
  diagnostic: PLY samples were shifted by half a cell, and the current robot
  origin was included in every future-plan collision query. They inflated the
  diagnostic collision percentage but did not change official SR/SPL.

## E006 — Flow-candidate-grounded observed geometry

- Root defect: E005's function named `_path_relative_geometry` was anchored at
  the straight PointGoal reference, not at the current Flow state. The decoder
  therefore read obstacles around the desired corridor but never queried
  whether its own noisy/candidate B-spline occupied them. This breaks the
  state-conditioned geometry principle used by NavDP's noisy-trajectory
  queries and leaves an easy PointGoal shortcut.
- Single architectural correction: decode every Flow state into its physical
  64-point B-spline and query the deployed observed C-space along that curve.
  Aggregate path geometry to the seven controls using the fixed positive
  B-spline basis. Scene relative attention is also anchored at candidate
  controls. PointGoal appears only as the relative vector from each candidate
  control to its metric local goal reference.
- Mathematical contract: the decoder is now
  `u_theta(z,r,t,D,g)=u_theta(z,r,t,M_D,H(z,D),G(z,g))`. The stopped JVP includes
  the differentiable `z -> B-spline -> bilinear C-space query` path almost
  everywhere. Unknown field corners contribute zero geometry and explicit zero
  coverage, not extrapolated free space.
- Scope: one Transformer, one improved-MeanFlow loss and one deterministic
  inference call remain. No clearance loss, hard constraint, candidate set,
  critic, selector, refinement pass or fallback was added. The architecture and
  checkpoint type change, so E005 weights are intentionally incompatible.
- Verification/start: focused forward, condition-causality and gradient tests
  pass; a compiled RTX 4090 BF16 forward/JVP/backward/optimizer/deployment smoke
  passes with finite loss and a finite `[4,64,2]` path. Training started from
  zero on free RTX 4090 GPUs 0--2 with exact global batch 1024; GPU3 belongs to
  another user and was not touched. Steady steps 40--100 reach
  `3.19--3.25k samples/s`, about `21.7--21.8 GiB/GPU` and 95--100% sampled GPU
  utilization; loss falls from `2.54` at step 1 to `1.07` at step 100. No E005
  score is attributed to E006.
- Step-800 source-truth gate: loss is `0.314`; ADE `0.04783 m`; full collision
  `14.87%`; first `0.5/1.0 m` collision `0.000/0.841%`; forward-detour first-1m
  collision `1.586%`; full detour/rear/moves-away collision
  `33.65/71.90/74.57%`. Full-curve curvature p95 is `14.37 m^-1` and fixed-MPC
  curvature limiting `5.24%`. This early checkpoint is not yet better than the
  converged E005 and its loss is far from convergence, so it does not establish
  the hypothesis; training resumed exactly from step 800 for the next gate.
- Final run: step 8,000 is readable with checkpoint SHA256
  `f3025da23a7ee67ece9eb302bc30118ab1623cfa3c7ff4de7e9c2e6bbc71a35b`.
  Three RTX 4090s sustain about `3.1k samples/s` with approximately
  `21.7--21.8 GiB/GPU`; the final logged loss is approximately `0.014`.
  The source-truth result is ADE `0.03448 m`, full collision `12.78%`, first
  `0.5/1.0 m` collision `0.016/0.396%`, and forward-detour first-1m collision
  `0.963%`. Full forward-direct/detour/rear/moves-away collision is
  `2.52/29.07/75.71/79.77%`. Endpoint and terminal-0.25m collision is
  `7.85/9.10%`; full-curve maximum-curvature p95 is `15.61 m^-1` versus
  expert `2.53 m^-1`. Batch-one latency is `43.90 ms` median / `45.46 ms`
  p95 (`22.78/21.997 Hz`).
- Offline decision: E006 did not improve teacher-forced source safety. Relative
  to E005, full collision changes `12.24 -> 12.78%`, first-1m collision
  `0.297 -> 0.396%`, detour first-1m collision `0.907 -> 0.963%`, and ADE
  `0.03395 -> 0.03448 m`. Querying the current Flow state is therefore not a
  direct final-trajectory/geometry contract: at one-step deployment that state
  is still the fixed source latent.
- Fixed online result: E006 reaches `5/10` success and mean SPL `0.450821` under
  the same Home scene, seed 1234 and `num-envs=1` protocol where E005 reached
  `1/10` and `0.092984`. Candidate-relative C-space is therefore retained as a
  real closed-loop improvement even though its teacher-forced safety did not
  improve. The five failures are timeouts. Failed and successful episodes have
  comparable nonzero command magnitude, but failed actual speed collapses to
  about `0.023 m/s`; replans remain stable and repeatedly point through blocked
  corners. This localizes the next defect to generated-trajectory geometry, not
  random planning, zero commands or the official metric.
- Online evaluator root audit: the real persistent-scene implementation passes
  raw `distance_to_image_plane` depth and the simulator `3x3` intrinsic matrix;
  PointGoal is `R^T(goal-world - robot-world)` and CurveNav consumes its body
  `x/y` directly; the runtime returns 63 future points and the evaluator inserts
  exactly one origin before the unchanged MPC. `EvalTerminationsCfg` contains
  only arrival and timeout, while trace aggregation independently recomputes
  every SR/SPL row. Thus no metric, axis, transport or termination bug can turn
  a genuine success into the E005 `1/10` result. The YAML field
  `arrival_threshold: 1.0` is unused: the executable fixed condition is
  `<0.5 m`, `<0.25 m/s`, then 40 accumulated Dingo control steps. Its upstream
  timer latches once started, which can only inflate success and remains
  unchanged for cross-model comparability. CurveNav's server now uses the
  supplied intrinsic matrix. Its separate map diagnostic now uses native 5 cm
  cell centres, a half-open raster extent, and excludes the current origin from
  future collision; these corrections do not alter official SR/SPL.
- Paper falsification before the final gate: NavDP grounds denoising tokens in
  the current noisy trajectory, which motivates E006, but it updates that
  trajectory over multiple denoising steps. The contemporaneous FlowPilot
  result instead couples a future-depth stream and a polynomial-action stream
  through shared attention; its reported depth-frozen/action-only intervention
  raises collision from `4%` to `26%`, and deployment normally uses three Euler
  steps. This is evidence that explicit future-geometry/action co-training can
  matter; it is not evidence that a loss term should be appended to E006. If
  the converged E006 still fails on currently visible prefix collisions, the
  next architectural hypothesis must compare one-step candidate grounding
  against a single jointly trained geometry-action process. No future-depth
  head, extra denoising step or safety penalty is introduced mid-run.

## E007 — single-call clean-proposal-grounded improved MeanFlow

- Rejected pre-run draft: a parameter-shared three-step MeanFlow was briefly
  implemented but never trained. Its training still gave exact fixed-source
  coverage to `(r,t)=(0,1)`, while deployment called the distinct intervals
  `(2/3,1)`, `(1/3,2/3)`, `(0,1/3)` on policy-updated states. CPU tests proved
  execution, not that statistical contract. It was removed when the production
  requirement was reaffirmed as single-step; no checkpoint or score belongs to
  that draft.
- Root hypothesis: E006's one C-space query is performed on its fixed Gaussian
  source. A one-call generator needs an estimate of its clean endpoint before
  the average field can read output-relevant obstacle geometry.
- Architecture: retain improved MeanFlow's supervised instantaneous field.
  Six proposal blocks use target-independent scene memory and the metric goal
  corridor to predict `v_theta`. The exact linear-interpolant form supplies the
  learned endpoint estimate `x_tilde=z_t-t*v_theta`. It is decoded immediately;
  the next six blocks query observed C-space and BEV at this proposal and use
  candidate-relative PointGoal displacement to predict `u_theta`. Deployment
  remains exactly `x_hat=e*-u_theta(e*,0,1,c)` with one decoder call.
- Objective: unchanged single improved-MeanFlow scalar, the equal mean of
  instantaneous velocity regression and reparameterized interval-average
  regression. The JVP differentiates through proposal decode and geometry
  query. There is no clearance penalty, future-depth loss, critic, candidate
  set, selector, projection, rollout loss or inference loop.
- Mathematical gate: tests must prove one deployment call, distinct reference
  and learned-proposal geometry queries inside that call, structural
  independence of `v_theta` from `r`, fixed-source `(0,1)` parity, finite JVP
  and full-model gradients, and strict checkpoint incompatibility with E006.
- Performance gate: compile BF16 on RTX 4090, measure parameters/latency and
  sustain at least `3000 samples/s`; stop first at step 800 for all 6,064
  source-truth samples. Continue only if first-1m, visible-detour, progress and
  curvature move together. Final evidence remains the unchanged ten-episode
  closed loop, followed by a larger protocol before any `80%` claim.
- Pre-training verification: the complete CPU suite is `108 passed, 4 skipped`;
  the skipped cases require local CUDA or optional plotting. A real RTX 4090
  compiled BF16 forward/JVP/backward/optimizer/deployment smoke passes in
  `256.57 s`, including a finite one-step `[4,64,2]` output. The instantiated
  policy/decoder contain `33,898,916/29,018,980` parameters. Training started
  from zero on the only free RTX 4090 GPU2 with global batch 1024; other users'
  GPUs 0, 1 and 3 were not touched. No offline or online score is claimed yet.
- Step-800 gate: training is finite and sustains `1.00--1.02k samples/s` on one
  RTX 4090 at about `22.1 GiB`. Versus E006 at the same step, ADE changes
  `0.04783 -> 0.04223 m`, full collision `14.87 -> 13.49%`, first-1m collision
  `0.841 -> 0.511%`, forward-detour collision `33.65 -> 31.50%`, and rear-goal
  collision `71.90 -> 63.33%`. Fixed-MPC first-12 curvature p95 improves
  `1.35 -> 1.13 m^-1` and limiting falls `5.24 -> 4.53%`.
- The gate is not yet accepted as a safe model. Full-curve maximum-curvature
  p95 regresses `14.37 -> 21.79 m^-1` and complete comparison-horizon coverage
  falls `97.79 -> 85.32%`, exposing an under-converged short/sharp tail. The
  exact step-800 checkpoint is preserved; training resumes to step 2,400 to
  test whether both defects converge away before any online evaluation.
- Case evidence prevents attributing all remaining collision to unseen space.
  The selected current-visible failure first collides at `0.725 m` and contains
  11 collision points recognized by the current depth frame; the hard detour
  case first collides at `1.30 m` with 10 current-visible and 73 unrecognized
  collision points. E007 therefore still has both a learned visible-geometry
  error and an unobservable-tail error at step 800. Step 2,400 distinguishes
  under-training from a surviving architecture defect; no new loss is added
  while that distinction is unresolved.
- The first continuation from step 800 reached approximately step 1,140 at
  about `1.0k samples/s`, then its remote tmux session exited before the next
  checkpoint. The preserved checkpoint is still exactly step 800. The local
  aTrust tunnel subsequently disappeared, so kernel/tmux exit evidence has not
  yet been retrieved and no later checkpoint is claimed. The next launch must
  retain the dead tmux pane and exit status before continuation resumes.
- Because the 4090 aTrust route remained unavailable, the unchanged E007 code
  was validated on local V100 GPU6: compiled FP16 forward, MeanFlow JVP,
  backward, optimizer step and deployment sample passed in `212.19 s`. A fresh
  two-rank run briefly started on otherwise empty V100 GPUs 6/7 with the same
  global batch 1024. It was stopped cleanly before producing a checkpoint once
  the 4090 route recovered, avoiding two independent trainings of the same
  experiment; its retained tmux pane records signal 2.
- A second, bounded V100 6/7 throughput probe was made after the 4090 resumed.
  Both ranks remained in CUDA/NCCL initialization for more than seven minutes
  with only about `0.63 GiB` allocated per GPU and never emitted
  `training_start`; the probe was stopped cleanly. In the same interval the
  single 4090 advanced from step 800 to beyond 1,060 and sustained
  `1013--1019 samples/s`. For this checkpoint and remaining wall time, the
  measured faster route is therefore the single 4090; no model or launcher
  branch is introduced for the V100 host.
- The 4090 route was restored through the already-running user-namespace
  aTrust `utun7`; authentication had not expired. Remote GPU2 was empty and
  showed no kernel OOM or NVIDIA Xid around the prior exit. Training resumed
  from the exact step-800 checkpoint in `curvenav-e007`, with tmux now retaining
  the dead pane and exit status. The unchanged watcher stops after the step
  1,600 and 2,400 checkpoints.
- Closed-loop rendering was refined without changing SR, SPL, timeout, MPC or
  proxy metrics: episode and scene maps now crop to the actual task region,
  generated plan samples outside frozen robot-center free space are overlaid
  in red, and executed positions outside that proxy are marked separately.
  The complete CPU regression after the evaluation diagnostics is `110
  passed, 4 skipped`.
- Step-2,400 final gate: the resumed single RTX 4090 run sustained
  `1013--1021 samples/s`, saved SHA-256
  `45079517e3a3e6f45d05ab8b3fc9e1a599d73c622b28ddcffc28f6f351bb7ddc`,
  and stopped cleanly. Loss reached `0.04254`. On all 6,064 source-truth
  validation items, ADE is `0.03559 m`, full collision `12.269%`, first
  `0.5/1.0 m` collision `0.016/0.363%`, forward-direct/detour collision
  `2.470/28.045%`, and rear/moves-away collision `70.476/75.145%`.
  Continuing from step 800 improves aggregate collision by only `1.22 pp` and
  worsens rear-goal collision by `7.14 pp`; training duration is not the root
  safety variable.
- Proposal/final falsification: proposal collision is `13.391%`, final
  collision `12.269%`, mean path change is `0.01493 m`, proposal-safe to
  final-collision is `0.792%`, and proposal-collision to final-safe is `1.913%`.
  Hence the PointGoal-conditioned final update is not destroying an already
  safe proposal. The proposal itself never learned a safe manifold. Of 744
  final colliding curves, `45.56%` contain current-frame recognized collision,
  `4.03%` history-only recognized collision and `50.40%` no four-frame
  recognized collision; visible learning failure and unobservable tail are
  both real.
- E007 is rejected as the final architecture. Candidate-relative geometry is
  retained because E006 improved closed loop, but querying geometry while both
  Flow heads optimize only expert coordinate error is not a feasibility
  objective. No additional E007 epochs or online score will be used to hide
  this result. The 3090 VPN route is present but SSH port 22 remains unreachable.

## E008 — deployment-curve observed-feasibility MeanFlow

- Root hypothesis: safe expert labels identify the desired route, but finite
  regression error has no mathematical implication for signed clearance. With
  source-safe expert `p+`, `d(p+)>=0.10 m`, safety would follow only from a
  uniform path error below the margin; E007 has mean FDE `0.0947 m`, so low ADE
  cannot supply that bound. SanD resolves this with ESDF selection, NavDP with
  a learned critic and privileged negative paths, and ViPlanner/iPlanner use a
  differentiable trajectory cost. A single-output CurveNav that rejects those
  inference mechanisms requires direct final-path feasibility training.
- Unique change: keep one E007 decoder call and one output, but evaluate the
  exact deployment-quarter final path against its own raw-depth C-space at
  2.5 cm. The sole added scalar is strict-observed squared clearance deficit
  below `0.10 m`. It has no network parameters and is absent at inference.
  Every bilinear support cell must be observed, so unknown EDT extrapolation
  cannot become free-space evidence. Fixed-grid normalization and detached
  radial scale remove coverage normalization and uniform-shortening shortcuts.
- This is one constrained generator objective `L_MF+L_vis`, not a safety head,
  candidate scorer, hard projection, privileged teacher, fallback or second
  launcher. PointGoal and depth remain joint conditions; “safety first” is the
  feasible-set semantics of the output, not an unsupported depth-only policy
  that must guess one route before seeing the goal.
- Pre-run CPU gate: policy/checkpoint/evaluation tests are `54 passed, 1
  CUDA-skipped`; tests explicitly prove strict unknown masking, clearance
  gradient direction, zero uniform-shortening gradient, exact loss
  decomposition and full-model finite gradients. CUDA compile, throughput and
  the step-800 source-truth gate remain required before any E008 score.
- Step-800 gate: single-4090 steady throughput is `970--994 samples/s` at
  `22.1 GiB`, only about 4% below E007. The frozen checkpoint SHA-256 is
  `a5e2fdce942c490635a01ec31e4f28f811311739b911a20fda16c154f5521dd0`.
  Against E007 at the same step, source collision changes
  `13.489 -> 11.412%`, forward-direct `3.155 -> 1.859%`, detour
  `31.501 -> 28.215%`, rear `63.333 -> 56.190%`, and moves-away
  `64.162 -> 56.647%`. Current-frame-recognized colliding curves fall
  `418 -> 265` (`-36.6%`), direct evidence that final-path geometry gradients
  work.
- The gate is not yet an online candidate. ADE changes `0.0422 -> 0.0505 m`,
  first-1m collision `0.511 -> 0.742%`, first-12 curvature p95
  `1.127 -> 1.341 m^-1`, and mean progress regret grows to `0.0302 m`.
  Unrecognized colliding curves rise approximately `375 -> 400`. A repeated
  support audit measures strict raw-depth coverage `82.57%` for predictions
  versus `84.59%` for their experts (`-2.02 pp`; 33.2% below reference).
  This is a warning, not yet proof of a dominant unknown-space shortcut.
- Decision: resume only to step 1,600, then repeat the complete gate. Accept
  the simple objective only if visible collision keeps falling, ADE/curvature
  recover and the support gap does not expand. Do not append an unknown-space
  penalty from one under-converged checkpoint.
- Step-1,600 gate passes that decision. Versus step 800, full source collision
  changes `11.35 -> 9.96%`, detour `28.10 -> 22.66%`, first-1m
  `0.742 -> 0.462%`, ADE `0.0505 -> 0.0430 m`, progress regret
  `0.0302 -> 0.0091 m`, and first-12 curvature p95 `1.333 -> 1.289 m^-1`.
  Strict observed support recovers from `-2.02` to `-0.36 pp` relative to each
  expert, disproving a growing unknown-space shortcut. Current-visible
  colliding curves fall again from `264 -> 193`; unrecognized curves remain
  nearly fixed at about `388`, exposing the local sensor's remaining limit.
  Checkpoint SHA-256 is
  `e9652616c46b63bccd2f0547f66ef86450fbc86217112b351435860862c6169f`.
- Proposal/final strata also reject the claim that the final PointGoal update
  breaks a safe path: all/detour/rear proposal collision is
  `11.18/25.78/67.14%`, while the final is `9.96/22.66/64.76%`.
  Rear-goal failure is already present in the proposal and is dominated by
  unrecognized collision (`53.81 pp` of `64.76%`); the model also advances
  `0.280 m` more toward the mission goal than the locally retreating expert.
  Without topology memory, this is not identifiable from visible clearance.
- Decision: keep E008 unchanged and continue only to step 2,400 for the final
  convergence gate. Do not add a PointGoal gate, unknown penalty, critic or
  extra stage. The fixed online ten-episode run remains required, but the 3090
  VPN currently reaches the route while SSH port 22 remains unavailable.
- Step-2,400 rejects further blind training. The single RTX 4090 sustained
  `973--979 samples/s`; checkpoint SHA-256 is
  `333b3a0aff2c09b8ef13e595a63111d4a1d456f87d721272be4b05ae400b94cc`.
  ADE improves `0.0430 -> 0.0368 m` and first-1m collision improves
  `0.478 -> 0.379%`, but full collision regresses `9.96 -> 10.41%`, detour
  `22.66 -> 23.40%`, rear `64.76 -> 69.05%` and moves-away
  `65.32 -> 69.36%`. Step 1,600 remains the safer checkpoint; lower imitation
  error is not evidence of better geometric feasibility.
- The earlier current/history/unrecognized trajectory labels meant that any
  collision point on a curve had that evidence, not that its first collision
  did. The corrected first-hit audit partitions every colliding curve exactly.
  At step 1,600, only `52/604=8.61%` first hits are recognized by the current
  frame, `12/604=1.99%` only by history and `540/604=89.40%` by neither. Across
  all 6,064 observations, first-hit current-visible collision is `0.857%` and
  history-only is `0.198%`; the remaining `8.905%` is unrecognized. Step 2,400
  shifts further toward unrecognized first hits (`583/631=92.39%`). Thus E008
  did learn visible avoidance below the 1% observation-level target, while its
  full-path score is now dominated by geometry absent from all four inputs.
- Root decision: do not increase the clearance weight, append an unknown-space
  penalty or add decoder stages. Those cannot identify hidden geometry. At the
  step-2,400 gate, step 1,600 was the provisional safer checkpoint; the later
  completed-run audit below supersedes that checkpoint choice. Treat the first
  `0.5/1.0 m` source-safe prefix as the local receding-horizon execution gate.
  A future architecture change is justified only if online traces show
  first-hit collisions in observed space; hidden long-horizon topology belongs
  to the planned topology-map input, not to a local-depth patch.
- The user-directed complete run reached step 8,000 without changing code,
  data or objective. Single-4090 throughput stayed near `970--980 samples/s`;
  the frozen EMA checkpoint SHA-256 is
  `20972dc0c7f28b6654ec16ada03c6239e46708e4b4ebb8493233dc2ec8b82e73`.
  The full 6,064-item source-truth audit gives ADE `0.03105 m`, full collision
  `11.560%`, first `0.5/1.0 m` collision `0.000/0.396%`, and mean first-hit
  distance `2.764 m`. Relative to step 1,600, imitation and the executable
  prefix improve, while full collision regresses `9.960 -> 11.560%`.
- The regression is localized rather than a new visible-avoidance collapse.
  Step 8,000 has 701 colliding curves: first hit is current-visible for 57
  (`0.940%` of all observations), history-only for 14 (`0.231%`), and absent
  from all four frames for 630 (`10.389%`). Terminal-0.25 m collisions rise
  from about 428 at step 1,600 to 516, accounting for 88 of the 97 added
  collisions; endpoint collision reaches 453. The correct conclusion is that
  the fully predicted 2.9 m tail increasingly enters unobserved space, not
  that the model forgot visible obstacles near the next executed prefix.
- The execution-prefix attribution closes the remaining ambiguity. Of the 24
  curves colliding within the first `1.0 m`, the first hit is current-visible
  for `0`, visible only through history for `1`, and unrecognized by all four
  frames for `23`. Thus no validation example drives the executed prefix into
  a source obstacle that the current raw depth C-space already recognizes.
  The one history-only case is a genuine temporal-fusion residual; the other
  95.83% are not identifiable from this local observation. This metric is now
  reported directly by the evaluator rather than inferred from selected cases.
- Geometry refinement remains real but small: proposal collision is `12.632%`,
  final collision `11.560%`, and their mean path difference is `9.16 mm`.
  Training also reduces curvature variation `594.9 -> 130.4 m^-2` and
  full-curve maximum-curvature p95 `21.46 -> 14.52 m^-1`, but both remain far
  above the expert's `7.27 m^-2` and `2.53 m^-1`. The controller-facing first
  12 points are much better behaved: curvature p95 `1.118 m^-1`, with `4.58%`
  speed-limited. Closed-loop execution, not another offline penalty, must now
  determine whether the remaining tail is harmless under replanning.
- Rear-goal and locally retreating-expert collision remain `70.48/71.68%`.
  The model changes its path by nearly the same amount under a depth swap
  (`0.1322 m`) and PointGoal swap (`0.1328 m`), so neither condition is absent;
  the unresolved ambiguity is that local depth cannot identify when global
  progress requires temporarily moving away from the goal. Do not encode that
  topology as a PointGoal heuristic or an unknown-space penalty.
- The fixed online ten-episode candidate is step 8,000 because it has the best
  executed-prefix collision, ADE and smoothness, despite its worse unobserved
  full tail. Online traces must record first-metre plan feasibility at every
  replan and overlay plans, executed motion and frozen C-space. If unsafe
  observed prefixes precede stalls, the generator remains wrong; if prefixes
  stay safe while the vehicle stalls, investigate deployment coordinates/MPC.
  The 3090 VPN route currently exists but port 22 is unreachable from both the
  workstation and the 4090 host, so this fixed online run has not started.
- Operational note: a one-off watcher used tmux prefix matching and confused
  `curvenav-e008-full-gate` with the completed `curvenav-e008-full` session.
  The step-8,000 checkpoint itself was complete and valid. The watcher was
  stopped and the unchanged evaluator was launched directly; future lifecycle
  checks must use exact tmux session targets rather than prefixes.
- Fixed online result: the exact step-8,000 EMA checkpoint reaches `6/10` SR
  and `0.582788` mean SPL on Home episode `0--9`, seed 1234 and `num-envs=1`.
  Successful episodes are `1,2,4,5,6,8`; failures are `0,3,7,9`. The identical
  first ten episodes give official NavDP `5/10` and X-NavDP `8/10`, so E008 is
  a credible supervised baseline but does not inherit X-NavDP recovery.
  In the four failures, adjacent first-metre replans differ by only
  `0.9--5.1 mm`, desired speed remains `0.36--0.50 m/s`, and stalled-step
  fractions are `66.4/80.3/95.9/98.4%`: stable unsafe intent, not stochastic
  switching or a zero-command controller, is the dominant signature. Frozen
  map overlays and numeric traces are stored under
  `outputs/online-e008-step8000-online10/20260901_115641`.

## E009 — Flow-state-grounded one-call MeanFlow

- Root defect: E008 places the first six blocks' C-space and BEV queries on the
  analytic straight PointGoal corridor. The final six blocks query the learned
  clean proposal, but offline they alter it by only `9.16 mm`; online failures
  repeat one wrong route for the entire timeout. The goal therefore still
  chooses where the first obstacle lookup occurs, contrary to the intended
  target-independent geometry contract.
- Paper/source basis: SanD's conditional U-Net and NavDP's Transformer denoise
  the current noisy trajectory variable while attending to depth/goal context;
  neither substitutes a straight PointGoal path for that variable. Their
  candidate scorer/critic remains deliberately excluded.
- Unique change: decode the affine standardized Flow state `z_t` to its physical
  controls and B-spline, and use those positions for the first C-space/BEV
  query. The learned clean estimate remains the second query. PointGoal is only
  relative intent from each physical control. Model size, block count,
  objective, source distribution, one-call inference and FLOP order are
  unchanged; no loss, head, candidate, projection or fallback is added.
- Mathematical gate: affine coordinate decode commutes with the linear Flow
  interpolant, so every `z_t` has an exact physical query curve and the JVP
  differentiates through both state and proposal queries. The fixed deployment
  source decodes to a finite `2.563 m` forward B-spline. Tests must prove that
  changing PointGoal changes goal features but cannot relocate the Flow-state
  geometry query before CUDA training starts.
- Pre-training verification passed on the current worktree: `114 passed, 4
  skipped` on CPU, followed by a compiled V100 FP16 forward, MeanFlow JVP,
  backward and deployment smoke (`2 passed` in `261.08 s`). A three-V100
  topology on local GPUs 0--2 entered NCCL initialization but made no training
  step; it was stopped before any checkpoint and all ranks exited. The same
  unmodified launcher immediately reached `training_start` on GPUs 0--1 with
  exact global batch 1024, 512 samples per rank, two 256-sample microbatches,
  FP16 and 8,000 target updates. This is a hardware communication-topology
  fact, not a model branch or a CurveNav loss result; throughput and model
  quality remain unreported until real updates are observed.
- Completed run: four V100S GPUs used global batch `1792`, `200` epochs and
  `4,600` optimizer updates, processing approximately the same number of
  examples as E008. Final loss was `0.10050`; the EMA checkpoint SHA-256 is
  `34b899677f32698a785fcd1ec5c8c0a1c0395a1cb344a95bde7ff69371320d2b`.
  Source-truth offline ADE/FDE are `0.03622/0.09933 m`, full collision is
  `11.807%`, and first `0.5/1.0 m` collision is `0.049/0.577%`. These are only
  slightly worse than E008 and do not predict the closed-loop regression.
- Fixed online rejection: on the exact resident Home scene, episodes `0--9`,
  seed `1234` and `num-envs=1`, E009 reaches only `3/10` SR and `0.296271`
  mean SPL, versus E008's `6/10` and `0.582788`. It loses E008 successes 1, 2
  and 8. Across all ten traces, actual-position free fraction falls
  `69.6 -> 41.2%`, commanded low-speed stall fraction rises
  `35.4 -> 64.1%`, free-origin first-metre plan collision rises
  `60.3 -> 69.9%`, and full-plan collision rises `81.0 -> 94.7%` on the same
  frozen benchmark proxy. This is stable wrong-route behavior, not a server,
  MPC, seed or episode mismatch.
- Rejected inference: exact affine decode proves only that `z_t` denotes a
  finite curve. Near the one-step deployment boundary it is still the fixed
  artificial Gaussian source, not an estimate on the clean navigation
  manifold. SanD/NavDP can repeatedly replace a noisy variable and select among
  candidates; that does not license a path-local C-space lookup on CurveNav's
  single-call source. E009's first lookup is mathematically defined but has the
  wrong navigation semantics.

## E010 — goal retrieval, clean-proposal geometry

- Root correction: restore the metric PointGoal reference as the first six
  blocks' spatial retrieval coordinate. It is explicitly an attention prior,
  not an executed trajectory, additive residual or hard goal corridor. The
  instantaneous field still receives the complete Flow state and predicts the
  clean endpoint. Only that learned clean proposal receives candidate-path
  C-space/BEV queries before the interval-average field produces the final
  B-spline.
- Scope: no extra block, head, candidate, critic, loss, projection, inference
  iteration or fallback. The four-frame encoder, target-independent observed
  BEV, improved-MeanFlow identity, deployment-boundary clearance objective and
  one-call runtime are unchanged. E009 code is removed rather than retained as
  a switch.
- Causal training comparison: keep E009's global batch `1792`, sample budget,
  FP16/FP32 precision contract and launcher for the first E010 run. This
  isolates the retrieval-location correction from the separate E008/E009
  optimizer-topology difference. Evaluate by source-truth validation and the
  same resident ten episodes before changing batch or learning rate.
- Completed run: four V100S GPUs reached `200` epochs and `4,600` optimizer
  updates at a final throughput of `2,820 samples/s`. The final loss was
  `0.06510`; the EMA checkpoint SHA-256 is
  `57a7bbabaeeb39f1632089915624acdde05f0d191aeb9dca70e7f0ccee0af3c9`.
  Source-truth offline ADE/FDE are `0.03758/0.10204 m`, full collision is
  `12.187%`, and first `0.5/1.0 m` collision is `0.000/0.495%`. Relative to
  E009, only the immediate execution prefix improves slightly; full collision
  and imitation error do not. The fixed resident ten-episode online run below
  remains the acceptance test.
- Fixed online rejection: on the same resident Home scene, episodes `0--9`,
  seed `1234` and `num-envs=1`, E010 reaches `3/10` SR and `0.298922` mean SPL.
  Successful episodes are `4,5,6`; E009 also reaches `3/10` and `0.296271`,
  while E008 reaches `6/10` and `0.582788`. E010 improves median final goal
  distance (`3.64 -> 2.54 m`), progress fraction (`0.412 -> 0.512`) and online
  plan peak-curvature p95 (`80.66 -> 36.63 m^-1`) relative to E009, but does
  not recover any additional success. The retrieval-order correction is
  therefore insufficient and must not replace E008 as the accepted baseline.

## E011 — target-independent global geometry, one proposal query

- Root diagnosis: E009 and E010 both use a non-proposal as the first
  path-local obstacle query. E009 uses the fixed Gaussian Flow source and
  reaches `3/10`; E010 uses the straight PointGoal ray and also reaches `3/10`.
  The former has no clean navigation semantics, while the latter lets target
  intent choose where geometry is read. Neither is repaired by more epochs.
- Paper/source basis: LoGoPlanner first extracts task-specific metric geometry
  and state context, then fuses goal intent into its diffusion policy. Its
  released runtime still uses ten denoising steps, sixteen samples and a critic;
  those mechanisms are explicitly excluded. CurveNav already has calibrated
  depth and known SE(2) history, so learned localization and point-cloud heads
  would duplicate more accurate inputs.
- Unique change: the first six blocks attend to the complete BEV/motion memory
  with metric bias relative to the robot origin. PointGoal remains in query
  content but cannot relocate memory coordinates, and no C-space path lookup
  occurs. The instantaneous field produces one clean proposal; only that
  proposal is decoded and queried against observed C-space before the final six
  blocks produce the MeanFlow average field.
- Complexity contract: one deterministic source, one decoder call, one clean
  proposal, one candidate C-space query and one final B-spline. No new block,
  head, candidate, critic, loss, projection, recurrent inference or fallback.
  Removing the E010 reference-ray decode/query slightly reduces work.
- Acceptance gate: preserve or improve E008's `6/10` fixed online result while
  reducing source-truth first-metre and forward-detour collision. E009/E010's
  `3/10` result rejects the version even when ADE, progress or curvature looks
  better. Training and measurements must remain bound to the E011 code commit.
- Pre-training verification: the complete CPU suite passes (`114 passed, 4
  skipped`; two visualization skips lack Matplotlib and two are CUDA-only).
  The production V100 runtime passes compiled FP16 perception, conditioning,
  MeanFlow primal/JVP, backward, optimizer step and deployment sampling in
  `178.74 s`. The graph remains `33,898,916` parameters with `29,018,980` in
  the decoder; E011 adds no parameter and removes one pre-proposal curve/C-space
  evaluation.
- Current training: the exact E011 run uses V100S GPUs 0 and 3, global batch
  `1792`, two `448`-sample microbatches per rank and FP16. At step `1820/4600`
  it sustains approximately `1393 samples/s` with both GPUs compute-saturated
  and about `27.9 GiB` allocated per device. This is approximately
  `696.5 samples/s/GPU`, slightly above the prior four-V100 E010/E011 topology's
  `692--705 samples/s/GPU`; the lower aggregate rate is entirely the two-card
  allocation, not duplicated model or data work.
- Throughput audit: packed depth stays resident on GPU, transfer is prefetched,
  condition K/V is projected once per microbatch, DDP uses `no_sync` for the
  first accumulation, AdamW/EMA are fused/foreach, and perception, conditioning,
  primal and JVP graphs are already compiled. On an idle RTX 4090, the eager
  FP32 MeanFlow JVP took about `150--155 ms` for the audit batch while the
  production compiled JVP took `23.8--25.3 ms`. BF16 JVP (`166--181 ms` eager),
  explicit attention and `max-autotune-no-cudagraphs` did not improve this path;
  the latter spent minutes enumerating infeasible Triton kernels. The metric
  depth projector costs only about `5 ms` in the same batch, so caching it would
  add a second data contract for a small upper bound. These variants are
  rejected: no code branch or custom kernel is retained. A material aggregate
  increase therefore requires more identical GPUs; it is not available from a
  mathematically equivalent local rewrite identified by this audit.
- Completed source-truth offline evaluation: ADE/FDE are `0.03713/0.09489 m`,
  full-path collision is `9.664%`, and first `0.5/1.0 m` collision is
  `0.000/0.346%`. Forward-direct collision is `1.516%`, but forward-detour,
  rear-goal and expert-moves-away strata remain `22.210/62.857/67.052%`.
  Approximately `69.97%` of colliding trajectories have no four-frame raw-depth
  evidence at their collision points. E011 therefore improves E010's
  `12.187%` full and `0.495%` first-metre collision, but it does not establish
  closed-loop safety.
- Vectorized online diagnostic: the fixed Home scene, official episodes `0--9`,
  seed `1234` and `num-envs=10` complete at `1/10` SR and `0.096125` mean SPL.
  Episode 6 is the sole success (`0.961246` SPL); the other nine fail. Scene
  startup plus execution takes about `16.2 min`, while the post-ready ten-episode
  workload takes about `8.1 min` (`1.23 episode/min`). There is no model, HTTP,
  Isaac or evaluator exception. This B10 result is a throughput diagnostic and
  must not be compared numerically with E008/E009/E010's B1 acceptance scores;
  nevertheless it rejects any claim that the improved offline prefix metric by
  itself predicts robust closed-loop navigation.
- Frozen-map replay closes the controller ambiguity. Across `11,043` E011 B10
  replans, `99.62%` of complete plans and `97.64%` of first-metre prefixes enter
  the benchmark robot-centre non-navigable raster; even among plans whose
  origin is free, `82.02%` of first-metre prefixes collide. Actual positions
  are free for only `19.65%` of sampled steps and the episode-mean stalled-step
  fraction is `73.77%`, while desired MPC speed remains `0.421 m/s`. Adjacent
  first-metre replans differ by only `3.59 mm`. E008 under the same frozen-map
  renderer had `69.56%` actual-position free fraction and `35.43%` stalled
  steps. E011 therefore generates stable unsafe intent; MPC is not the primary
  cause. The separately captured E011 B1 episode 0 also has `100%` full-plan
  collision and only `3.27%` actual-position free samples, so vector batching
  is not the source of the failure.
- Causal correction from the exact published bundles supersedes the earlier
  repository-commit audit. E008 and E010 have byte-identical `models/policy.py`;
  their decoder computation is also identical, with differences limited to
  comments, formatting and the contract string. The `6/10 -> 3/10` regression
  therefore cannot be attributed to goal-reference retrieval. E008 trained in
  BF16 with global batch `1024` for `8000` optimizer updates; E010 trained in
  FP16 with global batch `1792` for `4600` updates. The example budgets are
  close, but E010 made `42.5%` fewer parameter updates at the same learning
  rate. This optimizer-topology confound must be removed before another
  decoder mechanism is accepted.

## E012 — controlled E011 optimization audit

- Purpose: isolate E011's robot-origin global retrieval from the large-batch
  training change. The model, dataset, loss, deterministic source and one-call
  runtime are byte-for-byte E011; only the training contract returns to the
  evidence-backed E008 values.
- Contract: one RTX 4090, BF16/FP32 precision, global batch `1024`, three packed
  microbatches bounded by `342`, `40960` samples per epoch, `200` epochs and
  exactly `8000` successful optimizer updates. Learning rate remains `2e-4`;
  it is not rescaled because the original batch is restored.
- Active run: commit `fd76659` is running on the only free RTX 4090 (GPU2) in
  tmux `curvenav-e012`. After one-time graph compilation, steps `20--140`
  sustain `976--988 samples/s`; the device holds about `22.5 GiB` and reaches
  full compute utilization. The exact log is
  `/DataDisk2/hsb/curvenav-training-e011/train-e012.log`.
- Early paired evidence: archived E008 logs and E012 use the same RTX 4090,
  BF16, seed, data order, batch, schedule and update count. At steps
  `240/400/480/560`, E008 total loss is
  `0.6801/0.4811/0.4298/0.4061`, whereas E012 is
  `0.7534/0.5139/0.4876/0.4886`. The gap is principally MeanFlow imitation,
  not a missing clearance penalty. Repeating robot-origin geometry for all
  seven control tokens removes their distinct metric retrieval anchors and is
  the only remaining causal graph difference. This is preliminary convergence
  evidence; the run still completes before the architecture decision.
- The exact E012 checkpoints at steps `800/1600/2400` are retained. Their total
  losses are `0.41474/0.29187/0.17221`; the step-2400 split is MeanFlow
  `0.16962` plus visible-clearance `0.00259`. The active RTX 4090 run sustains
  about `986--1004 samples/s`. These snapshots will be evaluated with the same
  source-truth evaluator after the final checkpoint, so a lower training loss
  alone cannot accept robot-origin retrieval.
- Decision gate: compare the final EMA checkpoint with E011 using the same
  source-truth offline strata, then run the fixed ten episodes with
  `num-envs=1`. If E012 remains below E008, reject robot-origin retrieval and
  restore the goal-reference retrieval. If it recovers E008, the prior
  regression was optimization rather than architecture. No candidate set,
  safety projection, critic or extra loss is introduced during this audit.
- Completed source-truth comparison: steps `800/2400/8000` reach ADE
  `0.05554/0.03790/0.03124 m`, full-path collision
  `12.484/9.202/9.812%`, and first-metre collision
  `0.726/0.396/0.247%`. The final checkpoint restores E008's imitation
  accuracy (`0.03105 m`) while improving its full-path (`11.642%`),
  first-metre (`0.396%`), forward-detour (`26.572%`) and forward-direct
  (`2.152%`) collision rates to `9.812/0.247/23.456/1.174%`. It does not
  dominate E011: E011 remains slightly better on full-path and detour
  collision (`9.664/22.210%`), while E012 is better on first-metre and direct
  collision. Therefore the old `6/10 -> 1/10` evidence cannot be assigned to
  robot-origin retrieval without a matched online run; the training topology
  was a material confound.
- Checkpoint-selection lesson: step `2400` has the lowest full-path and detour
  collision, but the final checkpoint has substantially lower ADE and the
  lowest execution-prefix collision. Training loss or full-tail collision
  alone is not a valid checkpoint selector for receding-horizon navigation.
  At the final checkpoint, `68.24%` of colliding trajectories and all `15`
  first-metre collisions have no matching obstacle evidence in any of the four
  raw depth frames. More optimization of the same observation cannot directly
  correct those invisible cases.

## E013 — metric retrieval anchors with terminal-only goal intent

- Root correction: E012 repeats the robot-origin relative geometry for all
  seven control tokens. If memory token `k` is at `p_k`, its positional bias is
  `beta(p_k-0)` for every control index; token identity cannot restore the
  missing exact metric relation. E013 restores E008's distinct Greville
  reference anchors `R_i`, so the bias is `beta(p_k-R_i)` and each future
  segment retrieves the scene at its own physical location.
- PointGoal correction: E008/E010 encoded candidate control `C_i` relative to
  the corresponding straight-reference control `R_i`. That makes a safe
  lateral detour appear as an index-wise deviation from a straight template.
  E013 uses only the common terminal local goal `G=R_7`, encoding
  `[C_i/H,(G-C_i)/H,||G-C_i||/H]` for every control. The straight reference
  locates observation queries but supplies no intermediate output target.
- Scope: one decoder call, one instantaneous clean proposal, one proposal
  C-space query and one final B-spline remain unchanged. No parameter, block,
  candidate, critic, loss, projection, solver step, inference branch or data
  field is added. Relative to the accepted E008 graph, the sole learned
  semantics change is index-wise straight-template intent to terminal-only
  intent.
- Verification before CUDA: commit `41cb214` keeps exactly `33,898,916`
  parameters (`29,018,980` decoder), fourteen Flow coordinates and seven curve
  tokens. Focused config/checkpoint/policy/loader/precision regression is
  `65 passed, 1 skipped`; the skip is the production CUDA compile test reserved
  for GPU2 after E012. A direct counterfactual test changes all intermediate
  reference controls while fixing `G` and proves the terminal-goal feature is
  bitwise unchanged. Commit `0a5c0ca` records that contract.
- The complete CPU suite on the exact E013 release is `117 passed, 2 skipped`
  in `26.76 s` with CUDA hidden and eight pinned CPU cores. The two skips are
  exclusively the CUDA compile and CUDA prefetch tests; the former remains in
  the automatic GPU2 gate between E012 evaluation and E013 training.
- Paper/source basis: SanD explicitly separates generated B-splines from ESDF
  selection, while NavDP separates generation from critic selection. CurveNav
  deliberately has neither runtime selector, so trajectory-aligned observed
  C-space must enter its single generator. LoGoPlanner and DiffusionAnything
  independently support metric/trajectory-aligned geometry queries; none
  justify treating a straight goal ray as the desired intermediate path.
- Official-source audit: SanD's current condition encoder first self-attends
  depth/motion tokens and then injects one trajectory-endpoint token by cross
  attention. NavDP repeats its goal token in the generator, but explicitly
  zeros goal memory in the RGB-D critic that ranks safety. Both support a
  target-independent scene representation followed by goal intent; neither
  supports supervising intermediate controls against a straight goal ray.
- MeanFlow audit: the official iMF implementation returns average `u` and an
  auxiliary marginal velocity `v` from one network call, directly supervises
  both, and uses predicted `v` as the stopped JVP direction. E013 follows that
  contract; its only specialization is exposing `v` after the first half of
  the shared decoder so the resulting clean proposal locates the second
  half's geometry query. It does not use expert `e-x` as the JVP direction or
  perform a second inference call.
- Mixed-precision audit: Accelerate skips the underlying optimizer call when
  FP16 GradScaler detects overflow. The old loop correctly held LR and EMA but
  still advanced the reported step. The unique loop now recomputes the same
  sample batch and Flow source after the scaler is reduced, and counts the
  step only when parameters actually update. BF16 execution is unchanged.
- Acceptance: first finish and evaluate E012. E013 then trains from zero with
  the same BF16, global-1024, 8000-update contract. It must preserve E008's
  fixed `6/10` result and reduce source-truth forward-detour collision; average
  ADE alone cannot accept the model.
- Production training: the real RTX 4090 compiled forward, MeanFlow JVP,
  backward and deployment-sampling gate passes. The remote virtual environment
  had lost its declared `accelerate` dependency after E012; restoring the
  pinned `1.14.0` package repaired the sole import failure without changing
  code or adding a launcher path. E013 now runs on GPU2 with BF16, global batch
  `1024`, about `22.5 GiB` allocated and `975--996 samples/s` after compilation.
  At the matched step `800`, E013 total/MeanFlow/visible-clearance losses are
  `0.33864/0.32538/0.01326`, versus E012's
  `0.41474/0.39513/0.01961`. This is evidence of improved optimization only;
  the retained checkpoint still requires the same source-truth and online
  acceptance tests.
