# CurveNav architecture

This document is the exact contract of the only CurveNav model. Experimental
results and rejected designs are recorded in `EXPERIMENTS.md`.

CurveNav maps four calibrated depth frames, their relative SE(2) poses and a
robot-frame PointGoal to one 64-point planar cubic B-spline. Inference is one
deterministic improved-MeanFlow function call and one state update. The first
geometric retrieval uses fixed, target-independent metric horizon slots; the
PointGoal is only a terminal intent. There is no
candidate set, critic, safety head, map completion, trajectory projection,
online optimizer, fallback or legacy model branch. Training contains one
deployment-path feasibility term; inference contains no safety computation.

## 1. Evidence-backed root diagnosis

E005 reached 3.40 cm teacher-forced ADE but only 1/10 success in the fixed
closed loop. E006 moved all spatial queries from the straight goal reference to
the current Flow candidate. Under the identical ten-episode protocol it reached
5/10 success and mean SPL 0.451, so explicit configuration-space/path coupling
is useful and must be retained.

E006 nevertheless converged to 12.78% full-path source collision and failed five
online episodes. In failures, MPC commands remained nonzero and comparable to
successful episodes, while actual speed collapsed. Replans were highly stable,
not noisy, but repeatedly pointed into frozen non-executable geometry at corners
and blocked corridors. Official SR/SPL, PointGoal axes, trajectory-origin
insertion and MPC interfaces were independently reproduced; they do not explain
the failures.

E007 corrected a precise E006 information defect. At deployment its only decoder input is
the fixed Gaussian typical-set source `e*`. E006 decoded that source as a
physical curve and queried depth C-space along it, then jumped directly to the
answer. The generated curve was never the object of a geometry query. More
epochs cannot repair this information boundary, so E007 estimates a clean
endpoint before making the path-relative query.

The complete E007 step-2,400 audit rejects the stronger claim that this query
alone teaches safety. MeanFlow loss fell to 0.0425, yet source-truth collision
remained 12.27% and forward-detour collision 28.05%. The internal proposal was
less safe than the final path (13.39% versus 12.27%); proposal-safe to
final-collision occurred in only 0.79%, while the final field rescued 1.91%.
Therefore PointGoal's final update is not the primary destroyer. Both Flow
readouts are rewarded only for coordinate imitation, so low average error does
not require the deployed path to respond correctly to visible clearance.

E008 then completed the fixed ten-episode online run at `6/10` success and
`0.5828` mean SPL, versus `5/10` for the official NavDP checkpoint on the same
first ten episodes. Its four failed executions did not oscillate: adjacent
first-metre plans differed by only `0.9--5.1 mm`, MPC requested
`0.36--0.50 m/s`, and the robot repeatedly approached the same non-executable
geometry. The internal clean proposal and final output differed by only
`9.16 mm` offline. The remaining defect is therefore not a missing refinement
iteration.

E009 and E010 both reached `3/10`, but the published release bundles show that
this comparison cannot be used to explain E008's `6/10 -> 3/10` regression.
E008 and E010 have byte-identical `policy.py` and the same decoder computation;
their decoder files differ only in comments, formatting and the contract name.
The actual change was the optimizer topology: E008 used BF16, global batch 1024
and 8,000 updates, while E010 used FP16, global batch 1,792 and 4,600 updates.
They consumed nearly the same number of examples, but E010 performed 42.5%
fewer parameter updates at the same learning rate. Large batch is a throughput
choice, not an equivalent training transformation.

E011 changes only the first spatial retrieval from E008/E010's metric goal
reference to robot-origin global geometry. Its `1/10` vectorized diagnostic and
unsafe frozen-map replay reject the trained checkpoint. E012 restores E008's
1,024/8,000 BF16 training contract while keeping the E011 graph. The paired
training curves show slower early convergence, but the completed run restores
E008's imitation accuracy and improves its source-truth offline collision
metrics. Training topology was therefore a material confound, and the earlier
online regression cannot by itself reject robot-origin retrieval. The remaining
structural limitation is exact rather than empirical: with every control query
anchored at the origin, metric attention bias is identical across control
indices, so it does not directly encode the relation between each future curve
segment and each scene location.

E013 restores distinct metric reference anchors and removes a separate hidden
straight-path bias. E008/E010 computed the goal feature for candidate control
`C_i` from `R_i-C_i`, where `R_i` is the corresponding control on a straight
goal ray. That asks every intermediate control to compare itself with a
straight template even when the safe expert takes a detour. E013 instead uses
the same terminal local goal `G=R_7` for every control: `G-C_i`. The reference
ray locates geometry retrieval only; terminal intent no longer specifies an
intermediate path shape.

The current E014 single-step factorization is:

`depth history -> target-independent observed C-space visual BEV`

`Flow state + terminal-goal intent + fixed-horizon metric scene/C-space queries -> instantaneous clean-endpoint estimate`

`proposal trajectory -> observed C-space/BEV queries`

`proposal geometry + candidate-to-terminal-goal intent -> interval-average velocity`

`fixed source - average velocity -> one final B-spline`

`final B-spline x observed C-space -> training-only feasibility risk`.

The fixed metric horizon reference is a retrieval coordinate only. It is never
added to generated controls and is never executed as output. This is one improved-MeanFlow
field with two mathematically distinct readouts,
not a two-policy planner or a literal safety-first subpolicy. The instantaneous field is required to estimate the
data endpoint on the linear interpolant; the average field is required for
one-step MeanFlow transport. Safety-first means geometry-grounded generation: the
same final curve must imitate the expert while remaining outside the visible
footprint-inflated C-obstacle. It does not mean that a depth-only subnetwork
must guess one route before it knows the goal.

The train split contains 25,777 observations from 16 scenes. This limits scene
diversity and policy-induced recovery evidence, but it is not used to excuse an
architectural error: SanD demonstrates that expert-only generator training can
work when its complete inference system supplies explicit geometric selection.
CurveNav first requires its own single generated trajectory to read the correct
geometry. Data aggregation is a later closed-loop generalization question, not
a substitute for this root correction.

## 2. Tensor, frame and precision contract

- depth: `[B,4,1,126,224]`, calibrated pinhole depth normalized by 5 m;
- observation transforms: `[B,4,4]=(x,y,sin(yaw),cos(yaw))`, mapping each
  observation frame into the current robot frame;
- PointGoal: `[B,2]` in current `x-forward,y-left` metres;
- Flow state: `[B,14]`, standardized physical increments of seven planar
  B-spline controls;
- output: `[B,64,2]` metres in the current robot frame.

The benchmark supplies its actual camera matrix at reset. Deployment resamples
native depth into the canonical `126x224` training camera. Projection, SE(2),
configuration-space construction, B-spline algebra, Flow state, JVP and loss
arithmetic use FP32. Neural convolution, linear and attention kernels use BF16
on supported GPUs or FP16 with GradScaler on V100. Training, offline evaluation
and deployment use the same capability-selected precision route.

## 3. Four-frame calibrated perception

All four depth frames pass through one shared ResNet-18-style backbone in one
`B*4` batch and produce four `8x12` learned grids. Calibrated ray endpoints are
back-projected into the Dingo body frame and transformed into the current robot
frame before fusion. Thus an obstacle seen only in history remains spatially
available after leaving the current image.

The same depth projection builds one `64x64` observed configuration field over
`[-3.6,3.6]^2`. Body-height returns are inflated by the benchmark Dingo
footprint. Channels are signed clearance, its normalized planar gradient,
observed support and forbidden state. Learned geometry is canonicalized as

`(o*d, o*dx, o*dy, o, o*forbidden)`.

Unknown EDT extrapolation therefore cannot masquerade as observed free space.
Maximum-range depth is valid observed-free ray evidence but not a surface hit.
The native navigation grid is privileged truth used only to certify experts and
evaluate predictions; it never enters the policy.

## 4. Target-independent metric BEV and history

Learned frame tokens are bilinearly splatted at aligned metric positions into a
`16x16` robot-centric BEV. A convolutional encoder maps the observed `64x64`
C-space to the same grid. Their sum plus metric positional encoding forms 256
target-independent scene tokens. Three motion tokens encode valid historical
relative poses, for 259 tokens total.

PointGoal is absent from this memory, so changing the goal cannot rewrite the
obstacle representation. Each trajectory-control query attends to scene tokens
with explicit relative `x/y/z`, distance, observation age and token type.

## 5. Single-call metric-horizon geometry and proposal-grounded refinement

Let `g` be PointGoal, `H=3.6 m`, and

`xi=(1/15,1/5,2/5,3/5,4/5,14/15,1)`

be the seven non-origin Greville abscissae. The target-independent metric
retrieval controls are

`A_i=(xi_i H,0)`.

They form a fixed forward horizon used only to locate the first metric queries.
They never initialize or offset generated controls, constrain length or execute
a path. The terminal intent is independently defined as
`G=min(||g||,H)g/max(||g||,eps)`.

Let scene-memory token `k` have robot-frame position `p_k`. The first six
Transformer blocks preserve a distinct physical relation for control query `i`

`reference_ik=[(p_k.xy-A_i)/H,p_k.z/H,||p_k.xy-A_i||/H,surface_k,age_k,type_k]`.

The fixed reference path queries observed C-space, and the query token contains
Flow state `z_t`, interval endpoint `(t,t)`, control identity, reference-path
geometry and terminal intent

`intent_i=[A_i/H,(G-A_i)/H,||G-A_i||/H]`.

This does not declare the forward slots executable. It tells each control token
where to inspect the same target-independent scene at a distinct future
distance, while PointGoal cannot relocate that safety lookup or impose an
intermediate straight route. The planar readout predicts instantaneous velocity
`v_theta(z_t,t,c)`, which defines the learned clean proposal

`x_tilde_0 = z_t - t v_theta(z_t,t,c)`.

The codec immediately decodes `x_tilde_0` into seven physical controls and a
64-point B-spline. Every path point queries deployed observed C-space; the 64
features are aggregated to controls by normalized positive B-spline basis
weights. For proposal control `C_i`, the second six blocks receive

`path_i = aggregate_j[q_j/H, o*d/H, o*grad(d), o, o*forbidden]`,

`goal_i = [C_i/H,(G-C_i)/H,||G-C_i||/H]`,

plus BEV attention relative to `C_i`. Their readout predicts interval-average
velocity `u_theta(z_t,r,t,c)`.

This ordering keeps one reference-located scene/path evaluation and one
meaningful candidate-path evaluation without an inference loop. The reference
is not an output proposal; it supplies distinct metric horizon anchors that
E011/E012's origin-repeated query removes. Bilinear field lookup, affine coordinate
decode and B-spline evaluation are differentiable almost everywhere, so the
MeanFlow JVP includes

`(z_t,g,c_reference) -> v_theta -> x_tilde_0 -> B-spline -> observed geometry -> u_theta`.

## 6. Physical B-spline coordinates

The first control is fixed at `P_0=(0,0)`. Seven generated planar increments
obey

`Delta_i=P_i-P_(i-1)`, `P_i=sum_(j<=i) Delta_j`.

Each of the fourteen physical components is standardized using the source-safe
training split:

`e_k=(Delta_k-mu_k)/sigma_k`.

This is a nonsingular diagonal affine Euclidean coordinate change, so Gaussian
interpolation and MeanFlow velocity remain mathematically valid. Decoding
depends only on generated coordinates; PointGoal never leaks through an
analytic output reference.

The final path is one fixed clamped cubic B-spline

`p(u)=sum_(i=0)^7 N_i(u)P_i`, `u in [0,1]`.

It is origin anchored and `C2` continuous without curvature clipping, length
clipping or a target-distance constraint.

## 7. One-step improved MeanFlow

For expert coordinates `x`, Gaussian source `e`, and end time `t`,

`z_t=(1-t)x+t e`, `v*=e-x`.

The instantaneous readout is structurally independent of interval start `r`.
The average readout predicts `u_theta(z_t,r,t,c)`. With stopped FP32 material
derivative, improved MeanFlow trains

`v_theta -> v*`,

`u_theta + (t-r) stopgrad(D_t u_theta) -> v*`,

and `L_MF` is the equal mean of those two standardized Euclidean squared
errors.

Here `D_t u_theta` is the JVP in direction `(v_theta,0,1)`. This is iMF's
learned marginal-velocity direction, emitted by the auxiliary instantaneous
readout in the same decoder call and supervised by `v*`; it is not the
sample-specific expert velocity injected into the prediction function. The
first half of the shared decoder exposes that readout early only so its clean
estimate can locate the second half's trajectory-aligned geometry query.

For the exact deployment quarter only, let
`x_hat=e*-u_theta(e*,0,1,c)` and densely sample its decoded path at 2.5 cm. At
query `q_j`, `d_j` is raw signed clearance and `s_j` is true only when every
bilinear support cell is observed. The dimensionless feasibility risk is

`L_vis = mean_j 1[s_j] [relu((0.10-d_j)/0.10)]^2`.

The denominator is the fixed 3.6 m query-grid size, not observed count or path
length. Before querying, the curve is identity-normalized with detached arc
length: its forward value is unchanged, while the risk has zero derivative in
the uniform radial-scale direction. Thus `L_vis` changes turning shape rather
than teaching uniform shortening. Unknown EDT values never contribute. The
unique objective is `L=L_MF+L_vis`, with both terms nondimensionalized and unit
weight; there is no critic, ranking target, privileged map, reconstruction loss
or inference-time penalty.

Exactly one quarter of each global batch uses `(r,t)=(0,1)` and the same fixed
typical-set source used by deployment. One quarter samples positive-width
interior intervals and one half uses diagonal intervals. The instantaneous
field provides the stopped JVP tangent and is directly supervised by `v*`.

Inference performs exactly one decoder call:

`x_hat = e* - u_theta(e*,0,1,c)`.

`x_hat` is decoded once to the sole 64-point trajectory. There is no Euler/ODE
loop and no candidate selection.

Offline evaluation may also decode the already-computed internal
`x_tilde_0=e*-v_theta` to measure proposal/final alignment. This does not add a
deployment forward pass, candidate, score or selector: runtime still returns
only `x_hat`. A high `proposal-safe -> final-collision` rate would falsify the
assumption that proposal-grounded geometry reaches the deployed average field.

## 8. What is and is not guaranteed

Every serialized expert is re-certified against native source C-space at
2.5 cm spacing, with at least 0.10 m clearance and OOB non-executable. Expert
imitation therefore supplies a physically safe target. The learned clean
proposal gives the average field output-relative geometry; the final-path risk
adds the missing direct derivative from visible footprint clearance to the
deployed controls. Expert regression retains route and progress supervision,
while the risk has no uniform-length gradient. Neither term is a second policy.

This remains learned constrained imitation, not a hard collision guarantee.
Local depth cannot certify unseen topology, and a single deterministic generator does not
inherit the candidate selector of SanD or critic of NavDP. Evaluation therefore
separates the first `0.5/1.0 m`, the full path and the visibility of the first
source-truth collision. First-hit current-frame, history-only and four-frame
unrecognized masks form an exact partition; an ``any point visible`` label is
reported only as perception support and must not be interpreted as the cause of
the first collision. The stated engineering target is below 1% physical
collision, but it can only be claimed from source truth and fixed closed-loop
measurements, never from the raw `64x64` depth proxy alone.

## 9. Training and efficiency contract

- one prepared dataset, loader, policy, combined generator objective, launcher and
  checkpoint schema;
- global batch 1024, 40 updates per epoch, 200 epochs / 8000 updates;
- deterministic zero-dropout training/inference;
- compiled perception, conditioning, primal and stopped-JVP graphs;
- twelve decoder blocks: six reference-geometry-to-proposal plus six
  proposal-to-average blocks in one call;
- DDP preserves the exact global batch and exact deployment quarter;
- an FP16 overflow recomputes the same global sample batch and Flow source at
  the reduced loss scale; step, scheduler and EMA advance only after the
  optimizer commits the update;
- step-800 source-truth gate before a full run;
- RTX-4090 steady-state architecture gate at least 3000 samples/s.

The single-call block count matches E006's inference cost order. The instantiated
policy contains `33,898,916` parameters, including `29,018,980` in the decoder.
Latency and steady training throughput are reported only after the production
CUDA graph reaches a stable measured interval.

## 10. Relation to public systems

- [SanD-Planner](https://github.com/WangJinCheng1998/sandplanner): CurveNav keeps
  short depth history, a scratch visual encoder, low-dimensional cubic curve
  and expert imitation. SanD's complete inference samples trajectories and
  scores them with ESDF; CurveNav deliberately has neither inference mechanism
  and instead differentiates observed clearance through its sole output during
  training.
- [NavDP](https://github.com/InternRobotics/NavDP): its noisy trajectory tokens
  attend to visual context and iterative diffusion repeatedly updates them.
  CurveNav keeps the Flow state as the neural transport input but does not
  pretend its one-step Gaussian source is already a navigation proposal. It
  uses fixed metric horizon slots to retrieve the complete robot-centric scene,
  keeps terminal PointGoal intent separate, estimates one clean endpoint, and grounds that endpoint as a
  candidate path inside the same call.
  NavDP's critic and collision
  augmentation are not hidden in CurveNav. Their necessity is evidence that
  coordinate imitation alone is not a physical feasibility objective.
- [X-NavDP](https://arxiv.org/abs/2607.28560): its reinforcement post-training is
  outside the present supervised generator.
- [FlowPilot](https://arxiv.org/abs/2608.00635): its world/action coupling
  supports the principle that predicted future geometry should inform action,
  but CurveNav does not add a future-depth decoder or multi-step Euler solver.
- [MeanFlow](https://arxiv.org/abs/2505.13447): CurveNav preserves direct
  average-velocity transport and exact `(0,1)` deployment-boundary coverage.
  The instantaneous proposal is the improved-MeanFlow endpoint estimate used
  to place the geometry query inside the same function call.
- [LoGoPlanner](https://arxiv.org/abs/2512.19629): its task-specific state and
  geometry queries establish metric scene context before goal-conditioned
  diffusion. CurveNav adopts only that information-factorization principle:
  known SE(2) history and calibrated depth already provide more direct metric
  geometry than learned pose/point-cloud heads. It does not adopt VGGT, implicit
  localization, ten-step diffusion, sixteen candidates or the released critic.

The architectural contribution is a one-call, deployment-grounded improved
MeanFlow: target-independent four-frame observed C-space BEV, physical
Flow coordinates, metric-horizon geometry attention, terminal-only
PointGoal intent, a supervised instantaneous
endpoint estimate, a differentiable C-space query along that estimate,
query-relative PointGoal intent, one final physical B-spline transport and one
training-only feasibility objective on that exact output.
