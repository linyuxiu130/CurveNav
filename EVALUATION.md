# CurveNav evaluation contract

## Physical safety truth

所有 CurveNav 离线安全分数只来自准备数据时复制的 native `navigation_grid.npz`。每条轨迹在机器人局部坐标中以 `0.025 m` 稠密重采样，按保存的 world anchor 和 yaw 映射回原始 world grid；任一 source-grid 越界点都记作不可执行且 signed clearance 为负。

因此以下量是唯一可用于 checkpoint 选择和在线结论的物理安全指标：

- `footprint_collision_fraction`
- `safety_margin_violation_fraction`
- `min_clearance_m`
- `path_field_coverage_fraction`
- 以及它们相对 source-safe expert 的差值和按真实几何 strata 的分解。

评测在开始前会重新查询所有 reference expert。任一 expert 碰撞、净空低于 `0.10 m` 或越界都会直接失败，而不是继续生成分数。

## raw depth C-space 的正确用途

策略输入的 `64×64` raw C-space 只由四帧深度构造，网格间距约 `0.114 m`。它是局部观测，不能替代 native `0.05 m` source grid。未观测格即使在有限距离变换中有数值，也只以零几何、显式零 coverage 进入网络；不会被解释成自由空间。

离线报告仍会把 source-truth 碰撞点与 raw field 的同一点查询对齐，输出：

- source 碰撞点是否被 raw depth 覆盖；
- 覆盖点是否被 raw signed clearance 判为碰撞；
- raw depth 的 false-collision 点数。

这些是感知可观测性/架构诊断，不是物理安全分数，也不参与 checkpoint 排名。没有 learned completion IoU、completion 碰撞率、`local_clearance_m` 或 `p−` 反事实指标。

## 固定验证协议

每个 validation observation 只生成一条确定性的单步 MeanFlow B-spline。输出包括：

- 固定 `2 m` 比较域的 ADE/FDE、弧长、PointGoal progress 和 regret；
- 曲率、曲率变化、总 heading change、切线反向；
- source C-space 碰撞/净空；
- 真实 held-out 数据中的 `forward_direct`、`forward_detour`、`rear_goal` 和 `expert_moves_away_from_goal` strata；
- 只保留当前帧的历史消融，用于测量四帧时序证据的贡献；
- batch-32 吞吐和 batch-1 延迟。

`forward_detour` 由 source-safe expert 与 straight chord 的 source C-space 关系定义，不使用人工合成目标或阈值。目标在身后和专家暂时远离最终 PointGoal 都是原始 held-out route 的自然样本，而不是故意制造的异常测试。

## 运行

```bash
scripts/run_training_runtime.sh \
  ../.venvs/curvenav/bin/python -m curvenav.evaluation.offline \
  configs/base.yaml outputs/train_policy/checkpoint.pt \
  --artifact-dir outputs/offline-evaluation
```

评测加载 checkpoint 的 EMA 权重，并在运行前严格验证 policy contract。artifact 只写入：

- `offline-metrics.json`：唯一的数值报告；
- `offline-cases.json`：各真实 stratum 的代表性路径和最差 source collision，含 raw C-space、预测/专家轨迹和 PointGoal。

## 在线闭环解释

闭环 SR、SPL、success threshold、timeout 和控制器/MPC 均不在本文件或模型中改动。在线 trace 必须把每个局部 plan 用冻结 benchmark C-space 重投影，记录 plan point-free fraction、整条 plan-free、首 `1 m` free、OOB 和实际位姿 free；这样能区分“生成路径已经进入不可执行区”和“安全计划但执行层失配”。这些 trace 是失败归因，不会更改固定 benchmark 成绩。
