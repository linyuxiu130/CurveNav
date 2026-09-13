# CurveNav evaluation contract

测评吞吐、缓存和常驻场景的唯一维护文档是
[`EVALUATION_ACCELERATION.md`](EVALUATION_ACCELERATION.md)。本文件只定义
安全真值、指标和固定协议。

## Physical safety truth

所有 CurveNav 离线安全分数只来自准备数据时复制的 native `navigation_grid.npz`。每条轨迹在机器人局部坐标中以最大 `0.025 m` 间隔稠密重采样，短轨迹的真实终点也会显式查询；随后按保存的 world anchor 和 yaw 映射回原始 world grid。任一 source-grid 越界点都记作不可执行且 signed clearance 为负。

因此以下量是唯一可用于 checkpoint 选择和在线结论的物理安全指标：

- `footprint_collision_fraction`
- `safety_margin_violation_fraction`
- `min_clearance_m`
- `path_field_coverage_fraction`
- `execution_prefix_0p5m_collision_fraction`
- `execution_prefix_1p0m_collision_fraction`
- `distance_to_first_collision_m`
- 以及它们相对 source-safe expert 的差值和按真实几何 strata 的分解。

评测在开始前会重新查询所有 reference expert。任一 expert 碰撞、净空低于 `0.10 m` 或越界都会直接失败，而不是继续生成分数。

## raw depth C-space 的正确用途

策略和评估器用同一确定性投影从四帧深度与因果局部障碍记忆构造 `64×64` observed C-space，网格间距约 `0.114 m`。它是策略的目标无关输入表示，也是判断 source-truth 碰撞是否被传感器观测到的诊断场；它不是物理真值、训练目标或 learned completion，不能替代 native `0.05 m` source grid。连续路径位置只有所有非零插值权重的支撑格都 observed 时才可报告净空下界；部分 observed stencil 不作为安全证据。

离线报告仍会把 source-truth 碰撞点与 raw field 的同一点查询对齐，输出：

- source 碰撞点是否被当前帧直接识别、仅被历史帧补充识别，或四帧均未识别；
- 每条碰撞轨迹的第一个 source-truth 碰撞点严格且互斥地归入当前帧可见、仅历史可见或四帧未识别；这是判断首次阻塞根因的主口径；
- source 碰撞点是否在 raw depth 严格四支撑覆盖内，以及覆盖后是否被 raw signed clearance 判为碰撞；
- raw depth 的 false-collision 点数。

这些是传感器可观测性诊断，不是物理安全分数，也不参与 checkpoint 排名。没有 learned completion IoU、completion 碰撞率、训练净空损失或 `p−` 反事实指标。

## 固定验证协议

每个 validation observation 只生成一条确定性的两步 Flow Matching B-spline。输出包括：

- 固定 `2 m` 比较域的 ADE/FDE、弧长、PointGoal progress 和 regret；
- 曲率、曲率变化、总 heading change、切线反向；
- source C-space 碰撞/净空；
- 同一次 source 查询得到的前 `0.5 m`、前 `1.0 m` 碰撞率和首次碰撞距离；前
  `1.0 m` 的首次碰撞进一步严格分为当前帧可见、仅历史帧可见和四帧均未识别，
  避免把长轨迹末端的可见碰撞误判成下一次重规划前会执行的近端碰撞；
- 轨迹物理终点、末端 `0.25 m` 是否碰撞，以及碰撞是否完全局限于末端 `0.25 m`；
- 复现项目 MPC 曲率计算与限速律：按轨迹长度限速后的速度乘以 3 s 得到米制前视范围（含下一顶点），报告最大曲率、期望速度和曲率限速比例；
- 真实 held-out 数据中的 `forward_direct`、`forward_detour`、`rear_goal` 和 `expert_moves_away_from_goal` strata；
- 只保留当前帧并清空障碍记忆的推理干预；同时移除了体素离散化，不能视为单独记忆模块的重训练消融；
- 批内半周期配对的真实 depth-condition swap 与 PointGoal swap，只报告轨迹变化及原场景前缀碰撞变化，用于判断深度因果依赖和目标捷径；
- base-policy batch-32 吞吐、包含全部诊断的端到端评估吞吐和 batch-1 延迟。

前缀指标比整条 `3.6 m` 路径更接近 receding-horizon 执行：机器人会先执行局部路径前段，再用新观测重规划。MPC 指标不是另一个可学习评价头，也不调用 Acados；它只复现固定控制器在求解前已经执行的确定性曲率/速度计算，因此计算量相对模型推理可忽略。

condition swap 只在真实 held-out 条件之间做确定性配对，不生成目标，也不改变主验证集。交换 depth 时将四帧、内外参、相对位姿、年龄、有效掩码和障碍记忆作为一个完整条件一起交换；交换 PointGoal 时保持原深度不变。它们是因果敏感性审计，不是物理可达任务，因此不计入 checkpoint 的导航成绩或 strata。

`forward_detour` 由 source-safe expert 与 straight chord 的 source C-space 关系定义，不使用人工合成目标或阈值。目标在身后和专家暂时远离最终 PointGoal 都是原始 held-out route 的自然样本，而不是故意制造的异常测试。

## 运行

```bash
CUDA_VISIBLE_DEVICES=1 scripts/evaluate_policy.sh \
  data/policy_dataset-depth-memory/config.yaml outputs/train_policy-depth-memory/checkpoint.pt \
  --artifact-dir outputs/offline-evaluation
```

评测加载 checkpoint 的 EMA 权重，并在运行前严格验证 policy contract。artifact 只写入：

- `offline-metrics.json`：唯一的数值报告；
- `offline-cases.json`：各真实 stratum、当前可见碰撞、仅历史可见碰撞、四帧未识别碰撞、末端碰撞和最早执行前缀碰撞的紧凑数值案例；
- `offline-cases.svg`：上述案例的简洁 source C-space BEV。浅橙/浅红只表示 source margin/collision；青色和紫色分别表示当前帧与历史新增的 raw C-space 障碍，射线覆盖仅保留极淡背景。专家、四帧预测、当前帧预测固定为绿/蓝/橙三条线，当前可见、仅历史可见、四帧未识别的 source-truth 碰撞固定为红圆、紫菱形、黑叉；每个 panel 只显示 ADE、最小净空、首次碰撞距离和 C/H/U 计数。

## 在线闭环解释

测评采集端读取实际 K、光学相机位姿和 10 Hz 时间戳，按传感器时钟生成四帧 深度 快照。`navigator_reset` 返回 checkpoint 对应的观测配置，推理服务不再维护按请求次数累积的历史。原生相机、半整数像素中心、深度 缩放与训练共用契约；生成数据必须重渲染到测评视场，不能靠替换旧图像的内参实现统一。

训练与测评统一 BF16 神经网络计算，几何和 Flow 数学保持 FP32。真实训练集拟合出的归一化统计保存在准备目录的 `config.yaml` 中，加载 checkpoint 时必须使用对应配置。

观测场连续净空使用 1-Lipschitz 下界 `max(lower(node)-distance(query,node))`。覆盖不足仍是未知；它是观测诊断，source-grid 和官方 SR/SPL 定义保持不变。

固定 Dingo evaluator 的实际终止项只有 `arrive_goal` 和 `time_out`。它不使用 YAML 中未接入环境的 `arrival_threshold: 1.0`；真正的固定到达条件是距离 `<0.5 m`、速度 `<0.25 m/s`，之后到达计时累计 40 个 Dingo 控制步。官方 success 为 `1-time_out`，SPL 为 `success*d0/max(path_length,d0)`；trace 必须独立重算并与 `metric.csv` 逐回合一致。上游到达计时器启动后不会在暂时离开阈值区时清零，这只可能使 success 偏高，不可能把真成功变成超时；为保持与 NavDP/X-NavDP 固定协议可比，CurveNav 不单独改动这一上游语义。

常驻 `scene_evaluator` 会在原上游 `evaluate_pointgoal.py` 之前直接导入
Isaac Sim，因此不能依赖后者设置 EULA 环境。主机所有者接受 NVIDIA
Omniverse EULA 后，唯一 launcher 环境必须显式包含
`OMNI_KIT_ACCEPT_EULA=YES`；否则非交互启动会在场景构建前因读取 stdin
得到 EOF。该项只允许写入主机 runtime 配置，不能硬编码到模型或用交互
fallback 绕过。

闭环 SR、SPL、success threshold 和 timeout 由固定测评协议定义；控制器改动见文末的本地前进控制协议。在线 trace 把每个局部 plan 用冻结 benchmark robot-center PLY 重投影。PLY 点按原生 `0.05 m` 格心构造 robot-center 栅格，禁止用点坐标作为像素边界而产生半格偏移。局部 plan 以 `0.025 m` 稠密查询；当前机器人原点与未来轨迹点分开计数，避免机器人一旦离开 proxy free map 后把所有后续 plan 自动判为碰撞。报告同时包含全体 plan 和“原点仍 free”条件下的前 `0.5/1.0 m` 未来碰撞率。

该 PLY 栅格是冻结 benchmark 几何的高效在线诊断 proxy，不是 Isaac contact 真值，也不替代官方 SR/SPL。每回合还报告实际位姿 free fraction，以及相邻真实重规划在 world frame 前 `1 m` 的路径不一致度。这样可区分：

- 第一个局部计划已经不安全；
- 后续重规划逐渐驶入不可执行区；
- 路径安全但曲率使 MPC 严重限速；
- 路径安全但连续规划抖动或执行层失配。

闭环图只裁切到该回合实际轨迹和 PointGoal 周围的物理区域，不再把整套资产缩进一张图。青色细线表示在线产生的局部计划，计划中落入冻结 non-navigable proxy 的稠密点叠为半透明红色；实际机器人位置落出 proxy free space 时使用实心深红点。场景总览同样按所有回合的起终点、轨迹和目标自动取景。颜色只帮助定位失败链路，不改变任何数值指标。

已完成的 E006 固定 10 回合是对这条合同的回归样例：官方结果为
`SR=5/10`、mean `SPL=0.450821`，trace 逐回合重算与 `metric.csv` 完全一致。
五个失败回合的 MPC 命令并非零，但实际速度显著低于成功回合；修正后的 proxy 诊断又显示
失败时当前机器人仍在 free map 的局部计划中，前 `0.5/1.0 m` 未来碰撞比例明显高于成功
回合。因此低成功率不能归因于 SPL 公式、PointGoal 轴、HTTP 返回点数或原点插入错误。
曾发现的半格 raster 偏移和“把当前原点算进每条 future plan”只污染旧诊断图，不影响
官方 success、timeout 或 SPL；对应单元测试固定格心、半开边界和 origin/future 分离语义。

离线评估仍是 held-out expert observation 上的 teacher-forced 测量，不能伪装成闭环 SR 预测器：它不包含策略导致的状态分布漂移、接触动力学或累计控制误差。评估器因此不构造随机目标、假闭环或复合“离线成功率”；最终导航结论仍由固定在线协议给出，离线负责更快、更准确地筛掉明显不安全或不可跟踪的 checkpoint。

当前共享 MPC 的底盘线速度域为 [-v_max, v_max]，与上游一样允许倒车；
专家生成的前进运动要求与跟踪器速度域是不同的契约。
急弯与近目标减速不设置额外最低速度，求解失败直接报错。
SQP 使用 MERIT_BACKTRACKING、500 次迭代上限和 0.01 Hessian 阻尼。
这些设置及参考轨迹处理保存在运行时维护补丁中，属于项目统一跟踪器，
不能标为未经修改的官方控制器复现。数值回放通过不保证所有闭环参考都收敛。

当前 MPC 使用转角/双边弧长曲率、连续弧长投影和米制前视，没有最低期望速度。离线 `mpc_max_curvature_lookahead_inv_m` 与在线 reset 做直接数值比对；此前保存的 `first12` 诊断属于旧公式，不能用于解释当前控制器。该更正不改变已保存的 ADE、碰撞率或在线 SR/SPL。
