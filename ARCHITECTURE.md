# CurveNav 二维局部导航架构

本文是当前代码的唯一模型合同。CurveNav 模型只负责 PointGoal 条件的二维局部轨迹生成；仓库另含离线 HSSD 数据生成，但不包含仿真评测、服务器、传输、MPC 或机器人动力学适配。代码不保留旧模型兼容、架构分支、实验开关或推理启发式补丁。

## 任务与张量合同

输入 `PolicyCondition`：

```text
depth       float [B,4,1,126,224]  3 帧过去观测 + 1 帧当前观测；1 表示 5 m
point_goal  float [B,2]            当前机器人坐标系 PointGoal (x,y)
observation_to_current float [B,F,4]  每帧到当前帧的 (x,y,sin Δyaw,cos Δyaw)，此合同固定 F=4
observation_valid bool [B,4]          逐帧有效位；最后一个当前帧必须有效
```

深度相机合同固定为 SanD 原始轨迹相机：`224×126, fx=fy=166.80851, cx=112, cy=63`；相机位于机器人平面原点正上方 `0.40 m`，无前移且光轴水平。HSSD 由生成器直接使用相同外参渲染。内外参同时用于视觉 token 的度量反投影和显式评价器，并进入 source cache、dataset 与 checkpoint 合同。

训练目标 `TrajectoryTarget`：

```text
control_points float [B,8,2]   当前机器人坐标系的三次 B-spline 控制点
reference_path float [B,64,2]  同一路径的 64 点等弧长训练参考
```

输出 `TrajectoryPrediction` 包含选中的 `[B,64,2]` 路径、解析 heading/curvature，以及十六条候选控制点、候选路径、总几何代价和 clearance/length/goal 分项。参考路径保留来源的真实局部 horizon：SanD 与 HSSD 都最多取未来 24 个 0.15 m 专家步，真正到达 PointGoal 时自然缩短；二者都不缩放到固定长度。

## 唯一模型图

```text
4×depth -> shared ResNet-18 stage-3 -> 8×12 spatial tokens/frame
                     + 2-D sinusoidal position
                     + metric planar backprojection and current-frame alignment
                                      |
                                      v
                   64 learned queries cross-attend 384 visual tokens
                                      |
PointGoal ----> direction + log-range -> 1 goal token
                                      |
                                      v
                       2-layer joint self-attention
                                      |
                +---------------------+--------------------+
                v                                          v
     conditional rectified flow             explicit geometry evaluator
      16 B-spline candidates             current depth + robot footprint
                +--------------------- argmin ----------------+
                                      v
                        64-point equal-arc path
```

### 1. 多帧深度与几何对齐

每帧深度由同一个单通道 ResNet-18 编码，取 stage-3 的 stride-16 特征，经 `1×1` 投影后自适应池化为 `8×12=96` 个空间 token。二维正弦位置编码保留行列位置；四个 learned frame-slot embeddings 标识观测槽位。该实现吸收 SanD 官方源码的共享 ResNet-18、stage-3 特征、固定空间 token 网格和二维位置编码；由于 CurveNav 按行驶距离而非固定时间间隔取帧，序列信息由显式 frame-slot embedding 表达。

`observation_frames=4` 表示三个过去观测加一个当前观测，不是“四个历史帧”。这与 SanD 的四帧深度配置对齐。NavDP/X-NavDP 的 `memory_size=8` 是 RGB-D memory 的实现选择；CurveNav 是纯深度输入，当前保留 SanD 的四帧合同，避免无依据地扩展历史长度。

四帧共 384 个视觉 token 过长，因此使用 64 个 learned queries 做一次 masked cross-attention：

```text
Z = Attn(Q, V + P2D + Pslot, V + P2D + Pslot),   |Q| = 4×16.
```

无效历史深度先被确定性置零，避免其数值经 ResNet BatchNorm 影响有效帧，再在 cross-attention 的 key/value 侧完全 mask。该压缩方式来自 NavDP 的 learned-query RGB-D memory compressor；查询可从全部历史中提取任务相关统计，而计算量从后续层的 `O(384²)` 降为 `O(64×384 + 64²)`。

每个 `8×12` token 还显式携带一个度量平面点，而不是把整帧变换广播成特征。令归一化深度恢复为光轴距离 `z`，光学坐标为

```text
x_o = z(u-c_x)/f_x,
y_o = z(v-c_y)/f_y.
```

对相机前移 `a`、下俯角 `α`，反投影到机器人平面的点是

```text
p_t(u,v) = (a + cos(α)z - sin(α)y_o, -x_o).
```

`observation_to_current=(t_x,t_y,sin Δθ,cos Δθ)` 按刚体变换映射到当前坐标：

```text
p_current = R(Δθ) p_t + (t_x,t_y).
```

平面点和 pooled depth 经过三维输入 MLP，与 SanD 的视觉 token 相加，再交给 NavDP 风格 query compressor。这样历史运动直接作用于每个空间 token，后续模块不再维护单独的 pose-token 分支。最后将一个 PointGoal token 和 64 个压缩视觉 token 拼接，经两层联合 self-attention 得到 65 个目标相关条件 token。

该投影使用合同中的完整 pinhole 内参、固定相机外参和二维相对运动，数学上是标准深度反投影、camera-to-body 刚体变换与逐帧平面对齐；它不是地面交点投影。`observation_to_current` 是数据变换的名字，不使用含糊的刚体状态缩写；frame-slot embedding 只编码序列槽位，不重复编码几何。

SanD 源码中还有上一条选中轨迹的 initial-turn token，但当前 prepared dataset 没有“因果上的上一条已选轨迹”。用当前专家轨迹首段构造会泄漏标签；训练永远给 null、推理再给历史值又会造成分布偏移。因此当前合同不采用它，待数据明确保存上一决策周期的已选轨迹后再整体引入和验证。

### 2. PointGoal 编码

令 `r=||g||`，目标特征为

```text
(g / max(r,ε), log(1 + min(r,25 m)) / log(26)).
```

零距离时方向显式为零。方向和距离分离避免原始坐标尺度压过角度信息，并保留 25 m 内的距离差异；25 m 与 X-NavDP PointGoal 合同一致。PointGoal 表达任务意图，不直接指定本次局部轨迹长度。

### 3. B-spline 轨迹

模型预测八个夹持三次 B-spline 控制点 `Q_i∈R²`，固定 `Q_0=0`：

```text
τ(u) = Σ_i B_i,3(u) Q_i.
```

三次 B-spline 为 `C²` 连续。由于 `B_i,3(u)≥0` 且 `Σ_i B_i,3(u)=1`，曲线位于控制点凸包，并满足控制点扰动不放大界

```text
max_u ||τ_new(u)-τ(u)|| ≤ max_i ||ΔQ_i||.
```

这直接吸收 SanD 的八控制点、局部支撑和平滑表示。相对 NavDP 的 24 个稠密动作增量，平滑性由表示给出，不依赖部署后处理。

代码先以 `4×64` 个参数点过采样，再按累计弧长反查 64 个等弧长点。训练目标、评价和部署使用相同采样语义；heading 与 curvature 也在近似均匀米制步长上计算。

### 4. Conditional Flow Matching

控制点按 4 m 坐标尺度归一化但不截断；原点不加噪且不计入 Flow 损失。对自由控制点使用标准直线 conditional flow matching：

```text
ε ~ N(0,I),  t ~ U(0,1)
x_t = (1-t)ε + t x_1
u_t = x_1 - ε
L_flow = E ||v_θ(x_t,t,C)-u_t||².
```

轨迹 Transformer 先在八个控制点上做双向 self-attention，再 cross-attend 65 个条件 token。推理从固定种子的十六个独立 Gaussian noise 出发，以八步 Heun 积分 `dx/dt=v_θ(x,t,C)`，确定性地产生十六条候选。候选数对齐 SanD 部署合同，并低于 NavDP 默认的 32 条；Heun 的预测-校正步在相同步数下比左端 Euler 更准确。它替代 SanD/NavDP 的 DDPM，但不是来自 X-NavDP 的 RL 后训练。

训练时的一步数据估计

```text
x_hat_1 = x_t + (1-t)v_θ(x_t,t,C)
```

直接经 B-spline 和等弧长解码接受路径与切向监督，使最终可执行路径的误差能传回 Flow。

### 5. 显式几何轨迹评价

候选选择采用 SanD 的解析代价结构，不训练额外 scorer。评价器只读取当前帧 metric depth、相机标定、机器人尺寸、PointGoal 和十六条同状态候选。

当前深度先按完整 pinhole 模型反投影。对像素 `(u,v)` 和光轴深度 `z`：

```text
x_optical = (u-cx) z / fx,
y_optical = (v-cy) z / fy,
p_body.x = a + cos(α)z - sin(α)y_optical,
p_body.y = -x_optical,
h_body = h_camera - sin(α)z - cos(α)y_optical.
```

只保留 `0.05 m <= h_body <= 0.70 m` 的可见表面，过滤地面和高于机器人本体的表面。候选等弧长点 `x_j` 到这些平面障碍点的最小欧氏距离记为 `d_j`；超出当前水平视场、位于相机后方或超过 5 m 观测距离的点没有几何证据，令 `d_j=0`。机器人当前已占据的原点邻域（半径取机器人半径与相机前移量的较大值）保留点云距离，避免把轨迹必经的当前本体区域错误标成未知。安全中心距为机器人半径与余量之和：

```text
d_safe = r_robot + m_safe = 0.25 m + 0.10 m = 0.35 m.
```

十六条候选分别计算：

```text
J_clear = Σ_j γ^j max(0, d_safe-d_j) / Σ_j γ^j,  γ=0.95
J_length = Σ_j ||x_{j+1}-x_j||₂
J_goal = ||x_last-g||₂
J = 10 J_clear + J_length + J_goal.
```

选择 `argmin J`。`J_length+J_goal >= ||g||₂` 来自三角不等式，因此两项同权时不会奖励原地停止，同时会惩罚相对目标直达距离的额外绕行；十倍 clearance 权重和近端折扣取自 SanD 默认解析 critic。评价器无可训练参数、无标签输入，也不进入训练损失。

这里的 `d_j` 是当前可见深度表面点集的距离近似，不是完整占据图的有符号 ESDF；代码和指标均使用 `surface clearance` 命名。未知视场按不安全处理使选择保持保守，但单帧遮挡后的自由空间不会被虚构出来。NavDP 的 learned critic 需要 privileged ESDF 正负轨迹，X-NavDP 的 GQRM/RTC 需要在线 reward/Q target；当前数据没有这些监督，因此不引入语义不成立的 learned safety/Q 分支。

## 总损失

唯一训练目标是

```text
L = 1.0 L_flow + 0.5 L_path + 0.1 L_tangent.
```

- `L_path`：`x_hat_1` 解码路径与等弧长 `reference_path` 的逐点 Smooth-L1。权重 `0.25+exp(-4s)` 归一化到均值 1，强调 receding-horizon 即将执行的近端。
- `L_tangent`：有效目标段上的 `1-cos(Δx_hat,Δx*)`，使用同一近端权重，约束初始转向和局部跟踪方向。
不对不同自由控制点设置手工 Flow 权重。B-spline 已保证连续性，路径和切向损失负责将执行几何质量传回 Flow。

## 参考设计吸收边界

| 参考 | 当前已吸收 | 当前未采用及原因 |
|---|---|---|
| SanD | shared depth spatial tokens、2-D/frame-slot 编码、8 点三次 B-spline、等弧长输出、当前深度显式几何评价与 clearance/length/goal 代价 | DDPM 被标准 CFM 替代；initial-turn 缺少因果训练字段；当前深度表面距离不冒充全局 ESDF |
| NavDP | learned-query 多帧压缩、同状态多候选生成后评价 | 输入合同没有 RGB；没有 privileged ESDF 正负标签，故不采用 learned safety critic |
| X-NavDP | PointGoal 的 25 m 编码边界 | GQRM、twin Q、RTC 和 embodiment FiLM 都依赖在线 RL、执行历史或多本体数据 |
| FlowNav | 标准 conditional flow matching 与少步 ODE 推理 | 不预设其 NFE 优势会自动转化为 CurveNav 闭环增益 |

因此当前模型是“SanD 轨迹表示、深度时空 token 与显式候选评价 + NavDP learned-query 压缩与同状态多候选接口 + 标准 CFM”，而不是把监督条件互不兼容的 learned critic 或在线 RL 模块机械叠加。

主要来源：[SanD-Planner](https://arxiv.org/abs/2602.00923)、[SanD 官方源码](https://github.com/WangJinCheng1998/sandplanner)、[NavDP](https://arxiv.org/abs/2505.08712)、[NavDP 官方源码](https://github.com/InternRobotics/NavDP)、[X-NavDP](https://arxiv.org/abs/2607.28560)、[FlowNav](https://arxiv.org/abs/2411.09524)、[Implicit Behavioral Cloning](https://arxiv.org/abs/2109.00137)。显式几何对照：[Semantic MapNet](https://arxiv.org/abs/2010.01191)、[PETRv2](https://arxiv.org/abs/2206.01256)、[BEVDet4D](https://arxiv.org/abs/2203.17054)、[LoGoPlanner](https://arxiv.org/abs/2512.19629)。

## 训练、推理与验证

### 数据合同

唯一数据入口是 `scripts/build_dataset.sh DATA_ROOT`。它依次下载固定 commit 的 SanD 与 HSSD、生成 HSSD 专家 route、准备两类 route 深度缓存，并原子编译 `data/policy_dataset`；训练与 GPU 调度不属于数据生成链。各阶段只接受空输出，不提供历史版本、恢复模式或已有输出分支。HSSD 生成前从冻结 repository index 推导 20 个场景引用的全部资产，逐项验证文件存在、JSON 可解析及 GLB 头和声明长度正确；缺失物体不能以 Habitat 警告形式静默进入深度数据。20 个冻结场景按 16/4 划分 train/validation，并禁止同源 scene family 跨 split。每个场景生成 25 条无扰动的完整专家 route：近距 5 条、中距 10 条、远距 10 条，共 500 条，train/validation 分别为 400/100 条。

每条 HSSD route 沿 clearance-aware 路径按 0.15 m 等弧长采样，保存连续平面位姿以及逐位置 `224×126` metric depth，最终位置就是该 route 的任务 PointGoal。HSSD 相机内外参必须与上述当前深度合同完全一致，否则编译立即拒绝。编译阶段在每个非终点位置切出一个监督样本：历史四帧按 `[-1.35,-0.90,-0.45,0] m` 索引，未来最多 24 步作为局部路径，并计算 `observation_to_current=(x,y,sin Δyaw,cos Δyaw)`。同一 route 的深度只保存一次，局部样本通过索引共享。生成门禁验证数量、连续 clearance、0.15 m 间距、距离分布、深度/位姿对齐、split 无泄漏、原子提交和最终 SHA。

SanD 轨迹文件给出的相机高度恒为 `0.40 m`、pitch 恒为 0；缓存阶段验证该外参，并只把原始 `640×480` 内参重投影到 canonical `224×126`。HSSD 从根源使用同一个 `0.40 m` 水平相机渲染。两类 cache manifest 与最终 dataset manifest 都记录并严格校验同一标定，训练期 loader 因而只读取统一张量，不保留来源分支。

唯一链路：

```text
HSSD generation + SanD source -> calibrated prepared dataset
                              -> fixed-batch overfit diagnostic -> DDP mixed-precision training -> EMA
                 -> offline geometry metrics -> official closed-loop benchmark
```

checkpoint 严格记录输入标定、编码器与生成器类型、平面反投影对齐、每帧 16 个压缩 token、八控制点、等弧长倍数、损失语义和显式评价器参数；合同不一致时直接拒绝加载，不设置兼容分支。推理不读取标签，也不执行曲率裁剪、直线候选或轨迹反转。

开始完整训练前只保留三类验证：

1. 数学与接口单测：B-spline、等弧长、逐帧 mask、平面反投影、learned-query token 数、显式几何代价和 checkpoint；
2. 前向/反向：所有输出形状正确，全部可训练参数具有有限梯度；
3. 固定批过拟合及同协议离线/闭环评测：selected/oracle ADE、几何代价 margin、selected surface clearance、clearance violation、候选多样性、延迟、SR/SPL。

离线 ADE 不能替代闭环 SR/SPL；可见表面 clearance 不能解释为完整 ESDF 或碰撞概率；未进行 X-NavDP 在线 RL 时不得使用“Q 后训练”表述。
