# CurveNav 二维局部导航架构

本文是当前代码的唯一模型合同。CurveNav 模型只负责 PointGoal 条件的二维局部轨迹生成；仓库另含离线 HSSD 数据生成，但不包含仿真评测、服务器、传输、MPC 或机器人动力学适配。代码不保留旧模型兼容、架构分支、实验开关或推理启发式补丁。

## 任务与张量合同

输入 `PolicyCondition`：

```text
depth       float [B,4,1,126,224]  3 帧过去观测 + 1 帧当前观测；1 表示 5 m
point_goal  float [B,2]            当前机器人坐标系 PointGoal (x,y)
observation_to_current float [B,4,4]  每帧到当前帧的 (x,y,sin Δyaw,cos Δyaw)
observation_valid bool [B,4]          逐帧有效位；最后一个当前帧必须有效
```

训练目标 `TrajectoryTarget`：

```text
control_points float [B,8,2]   当前机器人坐标系的三次 B-spline 控制点
reference_path float [B,64,2]  同一路径的 64 点等弧长训练参考
```

输出 `TrajectoryPrediction` 包含选中的 `[B,64,2]` 路径、解析 heading/curvature，以及八条候选控制点、候选路径和组内对数概率。参考路径保留来源的真实局部 horizon：SanD 与 HSSD 都最多取未来 24 个 0.15 m 专家步，真正到达 PointGoal 时自然缩短；二者都不缩放到固定长度。

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
     conditional rectified flow                  trajectory scorer
       8 B-spline candidates                    same-context log-softmax
                +--------------------- argmax ----------------+
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

每个 `8×12` token 还显式携带一个度量平面点，而不是把整帧变换广播成特征。令归一化深度恢复为光轴距离 `z`，像素射线为 `x/z=(u-c_x)/f_x`，则相机水平平面点定义为

```text
p_t(u,v) = (z, -z (u-c_x)/f_x).
```

`observation_to_current=(t_x,t_y,sin Δθ,cos Δθ)` 按刚体变换映射到当前坐标：

```text
p_current = R(Δθ) p_t + (t_x,t_y).
```

平面点和 pooled depth 经过三维输入 MLP，与 SanD 的视觉 token 相加，再交给 NavDP 风格 query compressor。这样历史运动直接作用于每个空间 token，后续模块不再维护单独的 pose-token 分支。最后将一个 PointGoal token 和 64 个压缩视觉 token 拼接，经两层联合 self-attention 得到 65 个目标相关条件 token。

该投影只使用数据合同中存在的 pinhole 内参和二维相对运动，数学上是标准的深度反投影加平面刚体变换；它不是完整 BEV/地面交点投影，也不声称恢复相机高度或俯仰。`observation_to_current` 是数据变换的名字，不使用含糊的刚体状态缩写；frame-slot embedding 只编码序列槽位，不重复编码几何。

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

轨迹 Transformer 先在八个控制点上做双向 self-attention，再 cross-attend 65 个条件 token。推理从固定种子的八个独立 Gaussian noise 出发，以八步 Heun 积分 `dx/dt=v_θ(x,t,C)`，确定性地产生八条候选。Heun 的预测-校正步在相同步数下比左端 Euler 更准确；它替代 SanD/NavDP 的 DDPM，但不是来自 X-NavDP 的 RL 后训练。

训练时的一步数据估计

```text
x_hat_1 = x_t + (1-t)v_θ(x_t,t,C)
```

直接经 B-spline 和等弧长解码接受路径与切向监督，使最终可执行路径的误差能传回 Flow。

### 5. 轨迹组质量评分

当前数据没有 NavDP 的全局 ESDF 正负轨迹标签，也没有 X-NavDP 的在线 reward/Q target。把专家 ADE 称为 safety value 或 Q 在数学上不成立，因此 `TrajectoryScorer` 学习的是观测条件下的组内质量分布，而不是碰撞概率或 Q 值。每个 batch 组由本样本专家和其它样本的经验边缘轨迹组成；每个候选都由其等弧长路径对参考路径的 Smooth-L1 误差得到软目标：

对 batch 中观测 `C_i`，匹配专家 `Q_i` 是正样本，其他样本的专家轨迹是经验边缘负样本：

```text
q_{ij} = softmax_j(-ADE_{ij}),
L_rank = -E_i Σ_j q_{ij} log softmax_j(S(C_i,Q_{ij})).
```

这是 NavDP critic 的“候选生成后评价”结构和 X-NavDP 组内重加权思想在离线监督条件下的严格对应：目标分布不再假设专家位于第 0 个候选，且不伪装成在线 Q。推理时只在同一观测的八条 Flow 候选内计算 `candidate_log_probabilities=log_softmax(S)`，并选择最大质量 logit。

NavDP 的 privileged critic 和 X-NavDP 的 GQRM/RTC 仍需要 ESDF、碰撞 reward 或在线交互；当前实现只吸收其候选组排序接口，不把离线路径误差冒充安全值。要得到显式安全置信度，后续必须提供真实 ESDF/碰撞标签，或补齐相机外参和机器人 footprint 后实现 SanD 的解析几何评价。

## 总损失

唯一训练目标是

```text
L = 1.0 L_flow + 0.5 L_path + 0.1 L_tangent + 0.1 L_rank.
```

- `L_path`：`x_hat_1` 解码路径与等弧长 `reference_path` 的逐点 Smooth-L1。权重 `0.25+exp(-4s)` 归一化到均值 1，强调 receding-horizon 即将执行的近端。
- `L_tangent`：有效目标段上的 `1-cos(Δx_hat,Δx*)`，使用同一近端权重，约束初始转向和局部跟踪方向。
- `L_rank`：上述组内质量软标签交叉熵。

不对不同自由控制点设置手工 Flow 权重。B-spline 已保证连续性，路径和切向损失负责将执行几何质量传回 Flow。

## 参考设计吸收边界

| 参考 | 当前已吸收 | 当前未采用及原因 |
|---|---|---|
| SanD | shared depth spatial tokens、2-D/frame-slot 编码、8 点三次 B-spline、等弧长输出 | DDPM 被标准 CFM 替代；initial-turn 缺少因果训练字段；解析 ESDF 缺外参/footprint |
| NavDP | learned-query 多帧压缩、条件生成后候选评价 | 输入合同没有 RGB；没有 privileged ESDF 正负标签，故不冒充 safety critic |
| X-NavDP | PointGoal 的尺度边界；保留同状态候选归一化语义 | GQRM、twin Q、RTC 和 embodiment FiLM 都依赖在线 RL/执行历史/多本体数据 |
| FlowNav | 标准 conditional flow matching 与少步 ODE 推理 | 不预设其 NFE 优势会自动转化为 CurveNav 闭环增益 |

因此当前模型是“SanD 轨迹表示与深度时空 token + NavDP learned-query 压缩与 generate/rank 结构 + 标准 CFM”，而不是把三篇论文中监督条件互不兼容的模块机械叠加。

主要来源：[SanD-Planner](https://arxiv.org/abs/2602.00923)、[SanD 官方源码](https://github.com/WangJinCheng1998/sandplanner)、[NavDP](https://arxiv.org/abs/2505.08712)、[NavDP 官方源码](https://github.com/InternRobotics/NavDP)、[X-NavDP](https://arxiv.org/abs/2607.28560)、[FlowNav](https://arxiv.org/abs/2411.09524)、[Implicit Behavioral Cloning](https://arxiv.org/abs/2109.00137)。显式几何对照：[Semantic MapNet](https://arxiv.org/abs/2010.01191)、[PETRv2](https://arxiv.org/abs/2206.01256)、[BEVDet4D](https://arxiv.org/abs/2203.17054)、[LoGoPlanner](https://arxiv.org/abs/2512.19629)。

## 训练、推理与验证

### 数据合同

唯一数据入口是 `scripts/build_dataset.sh DATA_ROOT`。它依次下载固定 commit 的 SanD 与 HSSD、生成 HSSD 专家 route、准备两类 route 深度缓存，并原子编译 `data/policy_dataset`；训练与 GPU 调度不属于数据生成链。各阶段只接受空输出，不提供历史版本、恢复模式或已有输出分支。HSSD 的 20 个冻结场景按 16/4 划分 train/validation，并禁止同源 scene family 跨 split。每个场景生成 25 条无扰动的完整专家 route：近距 5 条、中距 10 条、远距 10 条，共 500 条，train/validation 分别为 400/100 条。

每条 HSSD route 沿 clearance-aware 路径按 0.15 m 等弧长采样，保存连续平面位姿以及逐位置 `224×126` metric depth，最终位置就是该 route 的任务 PointGoal。编译阶段在每个非终点位置切出一个监督样本：历史四帧按 `[-1.35,-0.90,-0.45,0] m` 索引，未来最多 24 步作为局部路径，并计算 `observation_to_current=(x,y,sin Δyaw,cos Δyaw)`。同一 route 的深度只保存一次，局部样本通过索引共享。生成门禁验证数量、连续 clearance、0.15 m 间距、距离分布、深度/位姿对齐、split 无泄漏、原子提交和最终 SHA。

SanD 深度由原始 `640×480` 相机重投影到 `224×126`；HSSD 直接用同一标定相机渲染。两者都归一化到 5 m 并按 route 打包，再编译成唯一 `data/policy_dataset`。训练期 loader 只读取统一张量和深度索引，不保留来源分支。

唯一链路：

```text
HSSD generation + SanD source -> calibrated prepared dataset
                              -> fixed-batch overfit -> DDP mixed-precision training -> EMA
                 -> offline geometry metrics -> official closed-loop benchmark
```

checkpoint 严格记录输入标定、编码器与生成器类型、平面反投影对齐、每帧 16 个压缩 token、八控制点、等弧长倍数和损失语义；合同不一致时直接拒绝加载，不设置兼容分支。推理不读取标签，也不执行曲率裁剪、碰撞 mask、直线候选或轨迹反转。

开始完整训练前只保留三类验证：

1. 数学与接口单测：B-spline、等弧长、逐帧 mask、平面反投影、learned-query token 数、组内质量损失和 checkpoint；
2. 前向/反向：所有输出形状正确，全部可训练参数具有有限梯度；
3. 固定批过拟合及同协议离线/闭环评测：selected/oracle ADE、弧长、曲率、反转率、候选多样性、延迟、SR/SPL。

离线 ADE 不能替代闭环 SR/SPL；候选组质量不能解释为安全概率；未进行 X-NavDP 在线 RL 时不得使用“Q 后训练”表述。
