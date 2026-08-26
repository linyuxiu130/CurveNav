# CurveNav 轨迹生成架构

本文是当前代码的唯一模型合同。CurveNav 是 PointGoal 条件的二维局部轨迹生成器：输入四帧度量深度、历史位姿和当前机器人系 PointGoal，输出一条由有界连续曲率定义的 64 点二维局部路径。当前阶段没有轨迹评价头、collision critic、Q head、候选代价加权或推理后轨迹修补。

## 1. 唯一张量合同与模型图

```text
depth                  float [B,4,1,126,224]  三帧历史 + 当前深度；1 表示 5 m
point_goal             float [B,2]            当前机器人系任务目标 (x,y)
observation_to_current float [B,4,4]           (x,y,sin Δyaw,cos Δyaw)
observation_valid      bool  [B,4]             历史补帧 mask；当前帧有效

target controls        float [B,8,2]
target reference path  float [B,64,2]
prediction path        float [B,64,2]
prediction geometry    heading/curvature [B,64]
```

生产模型固定为 `D=384`、8 attention heads。完整数据流为：

```text
4× metric depth
  └─ shared GroupNorm ResNet-18 stage-3 ── current 96 tokens ── 32 current queries ─┐
                                      └── all 4×96 tokens ── 32 context queries ───┤
4× (x,y,sin Δyaw,cos Δyaw) ──────────────── 4 explicit state tokens ──┤
PointGoal direction + log range ─────────────────────── goal token ───┤
                                                                      ↓
                                      4 ordered route queries, 2× cross-attention
                                      4× joint condition Transformer
                                                                      ↓
                                      8 learned curve proposal queries
                                      2× proposal Transformer
                                      deterministic executable source b(C)
                                                                      ↓
                                      fixed coordinate map y = 8x
                                      8× solver-collocated self-consistent Flow Transformer
                                      adaRMS-Zero(time, route)
                                                                      ↓
                    shared latent → velocity v and expert endpoint m
                                                                      ↓
                 b(C) → expert transport, 8-step velocity-Heun integration
                                                                      ↓
          PointGoal-scaled arc-length/curvature decoder → metric local path
```

训练和部署只组装这一张图。checkpoint 类型为 `curvenav_geometry_supervised_proposal_solver_collocated_self_consistent_flow_bounded_curvature_policy`，旧模型在加载前被严格拒绝，不存在兼容分支。

## 2. 视觉几何、历史状态与路线融合

### 2.1 度量深度 memory

每帧由同一个单通道 ResNet-18 编码到 stride-16 stage-3 特征，再池化为 `8×12=96` 个 token。卷积使用 GroupNorm，表示不依赖单卡 batch 或 DDP rank 的运行统计。token 同时包含二维图像位置编码和标定相机反投影得到的度量平面点。

对光轴深度 `z` 和像素 `(u,v)`，相机下俯角为 `α`、相对机器人前移为 `a`：

```text
x_o = z(u-cx)/fx,
y_o = z(v-cy)/fy,
p_body = (a + cos(α)z - sin(α)y_o, -x_o).
```

部署输入允许 benchmark 在保持 horizontal/vertical aperture 不变时改变像素采样率。
若参考内参 `K` 的分辨率为 `(W,H)`，实际深度分辨率为 `(W',H')`，则先使用
`K' = diag(W'/W,H'/H,1)K`；这保持每个归一化像平面射线和 FoV 不变，再重投影到
固定的 `224×126` 模型相机。它是同一相机合同的解析重采样，不是图像尺寸 fallback。

历史点用真实位姿变换到当前机器人系：

```text
p_current = R(Δyaw) p_body + (tx,ty).
```

每个 token 图像格取最近可见表面而不是平均深度，避免把近障碍和远背景平均成不存在的中间表面。无效历史帧在 depth backbone 前置零，并在 geometry cross-attention 的 key/value 侧屏蔽。

四帧一共 384 个视觉 token。单个 cross-attention 压缩器使用 64 个与目标无关的 learned geometry query：前 32 个通过固定 attention mask 只读取当前帧的 96 个 token，后 32 个读取全部有效时序 token。输出仍是固定 64 token，没有第二个视觉编码器、条件分支或额外参数。这个约束保证即时障碍几何不会在 384-token 历史 memory 中被稀释，同时保留对齐历史带来的视野补全；在 episode 冷启动只有当前帧时，两组 query 都自然读取当前帧，不需要 fallback。这里刻意不在视觉压缩前注入 PointGoal：障碍、通道和可通行边界是目标无关的场景事实；让目标过早控制压缩会丢掉当前路线之外、但绕障时可能需要的几何信息。这保留 NavDP 的 query compression 效率，吸收其当前深度与历史 memory 职责分离，以及 LoGoPlanner 将 geometry query 与 planning query 分工的做法。

### 2.2 显式状态和 PointGoal

`observation_to_current` 不只用于搬动深度点，也经过 MLP 形成四个 state token。平移除以 1.35 m 历史窗口尺度，旋转直接使用 `sin/cos`，不存在角度跳变。补帧使用 learned invalid-state token。

PointGoal 编码为方向和对数距离：

```text
r = ||g||,
feature(g) = [g/max(r,ε), log(1+min(r,25 m))/log(26)].
```

CurveNav 的任务合同始终提供 PointGoal，因此不采用 NoMaD/NavDP 为统一 goal-conditioned/goal-agnostic 策略而使用的 50% goal mask。对本任务机械照搬该 mask 会无依据地删除一半目标监督。

### 2.3 有序路线查询与条件曲线提案

单个二维局部终点不能区分绕过同一障碍的不同路线形状，也会把近端可执行方向和远端进展压进同一个 latent。CurveNav 使用四个有序 learned route query；每个 query 与 goal token 相加，连续两次 cross-attend `[state, geometry]`。随后 `[4 route, goal, 4 state, 64 geometry]` 共 73 个 token 经过四层联合 condition Transformer。四个 route latent 保留不同路线推理槽位，归一化均值形成逐层调制向量：

```text
c_i = RMSNorm(q_i),
c_route = RMSNorm(mean_i c_i).
```

不再从这四个 latent 另行回归一组与执行输出并列的 XY 锚点。八个 learned curve proposal query 经过两层 self-attention、对全部 73 个条件 token 的 cross-attention、SwiGLU 和 `c_route` adaRMS-Zero 调制，直接预测归一化的八个有界曲率坐标 `b(C)`。它同时接受三项同目标监督：生产 decoder 实际消费的 expert coordinate MSE、解码后 metric path loss 和 tangent loss。坐标项固定 Flow 源的内禀语义；metric path/tangent 项通过真实非线性 decoder 的 Jacobian 约束执行几何，尤其是同样坐标误差会造成更大航向偏差的强转弯。三项只训练同一个 proposal，不产生辅助轨迹或并列 head。

这个组合来自实测纠错而不是冗余 loss：coordinate-only self-consistent 图在 step 4800 的 proposal 强转弯 ADE 为 `0.4488 m`，明显差于旧 metric-supervised proposal 的约 `0.3711 m`。因此不能用内禀坐标 MSE 取代执行空间几何；两者分别约束参数辨识与导航误差。

这保留了 SanD 的结构化低维轨迹控制和 NavDP/X-NavDP 的有序动作查询，同时修复两者在“没有候选评价器、只执行一条轨迹”约束下不能直接照搬随机生成的缺口。route query 负责从 geometry/state/goal 推理低频拓扑，curve proposal query 负责把拓扑写成可执行曲线坐标，残差 Flow 只做连续细化；三者职责不重叠。

## 3. PointGoal 标度的有界曲率曲线

旧实现约束 XY 控制多边形的离散转角，但该条件不能推出最终三次 B-spline 的连续曲率上界。CurveNav 现在直接生成弧长域曲率函数。八个未来 token 的唯一语义为：

```text
token 0:  [total arc-length coordinate, initial-heading coordinate]
token 1..7: [curvature-control coordinate, fixed zero]
```

令 `H=3.6 m`、`ρ=min(||g||,H)`，总弧长与初始航向为：

```text
c = softplus_inverse(1),
L = ρ softplus(a+c),
θ0 = (π/2)tanh(h).
```

因此零坐标对远目标给出 3.6 m 直线，对近目标自然缩短到目标距离；当 PointGoal 为零时 `L=0`，任意 Flow 输出都严格解码为停止轨迹。初始切向始终位于机器人前半平面。

其余七个坐标生成夹持三次 B-spline 的曲率控制：

```text
ci = κmax tanh(ri),
κ(u) = Σ_i B_i,3(u) ci,
κmax = 8 m^-1.
```

三次 B-spline 基函数满足 `B_i,3(u)≥0` 且 `Σ_i B_i,3(u)=1`，所以对任何有限网络输出都有严格的连续曲率界：

```text
|κ(u)| ≤ Σ_i B_i,3(u)|ci| < κmax.
```

曲线由 Frenet 方程定义：

```text
s = Lu,
θ(u) = θ0 + L∫_0^u κ(v)dv,
p(u) = L∫_0^u [cos θ(v), sin θ(v)]dv,
p(0) = (0,0).
```

曲率 B-spline 为 `C²`，因此连续模型中的航向为 `C³`、位置为 `C⁴`；可执行性不再依赖 XY 控制多边形近似、推理裁剪或 MPC 兜底。代码在固定 `4×` 密集弧长网格上以 midpoint/circular-chord 积分，再等间隔抽取 64 点。所有矩阵均在初始化时预计算，训练和部署的 shape 固定。

训练标签中的八点平面 B-spline 只承担专家路径去噪和端点保持，不是生产输出表示。其平滑路径曲率 `κ*` 通过固定正则最小二乘投影到七个曲率控制：

```text
c* = (BᵀB + 0.3 I)^-1 Bᵀκ*,
r* = atanh(c*/κmax).
```

当前 30,642 个训练样本和 7,960 个验证样本的目标坐标全部有限，最大绝对值分别为 `2.710/1.113`；投影到可行曲线后，相对原专家路径的平均点误差为 `3.83/4.02 mm`，99% 样本的最大点误差为 `8.78/8.72 cm`。训练、离线评估和部署共用同一个有界曲率 decoder。

离线 ADE/RMSE 仍以原始等弧长专家路径为准；参考曲率则从上述去噪 B-spline 计算。离散专家折线在顶点处的有限差分曲率会被采样角点放大，不能作为连续生成曲线的正确曲率基准。

## 4. 条件曲线源上的 Self-Consistent Rectified Flow Transformer

### 4.1 训练与部署同源的 Flow Matching

三帧已执行历史已经通过 metric depth alignment 和四个显式 state token 进入条件序列。几何 decoder 消费的曲线坐标记为 `x∈R^(8×2)`。它们受硬几何尺度约束，当前验证集九个自由维度的标准差只有 `0.014–0.058`；直接把 `x` 当作 FP16 Flow 状态会使速度场长期处于 `10^-2` 量级，并让零初始化网络偏向无需修正的直线。训练和推理因此都使用唯一的固定无量纲坐标变换：

```text
y = 8x,       x = y/8.
```

尺度 8 把九个自由维度的典型标准差移到 `0.11–0.47`，同时保持零点、相对维度权重、可表示轨迹集合和曲率硬界完全不变。它对应 SanD/Diffusion Policy 对控制坐标做归一化的必要数值条件，但不读取验证统计、不保存数据集专用 normalizer，也不产生第二套 decoder。Flow 只生成 `y`，不把已知历史复制成生成目标。

随机扩散或高斯 Flow 只有在采样多条候选并由碰撞代价、critic 或 Q 值选择时，才定义了完整的局部决策。CurveNav 当前明确不使用评价头且只输出一条轨迹；旧结构却用高斯源训练、部署固定从零点积分。高斯零点既不是条件分布的均值/众数，也没有被监督成安全路线，这构成训练—部署起点错配。

当前结构先由条件网络产生唯一可执行源 `b_φ(C)`，再学习它到专家曲线的直线 Flow：

```text
y_1 = 8x_1,
b = b_φ(C) on the nine free coordinates,
k ~ Uniform{0,...,8},  t = k/8,
y_t = (1-t)sg(b) + t y_1,
u_t = d y_t/dt = y_1-sg(b),
L_flow = Σ mask·||v_θ(y_t,t,C)-(y_1-sg(b))||² / Σ mask.
```

每个 rank 用随机循环偏移把一个 batch 分层铺满九个 Heun 节点，相邻节点样本数最多相差一；因此部署必用的 `t=0`、内部节点和终点 `t=1` 每步都得到直接监督。它是固定 Heun8 求解器上的 collocation：不新增时间分支、边界 loss 或第二个训练入口，也不把只对独立高斯源成立的 Boundary RF 闭式公式错误套到条件 proposal 源。连续均匀采样在有限 batch 中几乎必然不含精确 `t=0`；coordinate-only 图在 step 4800 的强转弯 `t=0` velocity-residual cosine 为 `-0.060`，随后八步积分把强转弯 ADE 从 proposal 的 `0.4488 m` 恶化到 `0.4561 m`，这正是训练节点与部署节点错配。

`sg` 表示 Flow loss 不通过源坐标反向传播；提案由独立的坐标与 metric geometry 监督学习，避免提案与速度场仅靠互相抵消降低 velocity MSE。condition encoder 仍同时接收提案和 Flow 梯度。七个曲率 token 的第二通道由结构 mask 从提案、状态、损失和 ODE 更新中同时移除。

旧模型虽然在训练点拟合局部速度，但 held-out 诊断显示八步积分位移与真实残差夹角接近正交，step 8000 的 Flow 甚至把 proposal ADE 从 `0.0818 m` 恶化到 `0.0991 m`。根因是有限数据下只约束局部速度，不保证不同 `t` 的速度指向同一专家终点。当前 Flow 在同一个 Transformer latent 上同时预测局部速度 `v_θ` 与数据端点 `m_θ`：

```text
L_v = MSE_mask(v_θ, y_1-sg(b)),
L_x = MSE_mask(m_θ, y_1),
y_hat_1 = y_t + (1-t)v_θ,
L_c = MSE_mask(m_θ, y_hat_1),
L_flow = L_v + L_x + 0.1 L_c.
```

这是 Self-Consistent Flow 在线性 rectified path 上的代数恒等式：精确速度从任意 `t` 外推都必须到达同一个 `y_1`。端点监督提供低方差全局方向，velocity 监督在 `t→1` 时不含除以 `1-t` 的误差放大；`0.1` 是原论文一致性权重有效区间中的最小固定值。它不是轨迹评价头：不产生安全分数、不比较候选，也不改变执行路径。一步干净端点估计仍由稳定的 velocity 分支给出：

```text
y_hat_1 = y_t + (1-t)v_θ(y_t,t,C).
```

先用唯一逆变换 `x_hat_1=y_hat_1/8` 解码，再施加 metric path 和 tangent 监督，Flow 学到的不只是内禀坐标均方误差。部署使用完全相同的 `b_φ(C)`，并只积分数值稳定的 velocity 分支；endpoint 分支只在训练时约束共享 latent，因此没有 source distribution、采样温度、端点除法或特殊零点。

### 4.2 轨迹 Transformer

提案器有 8 个 learned curve token 和两层 conditional block；Flow 有另一组 8 个 future token embedding 和八层 conditional block。两者的每层均包含：

1. 8 token 双向 self-attention，使弧长、初始航向和七个曲率控制直接交换信息；
2. 对 73 个 condition token 的 cross-attention，保留局部几何和有序路线 query 的 token 级信息；
3. SwiGLU feed-forward；
4. adaRMS-Zero shift、scale 和 residual gate：提案器由 `c_route` 调制；Flow 由 Fourier time 与 `c_route` 共同调制。

adaRMS-Zero 的各分支 gate 零初始化，使深层残差分支平滑打开；提案坐标 projection、Flow velocity projection 和 endpoint projection 保留 PyTorch 线性层的方差缩放权重初始化并将 bias 置零，从第一步即可向输出头传递梯度。velocity 与 endpoint 只各增加一个 `D→2` projection，共用全部八层 Flow Transformer 表示，不复制生成网络。相较只在输入拼接一次时间/目标，逐层调制让路线和 Flow 时间控制每个残差更新；这吸收 DiT 的 adaptive-normalization 稳定性和 X-NavDP 的 FiLM 条件注入思想，但 geometry 仍通过 cross-attention 保持空间分辨率。

生产深度不是为了堆参数而设置：条件端四层负责 geometry/state/goal 路线推理，提案端两层把路线写入可执行坐标，Flow 端八层负责九个实际求解时间节点上的残差修正。

### 4.3 唯一推理轨迹

部署先计算 `y0=b_φ(C)`，再从这个与训练一致的条件源对 velocity 分支做八步 Heun 积分：

```text
y' = v_θ(y,t,C),
y_predict = y + Δt y',
y_next = y + Δt/2 [y' + v_θ(y_predict,t+Δt,C)].
```

输出只有一个 `[B,8,2]` 归一化 Flow 状态，经固定除 8 后送入曲线 decoder；不存在随机 episode 状态、候选维、候选排序、历史重建分数或 oracle 选择。SanD 的随机扩散 batch 后接 ESDF 评价，NavDP/X-NavDP 的多样候选后接 critic/Q 约束；没有评价器时随机执行其中一条并不是完整决策架构。条件提案把唯一输出的路线选择显式交给受监督网络，Flow 只细化该选择；全部时序一致性仍来自真实执行历史，不递归传播上一周期预测误差。

## 5. 唯一训练目标

所有项都先无量纲化，再直接相加：

```text
L = (L_v + L_x + 0.1L_c) + L_path + L_tangent + L_proposal.
```

- `L_v`：O(1) 条件提案源到专家端点的 velocity masked MSE。
- `L_x`：同一 Flow latent 对专家端点的 masked MSE。
- `L_c`：endpoint 与 velocity 隐含端点的 masked MSE；固定权重 `0.1`。
- `L_path`：每个等弧长位置的欧氏误差 `||p_hat-p*||₂/H`；使用 `0.25+exp(-4s)` 并归一到均值一，强调马上要执行的近端。
- `L_tangent`：有效相邻路径段的 `1-cos(Δp_hat,Δp*)`，使用相同近端权重。
- `L_proposal=L_proposal_coord+L_proposal_path+L_proposal_tangent`：同一条件提案分别匹配专家生产曲线坐标、平滑专家 metric path 和 tangent。它直接监督部署/Flow 实际使用的起点，而不是训练一个并列锚点 head。

不另加平滑 loss 或后处理：连续曲率界和高阶路径平滑由弧长域参数化直接给出，路径和切向项负责实际 metric 几何。训练日志固定记录总损失、聚合 Flow、三项 Flow 子损失、最终 path/tangent、聚合 proposal 和三项 proposal 子损失共十一项，避免总 loss 掩盖某个子任务失效。

## 6. 相对论文方案的取舍

| 来源 | 吸收的有效设计 | CurveNav 的针对性改进 |
|---|---|---|
| SanD | 四帧共享深度 backbone、空间 token、平滑低维轨迹先验、候选生成后评价 | 教师 B-spline 只做标签平滑；生产输出改为 PointGoal 标度的连续有界曲率曲线；没有 ESDF 评价器时用受监督条件提案完成唯一拓扑选择 |
| NavDP | `D=384` actor、当前深度与历史 memory 分工、learned-query 视觉压缩、trajectory-token cross-attention、扩散动作生成 | 32 个 current query 保证即时几何，32 个 context query 补全历史视野，四个有序 route query 和八个 proposal query 直接形成可执行源；同一无量纲坐标用于训练和 ODE |
| X-NavDP | 深层条件生成器、逐层 FiLM 思想、闭环时序一致性 | 用 adaRMS-Zero 做 time-route 调制；真实执行历史承担时序状态；当前阶段不做 GQRM/RL，也不随机执行未经 Q 选择的候选 |
| LoGoPlanner | geometry/state/route 的任务专用 query | CurveNav 已有标定 metric depth 和真实位姿，不复制重型视频三维重建模型；route query 做拓扑推理，curve proposal query 输出实际执行坐标 |
| Past-Token Prediction | 用可观测过去约束未来的思想 | 历史已经由 condition encoder 显式编码；不再联合生成过去，因为过去重建误差不能代表未来轨迹质量 |
| Flow Matching / DiT | 条件直线概率路径、少步 ODE、adaptive normalization | 在与可执行坐标严格双射的 O(1) 空间，从受监督条件曲线源学习短残差 transport；训练分层覆盖部署 Heun8 的九个 collocation 节点，训练和部署从同一点、同时间支持出发 |
| Self-Consistent Flow | 共享表示联合预测 velocity 与 endpoint，并用代数一致性改善有限数据优化和路径直度 | 两个轻量 projection 一次共享 Transformer 前向同时输出；训练约束全局端点，部署只积分近数据端稳定的 velocity，不引入候选或评价头 |
| NoMaD | 生成分布适合多模态局部行为 | 不照搬为 goal-agnostic 统一策略服务的 50% goal mask；CurveNav 始终严格 PointGoal 条件 |

主要来源：[SanD 论文](https://arxiv.org/abs/2602.00923) 与 [官方源码](https://github.com/WangJinCheng1998/sandplanner)、[NavDP 论文](https://arxiv.org/abs/2505.08712) 与 [官方源码](https://github.com/InternRobotics/NavDP)、[X-NavDP](https://arxiv.org/abs/2607.28560)、[LoGoPlanner](https://arxiv.org/abs/2512.19629)、[Past-Token Prediction](https://arxiv.org/abs/2505.09561)、[NoMaD](https://arxiv.org/abs/2310.07896)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003)、[Self-Consistent Flow](https://arxiv.org/abs/2607.12171)、[Consistency Flow Matching](https://arxiv.org/abs/2407.02398)、[DiT](https://arxiv.org/abs/2212.09748)。

## 7. 训练、推理和验证边界

训练只读取 `data/policy_dataset`。该目录当前仅包含本项目在固定 HSSD 资产上生成、按 Dingo 标定相机渲染的深度和专家轨迹，不混用论文作者的数据。四帧历史按行驶距离 `[-1.35,-0.90,-0.45,0] m` 取样；未来最多 24 个 `0.15 m` 专家点，近目标自然缩短。

唯一训练入口使用 FP16、GPU 常驻 depth bank、异步 prefetch、AdamW、cosine schedule、EMA 和静态 `torch.compile`；多卡时由同一入口启用 DDP。数学 batch 固定为 1024，显存 micro-batch 上限为每卡 171。对 world size `W`，每 rank 分配 `floor(1024/W)` 或 `ceil(1024/W)` 个互不重叠样本；局部 batch 均值乘 `W·B_r/1024` 后再经 DDP 求平均，严格得到全局 1024 样本均值。6 卡时分配为 `171×4 + 170×2`，每 rank 只执行一次前后向并立即同步；这样消除了旧上限 112 带来的第二个小 micro-batch 和一次额外模型调度。1–8 卡的每次 optimizer、schedule 与 EMA 更新仍保持同一数学合同，不需要 padding、重复样本或改变学习率。前一 coordinate-only self-consistent 图在单张 2080 Ti、真实数据、静态编译、FP16 完整前后向下实测 batch 171 峰值 allocated/reserved 为 `7.58/9.35 GiB`，六卡稳态 `4340–4396 samples/s`；因强转弯证据在 step 5580 主动停止。当前 geometry-supervised/collocated 图增加一次批量化 proposal decoder，必须重新验证 batch 171 显存与六卡实际吞吐。checkpoint 保存逐 rank CUDA RNG 以精确恢复求解节点循环偏移；部署没有 source RNG 或 episode latent，始终从当前条件提案积分。唯一部署入口加载 EMA 权重并使用上述单轨迹八步 velocity-Heun，没有部署 fallback。

在线部署使用 eager FP16 推理，不把分钟级编译成本放进短回合测评。`navigator_reset` 已知实际 batch size 后立即使用零观测完成 CUDA kernel 初始化和同步；该步骤发生在 evaluator 的 episode 循环开始前。因此首个真实观测不会承担初始化时间，也不会让机器人在开局持续执行零动作。训练仍使用静态 `torch.compile`，因为 8000 个优化器 step 足以摊薄一次编译成本；1024 样本离线检查同样使用 eager，避免编译时间超过实际评估计算。

必要验证分三层：

1. 张量/数学单测：相机反投影、历史 mask、geometry/route/proposal token、训练—部署同源 Flow、PointGoal 标度、零目标停止、连续曲率硬界、教师 B-spline、checkpoint 和部署接口；
2. 前向/梯度：训练 loss、全部参数梯度、条件提案的确定性推理、静态编译图和有限值；
3. 重新训练后的 held-out/闭环：ADE、弧长、目标进展、曲率、延迟，以及固定协议 SR/SPL。

旧 checkpoint 的低闭环成绩可以证明旧链路失败，但不能单独证明新模块有效。当前结构必须从头训练；离线几何通过后再进入固定协议闭环，最终结论以 SR/SPL 为准。
