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
production coordinates float [B,8]
prediction path        float [B,64,2]
prediction geometry    heading/curvature [B,64]
```

生产模型固定为 `D=384`、8 attention heads。完整数据流为：

```text
4× metric depth
  └─ shared GroupNorm ResNet-18 stage-3 ── current 96 tokens ── 32 current queries ─┐
                                      └── all 4×96 tokens ── 32 context queries ───┤
4× (x,y,sin Δyaw,cos Δyaw,valid) ── 1 ordered history-state token ───┤
PointGoal direction + log range ─────────────────────── goal token ───┤
                                                                      ↓
                                      4 ordered route queries, 2× cross-attention
                                      4× joint condition Transformer
                                                                      ↓
                                      8 ordered curve queries
                                      8× conditional trajectory Transformer
                                      adaRMS-Zero(route)
                                                                      ↓
                                      direct normalized coordinates y = 8x
                                                                      ↓
          bounded-horizon zero-tangent curvature decoder → metric local path
```

训练和部署只组装这一张图。checkpoint 类型为 `curvenav_direct_geometry_supervised_bounded_curvature_policy`，旧模型在加载前被严格拒绝，不存在兼容分支。

## 2. 视觉几何、历史状态与路线融合

### 2.1 度量深度 memory

每帧由同一个单通道 ResNet-18 编码到 stride-16 stage-3 特征，再池化为 `8×12=96` 个 token。卷积使用 GroupNorm，表示不依赖单卡 batch 或 DDP rank 的运行统计。token 同时包含二维图像位置编码和标定相机反投影得到的机器人系三维表面点。旧实现只保留平面 `(x,y)`，配置中的相机高度没有进入模型，因此下视相机看到的地面与竖直障碍在显式几何中不可区分；当前实现保留高度 `z_b`，不再要求 CNN 单独猜测可跨越地面。

对光轴深度 `z` 和像素 `(u,v)`，相机下俯角为 `α`、相对机器人前移为 `a`：

```text
x_o = z(u-cx)/fx,
y_o = z(v-cy)/fy,
p_body = (a + cos(α)z - sin(α)y_o,
          -x_o,
          h - cos(α)y_o - sin(α)z).
```

部署输入允许 benchmark 在保持 horizontal/vertical aperture 不变时改变像素采样率。
若参考内参 `K` 的分辨率为 `(W,H)`，实际深度分辨率为 `(W',H')`，则先使用
`K' = diag(W'/W,H'/H,1)K`；这保持每个归一化像平面射线和 FoV 不变，再重投影到
固定的 `224×126` 模型相机。它是同一相机合同的解析重采样，不是图像尺寸 fallback。

历史点用真实位姿变换到当前机器人系：

```text
(x_c,y_c) = R(Δyaw)(x_b,y_b) + (tx,ty),
z_c = z_b.
```

每个 token 图像格取最近可见表面而不是平均深度，避免把近障碍和远背景平均成不存在的中间表面。无效历史帧在 depth backbone 前置零，并在 geometry cross-attention 的 key/value 侧屏蔽。

四帧一共 384 个视觉 token。单个 cross-attention 压缩器使用 64 个与目标无关的 learned geometry query：前 32 个通过固定 attention mask 只读取当前帧的 96 个 token，后 32 个读取全部有效时序 token。输出仍是固定 64 token，没有第二个视觉编码器、条件分支或额外参数。这个约束保证即时障碍几何不会在 384-token 历史 memory 中被稀释，同时保留对齐历史带来的视野补全；在 episode 冷启动只有当前帧时，两组 query 都自然读取当前帧，不需要 fallback。这里刻意不在视觉压缩前注入 PointGoal：障碍、通道和可通行边界是目标无关的场景事实；让目标过早控制压缩会丢掉当前路线之外、但绕障时可能需要的几何信息。这保留 NavDP 的 query compression 效率，吸收其当前深度与历史 memory 职责分离，以及 LoGoPlanner 将 geometry query 与 planning query 分工的做法。

### 2.2 显式状态和 PointGoal

`observation_to_current` 不只用于搬动深度点，也形成一个有序 history-state token。平移除以 1.35 m 历史窗口尺度，旋转直接使用 `sin/cos`，不存在角度跳变。对每个 frame slot，先把无效变换严格置零，再拼入对应 validity；四个五维 slot 按时间顺序展平后只经过一个 MLP：

```text
z_state = MLP(vec([valid_i · (x_i/1.35,y_i/1.35,sin Δψ_i,cos Δψ_i), valid_i]_(i=1..4))).
```

因此无效 state 根本不会成为 condition key/value，也不需要 learned invalid-state embedding 或下游 attention padding mask；slot 顺序和可用历史长度仍显式可辨。视觉侧的无效深度 token 则继续在唯一 geometry compressor 的 key/value 侧严格屏蔽。

部署中每个 episode 必然依次经历 1、2、3、4 个有效观测，旧训练集却有 `7,260/7,960=91.2%` 的验证样本已经具备完整四帧。当前训练因此对每个原样本已有的 `m` 个有效后缀，采样

```text
K ~ Uniform{1,...,m},
valid'_j = valid_j · 1[j ≥ 4-K].
```

这不是删除深度或独立的数据增强分支，而是对物理上可能获得的历史长度做 Monte-Carlo 边缘化：优化目标变为 `E_(sample,K)[L(f(O_(t-K+1:t),g),p*)]`。每个被保留的历史仍使用真实深度和真实位姿；训练、恢复和 FP16 overflow retry 共用 CUDA RNG 状态，精确恢复同一前缀流。这样晚期路线样本也能作为合法 episode 冷启动状态训练，同时完整历史仍以 `K=4` 进入同一张图。

PointGoal 编码为方向和对数距离：

```text
r = ||g||,
feature(g) = [g/max(r,ε), log(1+min(r,25 m))/log(26)].
```

CurveNav 的任务合同始终提供 PointGoal，因此不采用 NoMaD/NavDP 为统一 goal-conditioned/goal-agnostic 策略而使用的 50% goal mask。对本任务机械照搬该 mask 会无依据地删除一半目标监督。

### 2.3 有序路线查询与条件曲线解码

单个二维局部终点不能区分绕过同一障碍的不同路线形状，也会把近端可执行方向和远端进展压进同一个 latent。CurveNav 使用四个有序 learned route query；每个 query 与 goal token 相加，连续两次 cross-attend `[history-state summary, geometry]`。随后 `[4 route, goal, 1 state, 64 geometry]` 共 70 个 token 经过四层联合 condition Transformer。四个 route latent 保留不同路线推理槽位，归一化均值形成逐层调制向量：

```text
c_i = RMSNorm(q_i),
c_route = RMSNorm(mean_i c_i).
```

不再从这四个 latent 另行回归一组与执行输出并列的 XY 锚点。八个 learned curve query 经过八层 self-attention、对全部 70 个条件 token 的 cross-attention、SwiGLU 和 `c_route` adaRMS-Zero 调制，直接预测归一化的八个有界曲率坐标。它同时接受三项同目标监督：生产几何解码器实际消费的 expert coordinate MSE、解码后 metric path loss 和 tangent loss。坐标项固定每个 token 的内禀语义；metric path/tangent 项通过真实非线性 decoder 的 Jacobian 约束执行几何，尤其是同样坐标误差会造成更大航向偏差的强转弯。三项只训练同一条执行轨迹，不产生辅助轨迹或并列 head。

这个组合来自实测纠错而不是冗余 loss：coordinate-only 图在 step 4800 的直接曲线强转弯 ADE 为 `0.4488 m`，加入 metric path/tangent 后降到 `0.2813 m`。因此不能用内禀坐标 MSE 取代执行空间几何；两者分别约束参数辨识与导航误差。

这保留了 SanD 的结构化低维轨迹控制和 NavDP/X-NavDP 的有序动作查询，同时修复两者在“没有候选评价器、只执行一条轨迹”约束下不能直接照搬随机生成的缺口。route query 负责从 geometry/state/goal 推理低频拓扑，curve query 负责把拓扑一次写成可执行曲线坐标；不存在第二生成阶段去改写已经受监督的路线。

## 3. 有限规划域内的零切向有界曲率曲线

旧实现约束 XY 控制多边形的离散转角，但该条件不能推出最终三次 B-spline 的连续曲率上界。CurveNav 现在直接生成弧长域曲率函数。八个未来 token 的唯一语义为：

```text
token 0:    total arc-length coordinate
token 1..7: curvature-control coordinate
```

令 `H=3.6 m`、`d=||g||`。现有 30,642/7,960 条训练/验证标签的弧长全部严格小于 `H`；相对 `min(d,H)` 的最大比值为 `1.954/1.812`。唯一标签支持域和生产解码域因此固定为：

```text
Lmax = min(H, 2d),
L = Lmax sigmoid(8a),
θ(0) = 0.
```

这同时满足 `0≤L≤H`、`L≤2d`：任何有限网络输出都不能越过当前 3.6 m 局部规划域；PointGoal 为零时严格有 `L=0`。旧 softplus 只有正值约束却没有上界，预测误差可以把短程参考放大到观测域外。初始切向现在严格等于机器人前向轴，而不是允许瞬时跳到 `±90°`；这与弧长参数化差速运动曲线的边界条件一致。零长度坐标对应 `Lmax/2`，近目标直达标签的自然中心也是 `d=Lmax/2`。

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
θ(u) = L∫_0^u κ(v)dv,
p(u) = L∫_0^u [cos θ(v), sin θ(v)]dv,
p(0) = (0,0).
```

曲率 B-spline 为 `C²`，因此连续模型中的航向为 `C³`、位置为 `C⁴`；可执行性不再依赖 XY 控制多边形近似、推理裁剪或 MPC 兜底。代码在固定 `4×` 密集弧长网格上以 midpoint/circular-chord 积分，再等间隔抽取 64 点。所有矩阵均在初始化时预计算，训练和部署的 shape 固定。

训练标签中的八点平面 B-spline 只承担专家路径去噪和端点保持，不是生产输出表示。为了与 `θ(0)=0` 一致，首个曲率样本由首段中点航向 `θ*_(1/2)` 和首段长度 `Δs_0` 定义为 `2θ*_(1/2)/Δs_0`；后续使用相邻段航向差除以中心支撑弧长。该曲率 `κ*` 再通过固定正则最小二乘投影到七个曲率控制：

```text
c* = (BᵀB + 0.3 I)^-1 Bᵀκ*,
r* = atanh(c*/κmax).
```

长度逆变换使用 `a=logit(L/Lmax)/8`；曲率控制超出可行连续曲率集合时投影到开区间边界，metric path/tangent loss 继续监督其最接近的可执行曲线。训练、离线评估和部署共用同一个有界曲率 decoder，不存在推理裁剪。

离线 ADE/RMSE 仍以原始等弧长专家路径为准；参考曲率则从上述去噪 B-spline 计算。离散专家折线在顶点处的有限差分曲率会被采样角点放大，不能作为连续生成曲线的正确曲率基准。

## 4. 单阶段有序曲线 Transformer

### 4.1 固定无量纲坐标

三帧已执行历史已经通过 metric depth alignment 和一个有序 history-state summary 进入条件序列。几何 decoder 消费的曲线坐标记为 `x∈R^8`：一个长度 logit 和七个有界曲率坐标。训练和部署共用唯一固定线性变换：

```text
y = 8x,       x = y/8.
```

尺度 8 使小曲率时的归一化目标近似实际曲率 `8·atanh(κ/8)≈κ`，长度维则恰好成为 `logit(L/Lmax)`。它不读取验证统计、不保存数据集专用 normalizer，也不产生第二套几何表示。

设条件序列为 `C`、route 调制向量为 `c_route`，唯一轨迹预测为：

```text
y_hat = f_θ(q_curve, C, c_route),
x_hat = y_hat / 8,
p_hat = Decode_bounded_curve(x_hat, point_goal).
```

八个 token 各输出一个有语义的标量，不再保留被 mask 的第二通道、死回归参数或 masked coordinate loss。训练和部署调用相同的 `f_θ` 与相同的几何 decoder，没有噪声源、时间变量、ODE、候选温度或上一周期预测状态。

### 4.2 八层轨迹解码器

解码器只有一组 8 个 learned curve token 和八层 conditional block。每层包含：

1. 8 token 双向 self-attention，使弧长和七个曲率控制直接交换信息；
2. 对 70 个 condition token 的 cross-attention，保留局部几何和有序路线 query 的 token 级信息；
3. SwiGLU feed-forward；
4. 由 `c_route` 控制的 adaRMS-Zero shift、scale 和 residual gate。

adaRMS-Zero 的各分支 gate 零初始化，使八层网络从稳定的 query 主干逐步打开；最终 `D→1` projection 保留方差缩放权重初始化并将 bias 置零，从第一步即可向输出头传递梯度。geometry 始终通过 cross-attention 保持空间分辨率，route 则逐层控制每次残差更新。这吸收 X-NavDP 的深层条件生成和 FiLM 思想，但把全部容量直接用于最终执行坐标，不再把十层容量拆成“2 层提案 + 8 层修正”。

### 4.3 为什么删除 Flow

随机扩散或 Flow 只有在多模态候选由碰撞代价、critic 或 Q 值选择时，才构成完整的局部决策。CurveNav 当前明确只有单专家监督、没有评价头且只执行一条轨迹；第二生成阶段必须在 held-out 上稳定优于直接监督轨迹才有存在依据。

geometry-supervised/self-consistent 实验在 step 4800 的 1024 条验证集上得到：直接曲线总体/强转弯 ADE 为 `0.07153/0.28128 m`，velocity-Heun8 为 `0.07661/0.29093 m`，论文默认 `τ=0.5` 的 Self-Consistent Flow 混合求解为 `0.07644/0.28993 m`，`t=0` endpoint 为 `0.07405/0.27972 m`。ODE 在总体与强转弯上都劣于直接曲线，endpoint 也牺牲总体和强转弯航向；增加边界采样、endpoint 一致性和 solver collocation 后仍不能消除这个现象。因此根因不是求解器步数或采样节点，而是无评价器时对已受监督决策做第二次欠约束改写。

当前实现据此删除完整 Flow 模块、时间嵌入、velocity/endpoint heads、积分器及其 loss；不是在推理时绕过仍存在的旧分支。输出只有一个 `[B,8]` 归一化坐标，经固定除 8 后送入有界曲率 decoder。全部时序一致性来自真实执行历史，不递归传播上一周期预测误差。

从头训练的未做历史边缘化直接基线 `9a9ca17` 进一步验证了这个判断。相同 1024 条 held-out 样本、EMA 权重和评估入口得到：

| step | 直接模型 ADE / 强转弯 ADE (m) | 旧 Flow 最终输出 ADE / 强转弯 ADE (m) |
|---:|---:|---:|
| 800 | `0.09915 / 0.40516` | `0.10915 / 0.47116` |
| 2400 | `0.07955 / 0.32453` | `0.09234 / 0.36964` |
| 4800 | `0.07195 / 0.29194` | `0.07661 / 0.29093` |

早中期两项都明确改善；step 4800 的总体 ADE 仍更好，而强转弯 ADE 基本持平。此时直接模型的强转弯终端航向误差为 `0.39878 rad`、全体最大曲率相关系数为 `0.76983`，分别优于旧 Flow 的 `0.42458 rad` 和 `0.74511`。因此现有证据支持删除第二生成阶段，但也明确指出剩余问题是强转弯几何学习和闭环执行，而不是重新加入随机 ODE。

上表只用于与旧 Flow 保持相同 1024 条样本的受控比较。唯一离线入口不再用固定前缀选 checkpoint，而是读取数据清单并遍历完整 7,960 条 held-out split；下表仍是 `9a9ca17` 基线：

| step | ADE (m) | 强转弯 ADE (m) | 强转弯终端航向 (rad) | 最大曲率相关系数 |
|---:|---:|---:|---:|---:|
| 800 | `0.10100` | `0.37511` | `0.59904` | `0.65327` |
| 2400 | `0.08294` | `0.31015` | `0.46369` | `0.72236` |
| 4800 | `0.07803` | `0.28821` | `0.42616` | `0.74210` |
| 6400 | `0.07806` | `0.28367` | `0.42337` | `0.73486` |
| 8000 | `0.07805` | `0.28196` | `0.43238` | `0.72692` |

总体 ADE 在 step 4800 后饱和，强转弯 ADE 到 step 8000 仍改善，但航向和曲率相关性开始波动。更关键的是，按自然历史长度分组后，四帧 ADE 从 step800 的 `0.09580 m` 降到 step8000 的 `0.07234 m`，而每回合必经的单帧 ADE 反而从 `0.18194 m` 升到 `0.21264 m`。把全部 7,960 条观测反事实地只保留当前帧后得到：

| step | 单帧 ADE (m) | 单帧强转弯 ADE (m) | 单帧强转弯曲率比 |
|---:|---:|---:|---:|
| 800 | `0.15516` | `0.51424` | `0.22762` |
| 2400 | `0.15183` | `0.49511` | `0.33438` |
| 4800 | `0.15765` | `0.48602` | `0.45063` |
| 6400 | `0.16256` | `0.49863` | `0.48782` |
| 8000 | `0.16802` | `0.49694` | `0.50808` |

这证明旧目标在不断优化占绝对多数的完整历史，同时牺牲冷启动条件；不是 100 个自然起始样本的偶然噪声。完全相同的 Home 场景 `MVUCSQAKTKJ5EAABAAAAABA8_usd`、seed1234、`num-envs=10` 的 10 回合吞吐诊断得到：

| step | SR | mean SPL | 10 回合耗时 (s) |
|---:|---:|---:|---:|
| 800 | `1/10` | `0.08687` | `466.54` |
| 2400 | `0/10` | `0` | `422.12` |
| 4800 | `0/10` | `0` | `434.91` |
| 6400 | `0/10` | `0` | `450.49` |
| 8000 | `0/10` | `0` | `581.98` |

该运行固定了同一 scene、episode、控制器和模型接口，但 `num-envs=10` 仅用于架构反馈，不作为 `num-envs=1` 固定协议成绩。后四个离线更优的 checkpoint 全部失败，证明较低总体 ADE 不能代表闭环成功。旧 `metric.csv` 中的 `distance` 是初始起终点直线距离，不是终止时剩余距离；旧日志也没有逐步位姿、局部轨迹和控制，所以现有 49/50 失败只能严格判定为 timeout，不能据此断言全部是碰撞、原地旋转或转弯后移。

“单一专家”也不是充分根因：SanD 同样能从少量单专家示范学习，但其完整系统生成多条局部样条并用深度构建的 ESDF 做安全/目标评价。CurveNav 当前按约束不增加评价头，因此生成器本身必须满足更强的几何合同。本轮已经从代码直接证实并根修三项错误：显式深度几何丢失相机高度、轨迹长度无局部上界、弧长曲线允许不连续的初始切向；同时删除由此产生的整条死坐标通道。专家路径净空不是同类错误：HSSD navmesh 已先按 `0.25 m` 机器人半径膨胀，记录的 `0.10 m` 是额外净空。沿专家切向采样导致纠偏状态不足仍是闭环分布风险，但不再作为 49 次失败的单独解释。

benchmark 唯一 evaluator 现在为每个完成回合写一个紧凑 numeric NPZ：仿真步记录世界位姿、机器人系目标、实际速度和已消费控制；重规划步记录模型局部轨迹、MPC 控制/预测状态、适应速度、参考曲率和 policy/MPC 延迟。下一轮只跑最小回合即可区分进展单调性、轨迹跨帧抖动、MPC 跟踪误差和卡住位置，再决定是否需要改变监督分布；旧权重与当前 XYZ/八标量合同严格不兼容，不能恢复训练。

## 5. 唯一训练目标

所有项都先无量纲化，再直接相加：

```text
L = L_coordinate + L_path + L_tangent.
```

- `L_coordinate`：归一化生产曲线坐标 `y_hat` 与专家坐标 `y*` 在八个真实自由维度上的 MSE。
- `L_path`：每个等弧长位置的欧氏误差 `||p_hat-p*||₂/H`；使用 `0.25+exp(-4s)` 并归一到均值一，强调马上要执行的近端。
- `L_tangent`：有效相邻路径段的 `1-cos(Δp_hat,Δp*)`，使用相同近端权重。

三项都监督同一个生产输出：坐标项保证参数语义可辨识，路径和切向项保证非线性解码后的真实执行几何。它们全部无量纲，直接相加，不使用数据集统计权重。也不另加平滑 loss 或后处理：连续曲率界和高阶路径平滑由弧长域参数化直接给出。训练日志只记录总损失和这三项，共四项。

## 6. 相对论文方案的取舍

| 来源 | 吸收的有效设计 | CurveNav 的针对性改进 |
|---|---|---|
| SanD | 四帧共享深度 backbone、空间 token、平滑低维轨迹先验、候选生成后以深度 ESDF 评价 | 教师 B-spline 只做标签平滑；生产输出改为严格有限规划域、零初始切向的连续有界曲率曲线；没有 ESDF 评价器时用直接监督解码器完成唯一拓扑选择，因此不能把 SanD 的少数据效果简化成“单专家自然收敛” |
| NavDP | `D=384` actor、当前深度与历史 memory 分工、learned-query 视觉压缩、trajectory-token cross-attention、点云几何和扩散动作生成 | 32 个 current query 保证即时几何，32 个 context query 补全历史视野；标定反投影保留机器人系 XYZ 而不是丢掉高度；四个 route query 和八个标量 curve query 一次形成可执行曲线，不随机执行未经 critic 选择的候选 |
| X-NavDP | 深层条件生成器、逐层 FiLM 思想、闭环时序一致性 | 用八层 adaRMS-Zero trajectory decoder 做 route 调制；真实执行历史承担时序状态；当前阶段不做 GQRM/RL，也不保留第二生成器 |
| LoGoPlanner | geometry/state/route 的任务专用 query | CurveNav 已有标定 metric depth 和真实位姿，不复制重型视频三维重建模型；route query 做拓扑推理，curve query 输出实际执行坐标 |
| Past-Token Prediction | 用可观测过去约束未来的思想 | 历史已经由 condition encoder 显式编码；不再联合生成过去，因为过去重建误差不能代表未来轨迹质量 |
| Flow Matching / DiT | O(1) 状态、adaptive normalization、连续生成 | 保留固定 O(1) 坐标和 adaptive normalization；删除实测持续恶化单轨迹的时间/ODE 分支，避免把多候选生成方法错误用于无评价器决策 |
| Self-Consistent Flow | velocity/endpoint 一致性和早晚时间分工 | 完整验证 endpoint、velocity-Heun 与论文默认 mixed sampler 后都未稳定优于直接曲线，因此不保留无收益的训练 head 或推理分支 |
| NoMaD | 生成分布适合多模态局部行为 | 不照搬为 goal-agnostic 统一策略服务的 50% goal mask；CurveNav 始终严格 PointGoal 条件 |

主要来源：[SanD 论文](https://arxiv.org/abs/2602.00923) 与 [官方源码](https://github.com/WangJinCheng1998/sandplanner)、[NavDP 论文](https://arxiv.org/abs/2505.08712) 与 [官方源码](https://github.com/InternRobotics/NavDP)、[X-NavDP](https://arxiv.org/abs/2607.28560)、[LoGoPlanner](https://arxiv.org/abs/2512.19629)、[Past-Token Prediction](https://arxiv.org/abs/2505.09561)、[NoMaD](https://arxiv.org/abs/2310.07896)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003)、[Self-Consistent Flow](https://arxiv.org/abs/2607.12171)、[Consistency Flow Matching](https://arxiv.org/abs/2407.02398)、[DiT](https://arxiv.org/abs/2212.09748)。

## 7. 训练、推理和验证边界

训练只读取 `data/policy_dataset`。该目录当前仅包含本项目在固定 HSSD 资产上生成、按 Dingo 标定相机渲染的深度和专家轨迹，不混用论文作者的数据。四帧历史按行驶距离 `[-1.35,-0.90,-0.45,0] m` 取样；未来最多 24 个 `0.15 m` 专家点，近目标自然缩短。

唯一训练入口使用 FP16、GPU 常驻 depth bank、异步 prefetch、AdamW、cosine schedule、EMA 和静态 `torch.compile`；多卡时由同一入口启用 DDP。CUDA allocator 固定使用 expandable segments，使编译图的临时 workspace 能复用已保留显存段，避免大块 reserved-but-unused 显存碎片化；它不捕获 OOM、不重试另一条图，也不改变 batch。数学 batch 固定为 1024，显存 micro-batch 上限为每卡 192。对 world size `W`，每 rank 分配 `floor(1024/W)` 或 `ceil(1024/W)` 个互不重叠样本；局部 batch 均值乘 `W·B_r/1024` 后再经 DDP 求平均，严格得到全局 1024 样本均值。6 卡时分配为 `171×4 + 170×2`，每 rank 只执行一次前后向并立即同步。1–8 卡的每次 optimizer、schedule 与 EMA 更新保持同一数学合同，不需要 padding、重复样本或改变学习率。单阶段八层图在 2080 Ti 上用真实数据、静态编译、FP16 完整更新连续运行 batch 192 至 step 121，稳态约 `793 samples/s`；batch 205 在反向图 OOM，因此 192 是保留实际余量的上限。正式六卡训练中 GPU0--5 稳定约 `97--98%` 利用率、每卡约 `9.5--10.7 GiB` 显存；排除编译、checkpoint 和溢出重试窗口后的 318 个日志窗口吞吐中位数为 `4,562 samples/s`，5%--95% 为 `4,526--4,604 samples/s`，删除 Flow 前为约 `4,340--4,400 samples/s`。完整 200 epoch/8,000 step 从进程创建到最终 checkpoint 共约 `34 min 04 s`。checkpoint 保存逐 rank CPU/CUDA RNG 以精确恢复训练数据流；部署没有 source RNG、episode latent 或数值积分，唯一入口加载 EMA 权重并单次前向一条轨迹。

在线部署使用 eager FP16 推理，不把分钟级编译成本放进短回合测评。`navigator_reset` 已知实际 batch size 后立即使用零观测完成 CUDA kernel 初始化和同步；该步骤发生在 evaluator 的 episode 循环开始前。因此首个真实观测不会承担初始化时间，也不会让机器人在开局持续执行零动作。训练仍使用静态 `torch.compile`，因为 8000 个优化器 step 足以摊薄一次编译成本；1024 样本离线检查同样使用 eager，避免编译时间超过实际评估计算。

离线入口自动使用 prepared manifest 中的完整 validation split，并校验实际遍历数等于清单样本数；当前为 7,960 条。它不提供短子集或随机抽样开关，避免训练后按有偏子集挑权重。

必要验证分三层：

1. 张量/数学单测：含相机高度的 XYZ 反投影、历史 mask、geometry/route/curve token、八标量坐标归一化、局部长度上界、零目标停止、零初始切向、连续曲率硬界、教师 B-spline、checkpoint 和部署接口；
2. 前向/梯度：三项训练 loss、全部参数梯度、直接曲线的确定性推理、静态编译图和有限值；
3. 重新训练后的 held-out/闭环：ADE、弧长、目标进展、曲率、延迟，以及固定协议 SR/SPL。

旧 checkpoint 的低闭环成绩可以证明旧链路失败，但不能单独证明新模块有效。当前结构必须从头训练；离线几何通过后再进入固定协议闭环，最终结论以 SR/SPL 为准。
