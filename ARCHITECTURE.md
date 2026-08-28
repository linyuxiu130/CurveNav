# CurveNav 轨迹生成架构

本文是当前代码的唯一模型合同。CurveNav 输入四帧度量深度、历史观测到当前帧的刚体变换和当前机器人系 PointGoal，以条件 Flow Matching 生成一条正弧长、连续曲率的二维局部轨迹。当前阶段没有评价头、多候选选择、RL 后训练或推理轨迹修补。

## 1. 失败证据与根因

上一版同场景 10 回合闭环为 `SR=5/10`，成功回合 SPL 为 `0.9055–0.9815`，证明坐标、HTTP、MPC 和成功判定链路能够工作。失败回合约 `91%` 的采样状态实际速度低于 `0.05 m/s`，但 MPC 约 `98%` 时间仍请求明显前进；失败轨迹 95% 最大曲率均值约 `0.48 m⁻¹`，成功轨迹反而约 `0.60 m⁻¹`。离线高转弯样本的预测总转角仅为专家的 `45%`。失败不是触碰 `4 m⁻¹` 曲率边界，而是困难状态下系统性欠转弯并重复同一条错误轨迹。

进一步对 checkpoint、失败轨迹和训练数据做边界与因果审计得到三个直接根因：

1. 训练曾固定 `x₀=0`，并使用 `x_t=t x₁`、`u*=x₁`。任意 `t>0` 都有 `x₁=x_t/t`，解码器可从带噪状态直接恢复标签而绕过深度与 PointGoal；真正需要条件的推理起点 `t=0` 在连续均匀训练中概率为零。实测速度 MSE 从 `t=0` 的 `1.337×10⁻³` 降到 `t=0.5` 的 `2.492×10⁻⁶`，相差约 537 倍。
2. 解码器只处理八个抽象标量，完整 Flow 积分结束后才首次解码路径。网络虽然读取视觉 token，却不知道当前中间状态对应的哪一段曲线靠近哪个障碍，因此没有 NavDP 式“当前生成轨迹与视觉逐层交互”。离线打乱当前深度仅使 ADE 增加 `0.00013 m`，而打乱 PointGoal 会使输出路径变化 `0.478 m`，说明模型主要学成了目标方向回归器。
3. HSSD 专家生成调用 `NavMeshSettings.set_defaults()` 后没有打开 `include_static_objects`；Habitat-Sim 的默认值为 false，而且默认 `agent_max_climb=0.20 m`，因此规划网格忽略静态家具并把部分低矮障碍视为可跨越。复现样本中，深度在前方 `1.42 m` 正确看到大型静态物体，旧专家却穿过该物体；将静态碰撞网格纳入机器人配置空间后，原路线后半段明确变为不可通行。旧训练监督因此直接教模型忽略可见障碍。

当前实现从数学链路上删除这些问题：训练源使用独立高斯随机变量，部署从同一先验典型集中的固定 latent 出发；Flow 的每一次速度场求值都先解码当前曲线，并显式建立路径锚点与当前深度障碍之间的度量注意力；专家 navmesh 强制包含 stage 和全部静态刚体；预测的干净轨迹还接受可微配置空间净空监督。轨迹坐标不使用 PointGoal 距离截断弧长，也不使用 `tanh` 饱和曲率。

## 2. 唯一张量合同

```text
depth                  float [B,4,1,126,224]
point_goal             float [B,2]
observation_to_current float [B,4,4]  (x,y,sin Δyaw,cos Δyaw)
observation_valid      bool  [B,4]

expert curve values    float [B,8]    (metric arc length + 7 curvature controls)
prediction path        float [B,64,2]
prediction heading     float [B,64]
prediction curvature   float [B,64]
```

二维坐标固定为机器人系 `x` 向前、`y` 向左、yaw 向左为正。训练只读取 `data/policy_dataset`；它是当前 CurveNav 自行生成并编译的唯一 prepared dataset，不在运行时混用 SanD、NavDP 或其他来源数据。`expert_navigation_geometry` 数据合同固定 Dingo USD 哈希、轮半径、轮距、完整 collision 包络和最大可跨越高度，旧的忽略家具数据或猜测机器人尺寸的数据会在加载前被拒绝。旧 checkpoint 因损失和模型合同不同同样直接拒绝，没有兼容分支。

生产图固定为 `D=384`、8 heads、42,847,312 参数：

```text
4 × calibrated metric depth
  └─ shared GroupNorm ResNet-18 stage-3 → 每帧 8×12=96 token
       ├─ 当前 96 token 保留
       └─ 三帧历史 288 token + null → 32 query 压缩

[PointGoal, current 96, history 32] = 129 token
  └─ 4 × joint self-attention condition encoder

8-D Gaussian-to-curve Flow state
  ├─ 平滑解码为 64 点连续曲率曲线
  ├─ 等弧长抽取 16 个 path token
  ├─ path-to-current-depth metric cross-attention
  └─ [8 control token, 16 path token]
       └─ 12 × (trajectory self-attention + condition cross-attention + SwiGLU)
            └─ 8-slot velocity → 8-step Heun → 唯一 64 点轨迹
```

## 3. 视觉、时序与目标融合

专家数据只建立一次障碍几何。机器人配置直接来自固定测评实际加载的 `dingo.usd`（SHA256 `43db9c…bcce20`），该资产与 NavDP、X-NavDP 仓库中的副本逐字节相同。启用的运动碰撞体给出 `0.167584539 m` 最小包围圆、基座坐标 `z∈[-0.044000001,0.117981499] m` 和 `0.161981500 m` 总高；轮半径为 `0.06125 m`、轮距为 `0.22616 m`。相机相对基座的 `0.28618 m` 前移和 `0.62532 m` 高度属于传感器外参，不再被误当机器人碰撞高度。

Habitat-Sim 将 stage 和全部静态物体 collision mesh 一同体素化；实测 footprint 半径负责水平配置空间膨胀，碰撞包络总高负责上方净空。统一的 `0.05 m` 最大可跨越高度低于真实轮半径，并决定低矮表面是否构成障碍。该 navmesh 投影出的最大连通二维网格是端点采样、A*、平滑、轨迹采样和最终审计的唯一几何来源，不再要求任意的“最大岛占比 80%”，不再先调用另一套 Habitat shortest-path，也不在渲染深度后过滤错误专家。

PointGoal 端点距离固定从所有 20 个场景共同支持的连续区间采样：near `[3,6) m`、middle `[6,8.5) m`、far `[8.5,10.5) m`，每场景分别生成 `5/10/10` 条。距离分布不按场景缩放，也不存在候选耗尽后的改档或重试；因此距离标签含义跨场景一致，同时不会要求小场景生成数学上不存在的 `[11,15) m` 安全端点对。

A* 只优化一个固定目标。若相邻网格步长为 `Δs`，其已膨胀障碍净空为 `c`，边代价为：

```text
ΔJ = Δs [1 + exp(-c / 0.167584539)]
```

第一项最小化长度，第二项以机器人半径为唯一尺度连续排斥贴障路线；可行域本身要求额外净空至少 `0.10 m`。网格折线仅在直连段仍满足净空且同一个积分代价不增加时才被化简，之后的 spline 平滑也必须在同一网格上保持净空。旧实现的独立 Habitat 路径预检、无目标一致性的贪心 shortcut、五组净空权重、曲率硬筛选、绕路比例和 Pareto 二次选择均已删除；曲率、转角和净空只作为数据审计统计，不决定隐藏分支。

共享单通道 ResNet-18 使用 GroupNorm，因此单样本输出不依赖 batch 或 DDP rank 的运行统计。每个 adaptive cell 保留最近的本体高度障碍；若没有障碍则保留最近可见表面。相机光轴深度为 `z`，像素为 `(u,v)`，下俯角为 `α`，相机前移为 `a`、高度为 `h`：

```text
x_o = z(u-cx)/fx
y_o = z(v-cy)/fy
p_body = (a + cos(α)z - sin(α)y_o,
          -x_o,
          h - cos(α)y_o - sin(α)z)
```

历史点用 `p_current=R(Δyaw)p_observation+t` 对齐到当前机器人系。历史仅补充已观测几何，由 32 个 learned query 压缩；`observation_to_current` 不作为独立状态 token，避免专家历史运动方向泄露未来路线。无效历史通过 attention mask 排除，额外 null token 使冷启动 memory 始终定义良好。

对每个视觉点 `p_i` 和目标单位方向 `ĝ`，输入显式包含：

```text
longitudinal_i = p_i · ĝ
lateral_i      = ĝ_x p_i,y - ĝ_y p_i,x
radius_i       = ||p_i||
goal_range, surface_valid, obstacle_valid
```

目标 token、当前 96 个空间 token 和 32 个历史 token 再通过四层联合 self-attention。与旧版只拼接未经上下文化 memory 不同，这一步允许当前障碍在生成开始前同时读取目标方向和历史遮挡信息，吸收 SanD 深度序列 Transformer 的有效设计。

## 4. 共享度量曲率流形

数据保存真实物理值 `q=[L,c₁,…,c₇]`：`L>0` 为米制总弧长，`c_i` 为 `m⁻¹` 曲率控制。Flow 状态位于无界坐标 `z∈R⁸`。长度使用无硬上限、数值稳定的 softplus 双射，曲率使用训练集统计量线性标准化：

```text
a = μL + σL z₀
L = softplus(a)
c_i = μκ + σκ z_i
κ(u) = Σ_i B_i,3(u)c_i

μL=2.9003251, σL=1.2550740
μκ=-0.01270257 m⁻¹, σκ=0.35367659 m⁻¹
```

这些统计量来自当前唯一训练集的物理专家控制。反变换为 `z₀=(softplus⁻¹(L)-μL)/σL`、`z_i=(c_i-μκ)/σκ`，因此正弧长与欧氏 Flow 空间一一对应；不存在 `L≤2||g||`、弧长截断、曲率 clip 或 `tanh` 饱和。PointGoal 只进入条件网络，不改变轨迹坐标定义。

三次 B-spline 保证曲率连续；它不再人为限制曲率幅值。轨迹由 Frenet 方程积分：

```text
θ(0)=0
θ(u)=L∫₀ᵘκ(v)dv
p(u)=L∫₀ᵘ[cos θ(v), sin θ(v)]dv
```

实现使用四倍密集弧长网格、梯形航向积分和 circular-chord 位置积分，再抽取 64 点。平滑性、正弧长和前向初始切向来自参数化本身，不依赖输出裁剪或推理后处理。专家投影和生产解码共用同一套曲率 basis 与积分实现。

## 5. 数学正确的条件 Flow Matching

对专家曲线坐标 `x₁`，训练定义：

```text
x₀ ~ N(0,I₈)
t = 0 with probability 1/9; otherwise t ~ Uniform([0,1])
x_t = (1-t)x₀ + t x₁
u* = d x_t/dt = x₁-x₀
L_CFM = E ||uθ(x_t,t,c)-u*||²₂
```

这是 Flow Matching 的 Optimal-Transport displacement interpolation。随机 `x₀` 使给定 `(x_t,t)` 不再能代数恢复专家标签；条件 `c` 在整个 `t<1` 区间都必须消除源不确定性。`1/9` 的训练样本显式落在 `t=0`，对应八步积分的九个时间节点之一，使第一次数值速度求值进入训练支持。

推理求解同一个 ODE `dx/dt=uθ(x,t,c)`，使用固定 8 步 Heun。单轨迹部署使用 seed `20260828` 生成一次标准高斯方向，再严格缩放到八维高斯典型半径 `||x(0)||₂=√8`；该 buffer 随 checkpoint 保存，每次闭环重规划完全确定。它属于训练源的典型集合，不再使用高维高斯中非典型的零向量。

训练还从当前速度场一次重建其预测终点：

```text
x̂₁ = x_t + (1-t)uθ(x_t,t,c)
P̂ = Decode(x̂₁)
```

对正确的直线 OT 条件路径，真实速度满足 `x₁=x_t+(1-t)u*`，所以该式不是额外生成器或第二阶段，而是直接检查当前速度预测最终会生成什么物理曲线。

## 6. 路径—障碍生成器

每次训练速度回归以及 Heun 的 predictor/corrector 求值，都会把当前 `x_t` 解码为实际路径。沿 64 点曲线固定抽取 16 个锚点，每个 path token 输入：

```text
[x/H, y/H, sinθ, cosθ, κ/σκ, arc_progress,
 (goal_x-x)/H, (goal_y-y)/H]
```

对路径点 `p_j` 和当前帧深度 cell `o_i`，度量注意力偏置由 learned MLP 处理：

```text
r_ji = [
  (o_i-p_j)/H,
  ||o_i-p_j||/H,
  (||o_i-p_j||-(0.167584539+0.10))/H,
  obstacle_valid_i
]

A_ji = softmax_i(q_j k_iᵀ/√d_h + MLP(r_ji))
```

`0.167584539 m` 是实测机器人 footprint 半径，`0.10 m` 是训练/几何合同中的额外净空。它们既作为生成器的度量输入，也定义训练期的软配置空间监督；不执行轨迹投影、硬拒绝或 fallback。由此网络能直接回答“当前这条中间曲线的第 j 段与哪个障碍相交或余量不足”。

八个 control token 与十六个 path token 拼成 24-token 轨迹序列，进入十二层 pre-norm Transformer。每层先做轨迹 self-attention，再对 129 个已上下文化条件 token cross-attention，最后做 SwiGLU；只从前八个 control token 读取八维速度。该结构吸收 NavDP 将显式带噪轨迹作为 decoder token 的关键优点，同时保留 CurveNav/SanD 紧凑曲线空间的平滑性和低输出维度。

## 7. 相对论文与官方源码

| 来源 | 吸收的有效设计 | CurveNav 的当前取舍与改进 |
|---|---|---|
| [SanD-Planner](https://arxiv.org/abs/2602.00923) | 从零训练的共享 ResNet-18、深度序列 Transformer、紧凑 B-spline 空间、少数据归纳偏置，以及 ESDF 净空评分 | 使用连续曲率 B-spline，专家与生产共用同一流形；把其安全几何转为训练期净空监督，不复制 16 候选和在线 ESDF 选择器。 |
| [NavDP](https://arxiv.org/abs/2505.08712) | `D=384` 深层 Transformer、显式带噪轨迹 token 与视觉 memory 的逐层交互；专家由 ESDF 安全路径生成 | 以 8 个曲率控制加 16 个可执行路径 token 取代 24 个自由 waypoint；HSSD 专家同样在完整静态场景配置空间规划，但当前不增加 critic。 |
| [X-NavDP](https://arxiv.org/abs/2607.28560) | 识别专家预训练在陷阱、长障碍绕行和时间一致性上的局限 | 当前不引入 GQRM、critic 或行为扰动；其后训练属于下一阶段，避免用价值头掩盖生成器基本错误。 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 从先验抽样、条件概率路径速度回归、OT 路径与 ODE 采样 | 删除会泄露标签的固定训练源；无界曲线坐标保证整个 ODE 中间态都可由光滑映射解码为合法曲线。 |
| [Motion Planning Diffusion](https://arxiv.org/abs/2308.01557) / [Safe Flow Matching](https://arxiv.org/abs/2504.08661) | 用 SDF 代价梯度或 barrier 在生成过程中增强安全性 | 当前不引入逐步引导、QP 或 CBF 求解；一次训练期 `x̂₁` 净空监督保留零额外推理成本。 |

## 8. 唯一损失、训练与验证

训练目标固定为：

```text
d_i = min_j ||P̂_i-o_j||₂
L_clear = mean_i [relu((0.267584539-d_i)/0.267584539)]²
L = L_CFM + L_clear
```

`o_j` 是当前深度中落入 Dingo 实际碰撞竖直区间的有效障碍点。具体地，地面在基座坐标中为 `-0.044000001 m`，所以越过 `0.05 m` 高低障碍后的下界是 `0.005999999 m`，上界为碰撞顶面 `0.117981499 m`；这与 Habitat 的“相对地面高度”定义严格对齐。无有效障碍的样本贡献零。`L_clear` 无量纲且位于 `[0,1]`，是对机器人圆形 footprint 加 `0.10 m` 裕量的软惩罚，不是曲率、长度或输出的硬约束。训练只额外执行一次 64×96 的 batch 距离计算和一次八维曲线解码；推理图、Heun 次数和输出完全不变。代码中仍不存在 candidate、critic、评价头、轨迹修补或第二生成链路。

训练固定 FP16 autocast、FP32 residual carrier、GPU 常驻 depth bank、异步 prefetch、fused AdamW、cosine schedule、EMA、静态 `torch.compile` 和 DDP。global batch 为 1024，每卡 batch 上限 256；1–8 卡通过精确 rank 分片保持同一全局样本流、更新次数和学习率。生产模型在 RTX 3090 上以 batch 256、真实 optimizer、compile 及 1.63 GiB 常驻深度库测试，峰值显存为 13.916 GiB。

必要验证固定为：

1. 标定反投影、历史刚体对齐、障碍高度选择、历史 mask 与 PointGoal 几何；
2. 物理值与无界 Flow 坐标往返、专家投影、正弧长、连续曲率和 PointGoal 弧长解耦；
3. 随机源 CFM 插值、`t=0` 覆盖、固定典型集部署源、`x̂₁` 重建、净空 loss 的掩码与有限梯度、完整前向、FP16 与 compile；
4. held-out 总体及高累计转向 ADE、预测/参考累计转向比、当前深度与目标打乱因果审计；
5. 固定场景闭环 10 回合 SR/SPL 和精简轨迹 trace。
