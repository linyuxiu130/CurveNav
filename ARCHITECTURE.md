# CurveNav 轨迹生成架构

本文是当前代码的唯一模型合同。CurveNav 是 PointGoal 条件的二维局部轨迹生成器：输入四帧度量深度、历史位姿和当前机器人系 PointGoal，输出一条由有界连续曲率定义的 64 点二维局部路径。当前阶段没有轨迹评价头、collision critic、Q head、候选代价加权或推理后轨迹修补。

## 1. 唯一张量合同与模型图

```text
depth                  float [B,4,1,126,224]  三帧历史 + 当前深度；1 表示 5 m
point_goal             float [B,2]            当前机器人系任务目标 (x,y)
observation_to_current float [B,4,4]           (x,y,sin Δyaw,cos Δyaw)
observation_valid      bool  [B,4]             历史补帧 mask；当前帧有效
episode flow source    float [B,8,2]           内部状态；每回合一次高斯采样

target controls        float [B,8,2]
target reference path  float [B,64,2]
prediction path        float [B,64,2]
prediction geometry    heading/curvature [B,64]
```

生产模型固定为 `D=384`、8 attention heads。完整数据流为：

```text
4× metric depth
  └─ shared GroupNorm ResNet-18 stage-3 ── 4×96 spatial/metric tokens ─┐
                                                                      ├─ 64 geometry queries
4× (x,y,sin Δyaw,cos Δyaw) ──────────────── 4 explicit state tokens ──┤
PointGoal direction + log range ─────────────────────── goal token ───┤
                                                                      ↓
                                      2× route-query cross-attention
                                      4× joint condition Transformer
                                      supervised local-route bottleneck
                                                                      ↓
                                      8 future bounded-curvature tokens
                                      fixed coordinate map y = 8x
                                      8× future Flow Transformer
                                      adaRMS-Zero(time, route)
                                                                      ↓
                 episode-persistent Gaussian source, 8-step Heun integration
                                                                      ↓
          PointGoal-scaled arc-length/curvature decoder → metric local path
```

训练和部署只组装这一张图。checkpoint 类型为 `curvenav_gaussian_flow_bounded_curvature_policy`，旧模型在加载前被严格拒绝，不存在兼容分支。

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

四帧一共 384 个视觉 token。64 个与目标无关的 learned geometry query cross-attend 这组 memory，将后续条件序列压缩到固定长度。这里刻意不在视觉压缩前注入 PointGoal：障碍、通道和可通行边界是目标无关的场景事实；让目标过早控制压缩会丢掉当前路线之外、但绕障时可能需要的几何信息。这保留 NavDP 的 query compression 效率，同时吸收 LoGoPlanner 将 geometry query 与 planning query 分工的做法。

### 2.2 显式状态和 PointGoal

`observation_to_current` 不只用于搬动深度点，也经过 MLP 形成四个 state token。平移除以 1.35 m 历史窗口尺度，旋转直接使用 `sin/cos`，不存在角度跳变。补帧使用 learned invalid-state token。

PointGoal 编码为方向和对数距离：

```text
r = ||g||,
feature(g) = [g/max(r,ε), log(1+min(r,25 m))/log(26)].
```

CurveNav 的任务合同始终提供 PointGoal，因此不采用 NoMaD/NavDP 为统一 goal-conditioned/goal-agnostic 策略而使用的 50% goal mask。对本任务机械照搬该 mask 会无依据地删除一半目标监督。

### 2.3 监督路线瓶颈

一个 learned route query 与 goal token 相加后，连续两次 cross-attend `[state, geometry]`；随后 `[route, goal, 4 state, 64 geometry]` 共 70 个 token 经过四层联合 condition Transformer。路线 latent `q_r` 预测当前 3.6 m 规划盘内的专家局部终点：

```text
a = MLP(q_r),
s_hat = 3.6 a / sqrt(1 + ||a||²).
```

这个光滑映射把任意二维向量映到开单位圆盘，零点处导数良好，也不需要预测后裁剪。预测的 `s_hat` 经 MLP 重新注入 route token：

```text
c_route = RMSNorm(q_r + MLP(s_hat / 3.6)).
```

Flow 同时 cross-attend 全部 70 个条件 token，并在每一层用 `c_route` 调制。训练与部署都使用预测路线，不把真实子目标喂给生成器，因此没有 teacher-forcing 落差。该 head 是生成条件的路线监督，不是对候选打分的评价头。

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

## 4. Future-only Gaussian Rectified Flow Transformer

### 4.1 高斯源条件 Flow Matching

三帧已执行历史已经通过 metric depth alignment 和四个显式 state token 进入条件序列。几何 decoder 消费的曲线坐标记为 `x∈R^(8×2)`。它们受硬几何尺度约束，当前验证集九个自由维度的标准差只有 `0.014–0.058`；直接把 `x` 当作 FP16 Flow 状态会使速度场长期处于 `10^-2` 量级，并让零初始化网络偏向无需修正的直线。训练和推理因此都使用唯一的固定无量纲坐标变换：

```text
y = 8x,       x = y/8.
```

尺度 8 把九个自由维度的典型标准差移到 `0.11–0.47`，同时保持零点、相对维度权重、可表示轨迹集合和曲率硬界完全不变。它对应 SanD/Diffusion Policy 对控制坐标做归一化的必要数值条件，但不读取验证统计、不保存数据集专用 normalizer，也不产生第二套 decoder。Flow 只生成 `y`，不把已知历史复制成生成目标。

局部绕障在部分可观测条件下具有真实多模态：同一障碍可能从左侧或右侧安全绕行。Dirac 零源 `y0=0` 的条件 MSE 只能学习条件均值；在对称障碍前，该均值会落到两种专家模式之间。CurveNav 因此使用标准高斯源直线条件 Flow：

```text
y_1 = 8x_1,
y_0 ~ N(0,I) on the nine free coordinates,
t ~ U(0,1),
y_t = (1-t)y_0 + t y_1,
u_t = d y_t/dt = y_1-y_0,
L_flow = Σ mask·||v_θ(y_t,t,C)-(y_1-y_0)||² / Σ mask.
```

七个曲率 token 的第二通道由结构 mask 从状态、损失和 ODE 更新中同时移除。一步干净端点估计为：

```text
y_hat_1 = y_t + (1-t)v_θ(y_t,t,C).
```

先用唯一逆变换 `x_hat_1=y_hat_1/8` 解码，再施加 metric path 和 tangent 监督，Flow 学到的不只是内禀坐标均方误差。训练和部署使用同一 masked isotropic Gaussian 源分布；固定为零的七个无效通道在采样、损失和 ODE 中始终保持为零。

### 4.2 轨迹 Transformer

8 个未来 token 加 learned token-position embedding。八层 Flow block 均包含：

1. 8 token 双向 self-attention，使弧长、初始航向和七个曲率控制直接交换信息；
2. 对 70 个 condition token 的 cross-attention，保留局部几何的 token 级信息；
3. SwiGLU feed-forward；
4. 由 Fourier flow time 与 `c_route` 共同产生的 adaRMS-Zero shift、scale 和 residual gate。

adaRMS-Zero 的各分支 gate 零初始化，使深层残差分支平滑打开；velocity projection 保留 PyTorch 线性层的方差缩放权重初始化并将 bias 置零。这样初始输出与标准高斯 velocity target 同为 O(1)，第一步即可给 route/cross-attention gate 传递梯度，而不是先等待全零输出头缓慢长大。相较只在输入拼接一次时间/目标，逐层调制让路线和 Flow 时间控制每个残差更新；这吸收 DiT 的 adaptive-normalization 稳定性和 X-NavDP 的 FiLM 条件注入思想，但 geometry 仍通过 cross-attention 保持空间分辨率。

生产深度不是为了堆参数而设置：条件端四层负责 geometry/state/goal 路线推理，Flow 端八层负责不同连续时间上的未来曲线修正，两者职责不同且没有重复生成器。

### 4.3 唯一推理轨迹

部署为每个环境维护一个 `[8,2]` masked Gaussian source。`navigator_reset` 时用固定生产 seed 的独立 generator 采样一次；同一 episode 的所有重规划周期复用该 latent，episode reset 后只替换对应环境的 latent。它既让生成器能选择一个绕障模式，又避免每帧重新采样导致左右模式振荡。随后从该 `y_0` 做八步 Heun 积分：

```text
y' = v_θ(y,t,C),
y_predict = y + Δt y',
y_next = y + Δt/2 [y' + v_θ(y_predict,t+Δt,C)].
```

输出只有一个 `[B,8,2]` 归一化 Flow 状态，经固定除 8 后送入曲线 decoder；不存在候选维、候选排序、历史重建分数或 oracle 选择。真实执行历史通过条件序列影响速度场，episode latent 只承担模式身份，不包含上一周期预测，因此不会递归传播旧计划误差。当前数据没有“上一周期模型已提交计划”字段，用专家未来伪造 previous-plan token 会产生标签泄漏。

## 5. 唯一训练目标

所有项都先无量纲化，再直接相加：

```text
L = L_flow + L_path + L_tangent + L_route.
```

- `L_flow`：上述 O(1) 归一化 Gaussian-source future masked velocity MSE。
- `L_path`：每个等弧长位置的欧氏误差 `||p_hat-p*||₂/H`；使用 `0.25+exp(-4s)` 并归一到均值一，强调马上要执行的近端。
- `L_tangent`：有效相邻路径段的 `1-cos(Δp_hat,Δp*)`，使用相同近端权重。
- `L_route`：预测局部终点与专家参考终点的欧氏误差 `||s_hat-s*||₂/H`。

不另加平滑 loss 或后处理：连续曲率界和高阶路径平滑由弧长域参数化直接给出，路径和切向项负责实际 metric 几何。训练日志分别记录四项损失，避免总 loss 掩盖某个子任务失效。

## 6. 相对论文方案的取舍

| 来源 | 吸收的有效设计 | CurveNav 的针对性改进 |
|---|---|---|
| SanD | 四帧共享深度 backbone、空间 token、平滑低维轨迹先验、归一化高斯生成、时序模式一致性 | 教师 B-spline 只做标签平滑；生产输出改为 PointGoal 标度的连续有界曲率曲线；用每回合单 latent 取代 ESDF 多候选筛选 |
| NavDP | `D=384` actor、learned-query 视觉压缩、trajectory-token cross-attention、高斯动作扩散 | query 先保存目标无关 metric geometry，再由独立 route query 融合目标；同一无量纲坐标用于训练和 ODE；无 ESDF 标签时不训练 critic |
| X-NavDP | 深层条件生成器、逐层 FiLM 思想、闭环时序一致性 | 用 adaRMS-Zero 做 time-route 调制；episode latent 固定模式而不递归输入旧预测；当前阶段不做 GQRM/RL |
| LoGoPlanner | geometry/state/route 的任务专用 query | CurveNav 已有标定 metric depth 和真实位姿，不复制重型视频三维重建模型；路线瓶颈直接监督当前局部专家终点 |
| Past-Token Prediction | 用可观测过去约束未来的思想 | 历史已经由 condition encoder 显式编码；不再联合生成过去，因为过去重建误差不能代表未来轨迹质量 |
| Flow Matching / DiT | Gaussian-to-data 直线 CFM、少步 ODE、adaptive normalization | 在与可执行坐标严格双射的 O(1) 空间学习多模态速度场；只保留残差 gate 零初始化，输出头从第一步传递梯度 |
| NoMaD | 生成分布适合多模态局部行为 | 不照搬为 goal-agnostic 统一策略服务的 50% goal mask；CurveNav 始终严格 PointGoal 条件 |

主要来源：[SanD 论文](https://arxiv.org/abs/2602.00923) 与 [官方源码](https://github.com/WangJinCheng1998/sandplanner)、[NavDP 论文](https://arxiv.org/abs/2505.08712) 与 [官方源码](https://github.com/InternRobotics/NavDP)、[X-NavDP](https://arxiv.org/abs/2607.28560)、[LoGoPlanner](https://arxiv.org/abs/2512.19629)、[Past-Token Prediction](https://arxiv.org/abs/2505.09561)、[NoMaD](https://arxiv.org/abs/2310.07896)、[Flow Matching](https://arxiv.org/abs/2209.03003)、[DiT](https://arxiv.org/abs/2212.09748)。

## 7. 训练、推理和验证边界

训练只读取 `data/policy_dataset`。该目录当前仅包含本项目在固定 HSSD 资产上生成、按 Dingo 标定相机渲染的深度和专家轨迹，不混用论文作者的数据。四帧历史按行驶距离 `[-1.35,-0.90,-0.45,0] m` 取样；未来最多 24 个 `0.15 m` 专家点，近目标自然缩短。

唯一训练入口使用 FP16、GPU 常驻 depth bank、异步 prefetch、AdamW、cosine schedule、EMA 和静态 `torch.compile`；多卡时由同一入口启用 DDP。数学 batch 固定为 1024，显存 micro-batch 上限为每卡 112，以适配 11 GiB 2080 Ti。对 world size `W`，每 rank 分配 `floor(1024/W)` 或 `ceil(1024/W)` 个互不重叠样本；局部 batch 均值乘 `W·B_r/1024` 后再经 DDP 求平均，严格得到全局 1024 样本均值。6 卡时分配为 `171×4 + 170×2`，每 rank 执行 `112+59/58` 两次前后向，只在末次同步梯度。这样 1–8 卡的每次 optimizer、schedule 与 EMA 更新都保持同一数学合同，不需要 padding、重复样本或改变学习率。2080 Ti 真实生产图的 50-step 稳态复测中，112 上限为 `0.2143–0.2154 s/rank-step`、峰值分配约 `6.30 GiB`；64 上限为 `0.2336 s/rank-step`，而单批 171 反而回落到 `0.2189 s/rank-step` 并触发一次 loss-scale 下调，因此生产值固定为实测吞吐最优且保留充足显存余量的 112。训练 source 使用 checkpoint 已保存的逐 rank CUDA RNG；部署 source 使用独立的固定 seed generator，并在 episode 内持久化。唯一部署入口加载 EMA 权重并使用上述单轨迹八步 Heun，没有部署 fallback。

在线部署使用 eager FP16 推理，不把分钟级编译成本放进短回合测评。`navigator_reset` 已知实际 batch size 后立即使用零观测和该批 episode latent 完成 CUDA kernel 初始化和同步；该步骤发生在 evaluator 的 episode 循环开始前。因此首个真实观测不会承担初始化时间，也不会让机器人在开局持续执行零动作。训练仍使用静态 `torch.compile`，因为 8000 个优化器 step 足以摊薄一次编译成本；1024 样本离线检查同样使用 eager，避免编译时间超过实际评估计算。

必要验证分三层：

1. 张量/数学单测：相机反投影、历史 mask、geometry/route token、PointGoal 标度、零目标停止、连续曲率硬界、教师 B-spline、checkpoint 和部署接口；
2. 前向/梯度：训练 loss、全部参数梯度、固定 latent 的可复现推理、静态编译图和有限值；
3. 重新训练后的 held-out/闭环：ADE、弧长、目标进展、曲率、延迟，以及固定协议 SR/SPL。

旧 checkpoint 的低闭环成绩可以证明旧链路失败，但不能单独证明新模块有效。当前结构必须从头训练；离线几何通过后再进入固定协议闭环，最终结论以 SR/SPL 为准。
