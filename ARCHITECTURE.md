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
  └─ shared GroupNorm ResNet-18 stage-3 ── 4×96 spatial/metric tokens ─┐
                                                                      ├─ 64 geometry queries
4× (x,y,sin Δyaw,cos Δyaw) ──────────────── 4 explicit state tokens ──┤
PointGoal direction + log range ─────────────────────── goal token ───┤
                                                                      ↓
                                      2× route-query cross-attention
                                      4× joint condition Transformer
                                      supervised local-route bottleneck
                                                                      ↓
three executed-past xy tokens + eight future bounded-curvature tokens
                                      8× past-future Flow Transformer
                                      adaRMS-Zero(time, route)
                                                                      ↓
                     8 antithetic joint samples, 8-step batched Heun
                     select by executed-past reconstruction consistency
                                                                      ↓
          PointGoal-scaled arc-length/curvature decoder → metric local path
```

训练和部署只组装这一张图。checkpoint 类型为 `curvenav_bounded_curvature_flow_policy`，旧模型在加载前被严格拒绝，不存在兼容分支。

## 2. 视觉几何、历史状态与路线融合

### 2.1 度量深度 memory

每帧由同一个单通道 ResNet-18 编码到 stride-16 stage-3 特征，再池化为 `8×12=96` 个 token。卷积使用 GroupNorm，表示不依赖单卡 batch 或 DDP rank 的运行统计。token 同时包含二维图像位置编码和标定相机反投影得到的度量平面点。

对光轴深度 `z` 和像素 `(u,v)`，相机下俯角为 `α`、相对机器人前移为 `a`：

```text
x_o = z(u-cx)/fx,
y_o = z(v-cy)/fy,
p_body = (a + cos(α)z - sin(α)y_o, -x_o).
```

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

## 4. Past-Future Rectified Flow Transformer

### 4.1 联合状态与 CFM

三帧已执行历史的当前系平移除以 1.35 m，形成 `h∈R^(3×2)`。未来曲线坐标为 `y∈R^(8×2)`。Flow 联合建模：

```text
x_1 = [h,y] ∈ R^(11×2).
```

无效历史以及七个曲率 token 的第二通道通过自由度 mask 从噪声、损失和 ODE 更新中同时移除。其余维度使用标准直线 conditional flow matching：

```text
z ~ N(0,I),  t ~ U(0,1),
x_t = (1-t)z + t x_1,
u_t = x_1-z,
L_flow = Σ mask·||v_θ(x_t,t,C)-u_t||² / Σ mask.
```

一步干净端点估计为：

```text
x_hat_1 = x_t + (1-t)v_θ(x_t,t,C).
```

直接解码 `x_hat_1` 的未来部分施加 metric path 和 tangent 监督，Flow 学到的不只是内禀坐标均方误差。

### 4.2 轨迹 Transformer

11 个状态 token 加 learned token-position embedding 和 past/future role embedding。八层 Flow block 均包含：

1. 11 token 双向 self-attention，使已执行历史和未来控制结构直接交换信息；
2. 对 70 个 condition token 的 cross-attention，保留局部几何的 token 级信息；
3. SwiGLU feed-forward；
4. 由 Fourier flow time 与 `c_route` 共同产生的 adaRMS-Zero shift、scale 和 residual gate。

adaRMS-Zero 的各分支 gate 零初始化，输出 velocity projection 也零初始化，使初始网络从稳定的零向量场开始学习。相较只在输入拼接一次时间/目标，逐层调制让路线和 Flow 时间控制每个残差更新；这吸收 DiT 的 adaptive-normalization 稳定性和 X-NavDP 的 FiLM 条件注入思想，但 geometry 仍通过 cross-attention 保持空间分辨率。

生产深度不是为了堆参数而设置：条件端四层负责 geometry/state/goal 路线推理，Flow 端八层负责不同噪声时间上的过去—未来联合恢复，两者职责不同且没有重复生成器。

### 4.3 生成器内部的历史自校验

部署使用八个固定、成对反号且逐维 RMS 为一的 Gaussian base，一次扩成 `[B×8,11,2]`，通过八步 Heun 同批积分。对第 `j` 个联合结果，以真实已执行历史计算：

```text
S_j = Σ valid·||h_hat_j-h||² / Σ valid.
j* = argmin_j S_j.
```

只解码 `j*` 对应的未来曲线。这里没有 learned score、碰撞概率或启发式总代价；同一个生成器同时重建 past 与 future，past 是推理时可验证的已发生事实。该设计把 Past-Token Prediction 的“用过去重建验证未来样本”原则迁移到局部导航。冷启动没有有效过去时，各候选分数同为零，固定顺序给出确定结果。

它比直接输入上一条预测计划更适合当前合同：真实执行历史不会传播上一周期的错误计划，prepared data 也确实拥有因果历史位姿；相反，当前数据没有“上一周期模型已提交计划”字段，用专家未来伪造 RTC/previous-plan token 会发生标签泄漏和训练—部署偏移。

## 5. 唯一训练目标

所有项都先无量纲化，再直接相加：

```text
L = L_flow + L_path + L_tangent + L_route.
```

- `L_flow`：上述 past-future masked velocity MSE。
- `L_path`：`path/H` 与 `reference/H` 的 Smooth-L1；沿等弧长进度使用 `0.25+exp(-4s)` 并归一到均值一，强调马上要执行的近端。
- `L_tangent`：有效相邻路径段的 `1-cos(Δp_hat,Δp*)`，使用相同近端权重。
- `L_route`：预测局部终点 `s_hat/H` 与专家参考终点的 Smooth-L1。

不另加平滑 loss 或后处理：连续曲率界和高阶路径平滑由弧长域参数化直接给出，路径和切向项负责实际 metric 几何。训练日志分别记录四项损失，避免总 loss 掩盖某个子任务失效。

## 6. 相对论文方案的取舍

| 来源 | 吸收的有效设计 | CurveNav 的针对性改进 |
|---|---|---|
| SanD | 四帧共享深度 backbone、空间 token、平滑低维轨迹先验 | 教师 B-spline 只做标签平滑；生产输出改为 PointGoal 标度的连续有界曲率曲线，并以真实 executed past 代替不可得的 previous-plan 标签 |
| NavDP | `D=384` actor、learned-query 视觉压缩、trajectory-token cross-attention | query 先保存目标无关 metric geometry，再由独立 route query 融合目标；无 ESDF 标签时不训练 critic |
| X-NavDP | 深层条件生成器、逐层 FiLM 思想、时序一致性的重要性 | 用 adaRMS-Zero 做 time-route 调制；用真实过去自校验，不让错误旧计划递归传播；当前阶段不做 GQRM/RL |
| LoGoPlanner | geometry/state/route 的任务专用 query | CurveNav 已有标定 metric depth 和真实位姿，不复制重型视频三维重建模型；路线瓶颈直接监督当前局部专家终点 |
| Past-Token Prediction | past/future 联合生成和 past reconstruction verification | 把 past action 改为当前系历史运动，候选选择只依赖可观测事实，不引入评价头 |
| Flow Matching / DiT | 直线 CFM、少步 ODE、adaptive normalization zero-init | Flow 作用于弧长/航向/曲率的可执行坐标空间，并以 token 级 metric geometry 作为 memory |
| NoMaD | 生成分布适合多模态局部行为 | 不照搬为 goal-agnostic 统一策略服务的 50% goal mask；CurveNav 始终严格 PointGoal 条件 |

主要来源：[SanD 论文](https://arxiv.org/abs/2602.00923) 与 [官方源码](https://github.com/WangJinCheng1998/sandplanner)、[NavDP 论文](https://arxiv.org/abs/2505.08712) 与 [官方源码](https://github.com/InternRobotics/NavDP)、[X-NavDP](https://arxiv.org/abs/2607.28560)、[LoGoPlanner](https://arxiv.org/abs/2512.19629)、[Past-Token Prediction](https://arxiv.org/abs/2505.09561)、[NoMaD](https://arxiv.org/abs/2310.07896)、[Flow Matching](https://arxiv.org/abs/2209.03003)、[DiT](https://arxiv.org/abs/2212.09748)。

## 7. 训练、推理和验证边界

训练只读取 `data/policy_dataset`。该目录当前仅包含本项目在固定 HSSD 资产上生成、按 Dingo 标定相机渲染的深度和专家轨迹，不混用论文作者的数据。四帧历史按行驶距离 `[-1.35,-0.90,-0.45,0] m` 取样；未来最多 24 个 `0.15 m` 专家点，近目标自然缩短。

唯一训练入口使用 FP16、GPU 常驻 depth bank、异步 prefetch、AdamW、cosine schedule、EMA 和静态 `torch.compile`；多卡时由同一入口启用 DDP。数学 batch 固定为 1024，显存 micro-batch 上限为每卡 128。对 world size `W`，每 rank 分配 `floor(1024/W)` 或 `ceil(1024/W)` 个互不重叠样本；局部 batch 均值乘 `W·B_r/1024` 后再经 DDP 求平均，严格得到全局 1024 样本均值。6 卡时分配为 `171×4 + 170×2`，每 rank 执行 `128+43/42` 两次前后向，只在末次同步梯度。这样 1–8 卡的每次 optimizer、schedule 与 EMA 更新都保持同一数学合同，不需要 padding、重复样本或改变学习率。唯一部署入口加载 EMA 权重并使用上述八候选 batched Heun 与 past consistency。没有训练专用生成器或部署 fallback。

必要验证分三层：

1. 张量/数学单测：相机反投影、历史 mask、geometry/route token、PointGoal 标度、零目标停止、连续曲率硬界、教师 B-spline、checkpoint 和部署接口；
2. 前向/梯度：训练 loss、全部参数梯度、确定性推理、静态编译图和有限值；
3. 重新训练后的 held-out/闭环：ADE、弧长、目标进展、曲率、延迟，以及固定协议 SR/SPL。

旧 checkpoint 的低闭环成绩可以证明旧链路失败，但不能单独证明新模块有效。当前结构必须从头训练；离线几何通过后再进入固定协议闭环，最终结论以 SR/SPL 为准。
