# CurveNav architecture

CurveNav 是一个单输出、PointGoal 条件的二维局部轨迹生成器。它只接收四帧标定深度、历史相对状态和当前机器人系 PointGoal；一次 MeanFlow 传输直接输出一条短程 B-spline 轨迹。它不包含候选采样、轨迹 critic、ESDF/地图补全、轨迹投影、推理期安全能量或后处理分支。

## 1. 坐标和物理合同

所有局部量采用 `x-forward, y-left`，PointGoal 为当前机器人系中任务终点的二维坐标。Dingo 碰撞体用半径 `0.167584539 m` 的圆形 C-space 膨胀；额外净空为

\[
m=0.10\;\mathrm m.
\]

规划视野和最大局部评测长度均为 `H=3.6 m`。源 C-space、离线评测和训练期路径查询共用 `0.025 m` 的物理采样间隔；它们的坐标变换、OOB 语义和机器人 footprint 均由 [`src/curvenav/physical.py`](src/curvenav/physical.py) 定义。

## 2. 数据合同

训练数据只保存模型真正需要的条件和专家曲线：

```text
depth_indices              uint32 [B,4]
point_goal                 float32[B,2]
observation_to_current     float32[B,4,4]  # dx, dy, sin(dyaw), cos(dyaw)
observation_valid          bool   [B,4]
curve_values               float32[B,8]
source_grid_index          int64  [B]
source_origin_xy           float32[B,2]
source_yaw_rad             float32[B]
```

最后三项只保留在数据集，用于 source C-space 专家证书和离线物理评测；它们绝不进入策略条件。准备阶段把每条专家 B-spline 以 `0.025 m` 在原始 `navigation_grid.npz` 上重新查询，要求全程在 world bounds 内且 signed clearance 不小于 `m`。因此 source C-space 是数据门控和独立裁判，不是策略输入或训练标签。

不再写入 `local_clearance_m`、碰撞反事实曲线或其有效位。它们对应的特权 map completion、BCE 和相对负轨迹损失已删除。

## 3. 可观测局部几何

`MetricDepthProjector` 对四帧深度逐帧反投影，并用 `observation_to_current` 的 SE(2) 变换对齐到当前机器人系。它构造固定 `64×64` 的局部 C-space field：

\[
F(q)=[d(q),\hat\nabla_xd(q),\hat\nabla_yd(q),o(q),b(q)].
\]

其中 `d` 是由已观测 body-height 障碍得到的 footprint-inflated signed clearance，`o` 是射线可观测性，`b=1[d\le0]`。`F` 是输入深度的确定函数；未知格不会被学习模块补成可通行或不可通行。距离变换为了在可见障碍附近连续查询而在整张有限栅格上有数值，但它在 `o=0` 时不是自由空间观测：进入卷积编码器和路径 token 的实际特征严格为

\[
[o d,\;o\hat\nabla_xd,\;o\hat\nabla_yd,\;o,\;o b].
\]

因此未知格不能借由正的外推 clearance 伪装成可通行空间；raw `d` 只在 `o` 掩码下用于连续训练期净空梯度和可观测性诊断。

这一区分是关键：`64×64` raw field 是部署端的局部观测，不是物理真值。物理真值仍是 native `0.05 m` source grid。

## 4. 视觉、时序与目标融合

共享的 ResNet-18 对所有四帧一次批量编码，得到每帧 `8×12` 个视觉 token。当前帧的 96 个 token 进入全局 scene memory；四帧共 384 个 token 连同其标定三维点和有效位，双线性 splat 到当前机器人系 `8×8` metric BEV memory。raw C-space 同时经 `64→32→16→8` 卷积编码并与该 BEV memory 相加。

因此历史帧既以确定的障碍 union 进入 raw field，也以可学习视觉特征进入同一度量 BEV；历史视觉不再被丢弃。三个过去时刻的 SE(2) 状态 token 提供因果运动信息。

当前视觉 token、状态 token 和 `8×8=64` C-space token 先经过四层**与目标无关**的 self-attention。随后才追加独立的、按 `H` 归一化的 PointGoal token。PointGoal 可以引导绕行方向，但不能改写测得的障碍证据。

## 5. 正则轨迹坐标

网络生成八维 Euclidean Flow 坐标：一维长度预激活和七维局部航向增量。解码为

\[
L=\operatorname{softplus}(\mu_L+\sigma_Lz_0),\qquad
\Delta\theta_k=\mu_k+\sigma_kz_k,
\]

随后累积成八个 clamped cubic B-spline heading control points，并按弧长积分得到 64 个路径点。该参数化保证正弧长、原点和初始前向切线，以及连续 heading/curvature；它不把长度硬绑定到目标距离，也不使用 `tanh` 曲率边界。

解码器有三次自细化。在后两次细化中，它沿当前估计曲线查询 raw C-space。`32` 个路径 anchor 覆盖 64 个输出点，间距约 `3.6/31=0.116 m`，与 `64×64` field 的 `7.2/63=0.114 m` 单元分辨率对齐；旧的 16-anchor 欠采样已删除。

## 6. 单步 MeanFlow 与训练期可见净空耦合

令专家标准化坐标为 `z`，训练源为 `e`，插值状态为

\[
x_t=(1-t)z+t e,qquad v^\star=e-z.
\]

网络同时预测瞬时速度和区间平均速度。训练使用 iMF 的 diagonal/interior 区间，以及固定部署边界 `(r,t)=(0,1)`；部署子集固定使用同一典型高斯源 `e*`。停止梯度的 JVP 形成平均速度的总时间导数，优化目标为瞬时和重参数平均速度对 `v*` 的等权平方误差。推理严格使用

\[
\hat z=e_\star-u_\theta(e_\star,0,1,c),\qquad
\hat\tau=D(\hat z),
\]

即一次平均速度传输，无 ODE 多步积分。

每个训练 global batch 的连续样本流按 `index mod 4` 分配 interval group，而不是在每张卡或每个 micro-batch 重新计数：group `0` 是部署边界、`1` 是 interior、`2/3` 是 diagonal。因此无论 1–8 张卡怎样切分，每个 global batch 都有精确 `1/4` 的部署样本，DDP 的路径净空项与全局目标一致。

仅在该精确部署子集上，对 `\hat\tau` 加入训练期 observed-clearance 项。令 `q_j` 是从起点开始、间隔不大于 `0.025 m` 且不重复短路径终点的 active 路径采样点，`d_j,o_j` 是从**同一个输入 raw field** 双线性查询的 clearance 和可观测性，`Q=145`：

\[
L_{\rm vis}=
\frac{1}{|\mathcal D|}\sum_{i\in\mathcal D}
\frac1Q\sum_j a_{ij}o_{ij}
\left[\max\left(0,\frac{m-d_{ij}}m\right)\right]^2,
\qquad
L=L_{\rm MF}+L_{\rm vis}.
\]

`a` 是执行弧长掩码，未知格的 `o=0`，所以 source map 不会泄漏为可微监督。对 `d<m`，位置梯度与 `+\nabla d` 同向，经过 B-spline 和 MeanFlow 直接推动生成轨迹远离已观测、已膨胀 C-obstacle。该项只在训练中建立“生成轨迹—自身输入几何”的因果联系；推理图完全不变。

## 7. 为什么删除 completion 和反事实负轨迹

旧链路用 source grid 栅格化得到全局 `local_clearance_m`，让一个 head 补全不可见区域，再用 source-unsafe `p−` 做相对轨迹边界。这两条监督都不能保证由输入深度判定。

在 source-cspace validation 上，2,247 条有效 `p−` 中有 35.43% 在真实碰撞段没有任何 raw observed-margin (`d<0.10m`) 证据；把它们全部称为“深度避障梯度”在数学上不成立。相反，1,024 条 source-safe 专家的 raw field 覆盖率为 89.31%，只有 0.364% 路径点（2.832% 轨迹）出现 observed `d<0.10m` 冲突。因此现在的软 observed-only 项保留局部传感器证据，避免把少量投影/分辨率差异变成硬约束。

## 8. 与 SanD、NavDP、X-NavDP 的关系

- SanD 的短深度序列、ResNet token 和 B-spline 轨迹参数化被保留；但 SanD 的实际安全性还依赖候选轨迹和 ESDF 选择，不能等同于单输出生成器本身已学会避障。
- NavDP 的 ESDF 负轨迹与 critic 属于特权评价/选择链路。CurveNav 不复制该分支：源 C-space 只做数据证书和评测，生成器只学习部署时可见的 raw C-space。
- X-NavDP 的 RL post-training 不在当前范围；本链路先保证模仿学习生成器、训练边界和物理评测一致。

相对这些方案，CurveNav 的改进不是增加第二个决策器，而是让唯一 MeanFlow 轨迹在与部署相同的输入 C-space 上获得连续、可微且不泄漏特权地图的避障梯度。

## 9. 效率和唯一链路

没有 completion decoder、BCE、`p−` 搜索、candidate batch 或 critic，因此主损失和数据准备都更短。四帧 ResNet 以 `[B·4]` 单次卷积执行；BEV splat 固定为 `8×8`；visible loss 只作用于四分之一部署样本的 `145×5` field gather，不额外执行 JVP 或推理。

训练、离线评测和部署共享同一 `CurveNavPolicy`、codec、raw C-space 和 CUDA precision contract。支持 BF16 的设备使用 BF16；V100 等设备使用同一代码路径下的 FP16 + GradScaler，checkpoint 保存对应 scaler state。不存在模型版本、fallback launcher 或旧 checkpoint 兼容分支。
