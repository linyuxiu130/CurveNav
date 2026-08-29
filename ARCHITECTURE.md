# CurveNav 模型架构

本文是当前代码的唯一模型合同。CurveNav 读取四帧标定深度、对应的历史观测到当前帧 SE(2) 变换和机器人系 PointGoal，以一次 improved MeanFlow 输运生成一条二维短程轨迹。模型没有候选集、评价头、ODE solver、硬长度/曲率裁剪、推理碰撞投影、旧模型兼容或 fallback。

## 1. 问题判断与唯一设计

旧模型的主要问题不是 Transformer 不够大，而是几何变量和场景交互不适合一步 Flow：

1. 七个累计航向控制量高度相关，训练集协方差条件数约为 `315`；各局部航向增量的条件数约为 `2.86`。同一条曲线流形用差分坐标表示后，Flow 不再重复运输累计误差。
2. 旧历史压缩器先把三帧度量几何压成 32 个潜变量，再要求路径 token 从潜变量中重新发现障碍位置。它丢失了避障最需要的连续空间对应关系。
3. 稀疏 cell 最近点和全局最近障碍 penalty 不能表示机器人 footprint、可见自由空间或整条路径的最坏风险。
4. PointGoal 距离曾被硬截断，远目标失去距离顺序；视觉 backbone 重复计算四帧，而短时历史真正需要的是已经配准的几何与运动状态。
5. 深度可见性曾把归一化深度同米制阈值比较，令无回波区域错误变成已观测空间。该单位错误已从投影根源消除。
6. 一步生成器曾只在解码后的 Flow 状态路径上查询配置空间。部署的状态恒为同一个典型高斯 latent，因此显式安全输入只覆盖固定 `1.75 m` 左弯附近的 `38/4096=0.93%` 栅格；修改其余 99.07% 场值不能改变网络输出。这使最终轨迹附近的障碍只能由当前图像隐式猜测，是可见绕行失败的结构根因。

唯一方案是：差分航向曲线坐标、当前帧 SanD 风格视觉特征、四帧配准的机器人配置空间场、完整配置空间 token、显式历史 SE(2) token，以及部署对齐的一步 improved MeanFlow。当前不增加 critic、多候选、RL 后训练或长期地图。

## 2. 张量与模块合同

```text
depth                  float [B,4,1,126,224]
point_goal             float [B,2]
observation_to_current float [B,4,4]   # x,y,sin Δyaw,cos Δyaw
observation_valid      bool  [B,4]

expert curve values    float [B,8]     # L, Δθ1,...,Δθ7
mean-flow coordinates  float [B,8]     # standardized log L and Δθ
prediction path        float [B,64,2]
```

生产模型维度固定为 `D=384`、40,394,440 个可训练参数：

```text
current depth
  └─ GroupNorm ResNet-18 stage3 + 8×12 pooling
  └─ calibrated current metric geometry
       └─ 96 current visual tokens

four aligned depths
  └─ slope-aware body-obstacle extraction
  └─ exact 64×64 Euclidean distance transform
       └─ [clearance,gx,gy,observed,forbidden]
       └─ 3-stage strided metric CNN
            └─ 64 complete configuration-space tokens

PointGoal token + 96 visual tokens + 3 historical SE(2) tokens [100 queries]
  └─ cross-attend all 64 configuration-space tokens once
  └─ 4 condition Transformer blocks
       └─ condition memory [B,100,384]

8 curve tokens + 16 decoded path/goal tokens
  └─ 12 self-attention + condition cross-attention + SwiGLU blocks
       └─ 8-D average velocity

fixed typical Gaussian latent
  └─ one average-velocity evaluation
       └─ one 64-point path
```

模块职责：

- `encoders/depth.py`：只对当前图运行视觉 backbone，并融合当前帧标定几何。
- `encoders/geometry.py`：四帧反投影、坡度分类、刚体配准和配置空间场。
- `encoders/configuration.py`：把完整配置空间场编码为带二维度量位置的全局安全 token。
- `conditioning/transformer.py`、`conditioning/motion.py`：目标、当前视觉和因果历史状态。
- `trajectory/heading.py`：训练与推理共用的八维正则曲线双射。
- `models/decoder.py`、`models/blocks.py`：路径/目标交互和读取完整条件记忆的平均速度网络。
- `models/policy.py`：唯一 MeanFlow 恒等式、损失和一步采样。
- `models/safety.py`：配置空间场查询与路径最坏风险；不修改推理轨迹。

## 3. 正弧长差分航向曲线

令 `L>0`，八个 clamped cubic B-spline 航向控制为 `θ0,...,θ7`，其中 `θ0=0`。网络的七个物理转向值是局部增量：

```text
θ0=0
θi=Σ_{k=1}^i Δθk
θ(u)=Σ_i B_i(u)θi
p(u)=L∫₀ᵘ[cos θ(v), sin θ(v)]dv
κ(u)=(1/L)dθ/du
```

所以任意有限网络输出都满足 `||dp/ds||=1`、起点为原点、初始方向向前；`θ∈C²`，因此 `p∈C³`、曲率连续。64 点之间用四倍过采样和线性航向圆弧弦公式积分。没有最大规划长度、`L≤2d_goal`、最大曲率、`tanh` 或 clip。

Flow 的无界欧氏坐标为：

```text
zL=(log L-μL)/σL
zi=(Δθi-μi)/σi
```

`log L` 保证长度严格为正；差分是从航向控制到局部转向的满秩线性双射，不改变 B-spline 曲线族。专家等弧长重采样后先拟合累计航向，再取控制差分。训练集统计写入唯一配置与数据 manifest。

## 4. 标定视觉与配置空间场

四帧深度使用同一 Dingo 相机标定反投影到机器人系。每个像素参与相邻四个图像三角面；重力方向法向坡度不超过 `45°` 的表面视为可通行地形。其余落在机器人碰撞高度带内的点是 body obstacle。所有有效历史点使用同一个刚体合同

```text
p_current = t_observation_to_current
          + R(yaw_observation-yaw_current) p_observation
```

对齐到当前机器人系。训练的 Habitat XZ 角度是右转为正，因此在写入该合同时先转换成左转为正；部署的 Isaac XY yaw 已是左转为正。两端最终张量语义完全相同。

局部场覆盖 `[-3.6,3.6]²` 的 `64×64` 网格。对栅格化障碍集合 `O` 使用可分离的精确欧氏距离变换：

```text
d(q)=min_{o∈O} ||q-o||₂
c(q)=d(q)-r_robot
forbidden(q)=[c(q)≤0]
```

其中 `r_robot=0.167584539 m`。场的五个通道是：footprint signed clearance、其单位梯度 `gx,gy`、由相机到有效深度表面的射线覆盖 `observed`、以及 `forbidden`。每条最大平面长度小于 `6.3 m` 的标定射线使用 `64` 个点栅格化，相邻点间距小于 `64×64` 场的 `7.2/63 m` 单元宽度。障碍一旦被观测，其 `c(q)≤0.10 m` 的机器人包络与安全裕度影响域也直接属于已观测已知区域；不能只把障碍点中心标成 observed。前四帧对障碍和可见区域取静态并集，因而过去看见但当前遮挡的近程障碍仍存在；无长期地图。

视觉 ResNet 只处理当前帧，避免四倍重复卷积。当前 `8×12` 特征同每个 cell 的 metric XYZ、深度、表面有效位和障碍位融合。三个过去位姿另以 `(x/h,y/h,sin Δyaw,cos Δyaw)` 编码；无效历史使用一个学习 null token。PointGoal 使用方向和无截断 `log1p(distance/3.6)`，因此任意有限距离保持顺序。

## 5. 路径条件生成器

完整 `64×64×5` 配置空间先把 clearance 除以 `3.6 m`，再经过三个 `3×3,stride=2` 卷积得到 `8×8` token；每个 token 加入覆盖 `[-3.6,3.6]²` 的二维正弦度量位置。PointGoal、96 个当前视觉 token 和三个历史状态 token 作为 100 个 query，对全部 64 个配置 token 做一次 cross-attention，再经过四层联合条件 Transformer。于是任何配置空间单元都能在一步轨迹生成之前影响条件记忆。

每次平均速度求值把八维状态解码为物理路径，并取 16 个固定弧进度锚点。每个路径 token 只包含：

```text
position/3.6, sin heading, cos heading, arc progress,
(PointGoal-position)/3.6
```

路径 token 与八个 curve token 一起通过 12 层 Transformer，并在每层对 100 个已经融合全局安全几何的条件 token 做 cross-attention。删除旧的 Flow-state 路径场查询至关重要：训练中的随机 `z_t` 和部署的固定高斯源都不是最终轨迹，只查询它们会造成安全信息与输出位置错位。完整场只在条件侧编码一次，也避免在 MeanFlow JVP 内重复扩大 12 层 memory。

## 6. Boundary-complete improved MeanFlow

采用数据端 `t=0`、高斯端 `t=1`。专家标准坐标为 `x`，源为 `e~N(0,I)`：

```text
z_t=(1-t)x+t e
v_c=e-x
```

网络 `uθ(z,r,t,c)` 表示区间 `[r,t]` 的平均速度，瞬时速度由退化区间定义：

```text
vθ(z,t,c)=uθ(z,t,t,c)
```

部署使用的数据锚定区间为 `[0,t]`。沿概率路径的总导数通过精确 forward-mode JVP 计算：

```text
D_tuθ = JVP(uθ; vθ,0,1)
Vθ = uθ(z_t,0,t,c)+t·stopgrad(D_tuθ)
```

由平均流恒等式 `v=u+(t-r)D_tu`，训练损失是：

```text
L_MF = 1/2 E[||vθ-v_c||² + ||Vθ-v_c||²]
```

每个 batch 先显式采样一次 `e~N(0,I)`，并用闭区间等距 collocation 覆盖 `[0,1]`。该源属于当前优化器 batch；若 FP16 overflow，重试复用同一个源，不重新抽样，也不复制/恢复整份 CUDA RNG 状态。不使用固定源训练、时间端点概率、课程开关或有限差分。推理选择训练高斯典型集内范数为 `√8` 的固定 latent：

```text
x_hat=e* - uθ(e*,0,1,c)
P_hat=Decode(x_hat)
```

这是严格 1-NFE，不存在 Euler、Heun、midpoint 或 `flow_steps`。

## 7. 生成路径安全目标

训练时同样从当前平均流估计数据端：

```text
x_hat_t=z_t-t·uθ(z_t,0,t,c)
P_hat_t=Decode(x_hat_t)
```

对 64 个等弧长路径点查询配置空间场，只在已观测位置计算 footprint 外的 `0.10 m` 软裕度，并取整条路径的最大违例：

```text
vj=[relu((0.10-c(pj))/0.10)]²
L_safe=mean_batch max_j (observed(pj)·vj)
L=L_MF+L_safe
```

取最大值针对任何单点碰撞都不能被长安全段平均稀释；配置空间距离已经减去机器人半径，因此阈值只剩额外净空。该项不裁剪、不重规划、不拒绝模型输出，也不对不可见空间声称安全保证。

## 8. 相对公开工作的依据

| 来源 | 吸收 | CurveNav 的改进/取舍 |
|---|---|---|
| [SanD](https://arxiv.org/abs/2602.00923) | ResNet 视觉、小样本轨迹先验、cubic spline、配置空间净空 | 保留解析平滑曲线，但改用严格正弧长和差分航向；完整配置空间在生成前进入唯一生成器，不复制候选 evaluator。 |
| [NavDP](https://arxiv.org/abs/2505.08712) | 轨迹 token 与视觉 memory 的深层交互、历史条件 | 保留深交互；以标定连续配置空间场替代要求 latent token 隐式恢复的碰撞几何。 |
| [X-NavDP](https://arxiv.org/abs/2607.28560) | 后训练用于恢复和分布外行为 | 当前先验证单生成器；不以 critic/RL 掩盖生成器的监督与几何错误。 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 合法随机高斯源和条件概率路径 | 禁止零源/固定源训练；确定性只在推理时选择典型 latent。 |
| [Improved MeanFlow](https://arxiv.org/abs/2512.02012) | 平均速度和 JVP 重参数化 | 同时监督瞬时边界与部署区间，单步训练/推理公式严格对齐。 |
| [Riemannian Flow Matching Policy](https://arxiv.org/abs/2412.10855) | 动作空间几何应进入 Flow | 先解析删除零切向奇点，再在标准化、良态的欧氏差分坐标中训练。 |

## 9. 数据、效率与验证合同

训练数据只来自 CurveNav 按 benchmark Dingo 配置生成的 HSSD 专家路线，不读取 SanD/NavDP 数据。当前 canonical dataset 为训练 `25,928`、验证 `6,087` 条；深度 bank 在编译时硬链接，不重复复制图像。

训练固定 global batch `1024`、每卡 micro-batch 上限 `384`、最大学习率 `2e-4`、`40` step/epoch、`200` epoch，共 `8,000` 次优化器更新。1--8 卡使用同一 DDP batch 分配；各 rank 份额最多差一个样本，并均衡拆成 micro-batch。AMP 为 FP16，MeanFlow JVP 使用数学 SDPA，EMA 与 optimizer/scheduler/RNG 都进入唯一 checkpoint。完整 MeanFlow/JVP 保持 eager；训练运行时只编译 JVP 外固定形状的 ResNet-18、度量几何投影和配置空间 CNN，分别融合视觉卷积/归一化与静态几何小算子。它们仍是同一组模块和参数，state dict、损失、梯度与推理数学图不变，没有第二训练实现。溢出结果直接读取 Accelerate 已记录的 `step_was_skipped`，正常 step 不再用 `GradScaler.get_scale()` 和 Python CUDA 布尔值造成三次额外同步。

当前必要验证：

- `compileall`、`git diff --check`；
- 90 个数学、数据、前向、梯度和合同测试；
- 生产 40,394,440 参数图的 FP16 前向、精确 JVP 与反向检查；
- 6,087 条验证专家的四帧配置空间审计：当前帧单独识别专家 footprint collision `0` 条、裕度违例 `89` 条、直线裕度违例 `787` 条；四帧静态融合后分别为 `9`、`143`、`1,325` 条。9 条碰撞均可由具体单独历史帧复现，位姿为正常的约 `0.45/0.89/1.33 m` 后向平移，不是跨帧符号或偶然 observed 拼接错误；
- 三张 V100S 按 `342/341/341` 分片，每 rank 单次前后向；编译 JVP 外静态子图后实测约 `26.7 GiB/GPU`，缓存命中的 step 40--120 稳定为 `1,890--1,915 samples/s`。相对同一全局场图未编译的 `1,745--1,755 samples/s` 提高约 `8.3--9.7%`；首次 kernel 编译约一分钟，只发生在新硬件/新图首次启动，8,000-step 正式训练能摊薄该成本；
- 当前图在 V100S、batch 1、FP16、预热后的纯 `policy.sample()` CUDA 延迟为 `33.89 ms`；包含深度重采样、四帧上下文、张量传输与路径回传的 runtime step 为 `40.18 ms`。4090 空闲态延迟必须在本轮训练结束后实测，不用训练占卡时的争用数据代替；
- 新模型最终 200 epoch 墙钟时间和离线指标必须由本次训练实测，不沿用旧 checkpoint。

## 10. 正确性边界

代码保证：标定与坐标一致；历史变换同时用于障碍配准和因果状态；`L>0`；起点为原点；初始切向前向；路径正则且曲率连续；Flow 训练源合法；时间端点属于训练域；MeanFlow JVP 符号与一步推理一致；安全损失作用于实际生成曲线。

代码不保证：有限数据必然复现专家；深度可见空间等价完整地图；无长期记忆时一定走出迷宫或死胡同；软损失等价控制屏障函数；离线 ADE 必然转化为闭环成功率。这些必须由严格离线分层和固定协议闭环测评验证。
