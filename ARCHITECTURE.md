# CurveNav 模型架构

本文是当前代码的唯一模型合同。CurveNav 读取四帧标定深度、对应的历史观测到当前帧 SE(2) 变换和机器人系 PointGoal，以一次 improved MeanFlow 输运生成一条二维短程轨迹。模型没有候选集、评价头、ODE solver、硬长度/曲率裁剪、推理碰撞投影、旧模型兼容或 fallback。

## 1. 问题判断与唯一设计

旧模型的主要问题不是 Transformer 不够大，而是几何变量和场景交互不适合一步 Flow：

1. 七个累计航向控制量高度相关，训练集协方差条件数约为 `315`；各局部航向增量的条件数约为 `2.86`。同一条曲线流形用差分坐标表示后，Flow 不再重复运输累计误差。
2. 旧历史压缩器先把三帧度量几何压成 32 个潜变量，再要求路径 token 从潜变量中重新发现障碍位置。它丢失了避障最需要的连续空间对应关系。
3. 稀疏 cell 最近点和全局最近障碍 penalty 不能表示机器人 footprint、可见自由空间或整条路径的最坏风险。
4. PointGoal 距离曾被硬截断，远目标失去距离顺序；视觉 backbone 重复计算四帧，而短时历史真正需要的是已经配准的几何与运动状态。
5. 深度可见性曾把归一化深度同米制阈值比较，令无回波区域错误变成已观测空间。该单位错误已从投影根源消除。
6. 把完整配置空间先压成 `8×8` token，再以它们更新视觉/目标 query，随后丢弃配置 token，仍然没有建立“输出路径点—障碍场位置”的直接对应。100 回合闭环中 CurveNav 只有 `32%` SR；失败回合仍给出约 `0.41 m/s` 前进命令，但实际速度约 `0.041 m/s` 且 87% 时间近乎静止。离线分支消融中，完整模型碰撞率为 `1.54%`，只保留当前帧视觉为 `4.70%`，只保留显式四帧配置空间却为 `15.67%`，与全空输入的 `16.31%` 几乎相同。这证明配置空间计算本身存在，但它在间接融合中没有成为可独立使用的规划变量。

唯一方案是：差分航向曲线坐标、当前帧 SanD 风格视觉特征、四帧配准的机器人配置空间场、保留到解码端的完整配置 token、显式历史 SE(2) token，以及在一次 improved MeanFlow 函数调用内部进行粗到细路径相对几何查询。当前不增加 critic、多候选、RL 后训练或长期地图。

## 2. 张量与模块合同

```text
depth                  float [B,4,1,126,224]
point_goal             float [B,2]
observation_to_current float [B,4,4]   # x,y,sin Δyaw,cos Δyaw
observation_valid      bool  [B,4]

expert curve values    float [B,8]     # L, Δθ1,...,Δθ7
mean-flow coordinates  float [B,8]     # standardized softplus pre-activation and Δθ
prediction path        float [B,64,2]
```

生产模型维度固定为 `D=384`、39,804,232 个可训练参数：

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

PointGoal + 96 visual + 3 historical SE(2) + 64 configuration tokens
  └─ 4 joint condition Transformer blocks
       └─ heterogeneous condition memory [B,164,384]

8 curve tokens
  └─ decoder blocks 1--4 ── shared readout ── coarse data-end curve
       └─ 16 path anchors query exact 64×64×5 field
  └─ decoder blocks 5--8 ── shared readout ── refined data-end curve
       └─ 16 path anchors query exact 64×64×5 field
  └─ decoder blocks 9--12 ── shared readout ── 8-D average velocity

fixed typical Gaussian latent
  └─ one average-velocity evaluation
       └─ one 64-point path
```

模块职责：

- `encoders/depth.py`：只对当前图运行视觉 backbone，并融合当前帧标定几何。
- `encoders/geometry.py`：四帧反投影、坡度分类、刚体配准和配置空间场。
- `encoders/configuration.py`：把完整配置空间场编码为带二维度量位置的全局安全 token。
- `conditioning/transformer.py`、`conditioning/motion.py`：目标、当前视觉、配置空间和因果历史状态的联合 token memory。
- `trajectory/heading.py`：训练与推理共用的八维正则曲线双射。
- `models/decoder.py`、`models/blocks.py`：一次函数调用内的共享-readout平均速度细化与路径相对配置空间查询。
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

令 `a=softplus⁻¹(L)=L+log(-expm1(-L))`。Flow 的无界欧氏坐标为：

```text
zL=(a-μL)/σL
zi=(Δθi-μi)/σi
L=softplus(μL+σL zL)
```

`softplus` 是从实数到正数的光滑严格单调双射，保证长度严格为正；它在大正输入处线性增长且导数始终小于 1，因此训练会遇到的大幅有限 Flow 中间状态不会再像指数坐标一样上溢。逆式使用 `expm1` 在短轨迹处保持数值精度。这不是长度裁剪：`L` 仍无上界，且该变换在整个数学定义域可逆。差分是从航向控制到局部转向的满秩线性双射，不改变 B-spline 曲线族。专家等弧长重采样后先拟合累计航向，再取控制差分。训练数据只保存米制弧长与弧度增量；坐标统计属于模型合同，只写入配置和 checkpoint，不污染物理数据合同。

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

## 5. 路径相对配置空间 MeanFlow

完整 `64×64×5` 配置空间先把 clearance 除以 `3.6 m`，再经过三个 `3×3,stride=2` 卷积得到带二维度量位置的 `8×8` token。它们不再被提前折叠并丢弃，而是与 PointGoal、96 个当前视觉 token 和三个历史状态 token 直接拼成 164 个一等 token，共同通过四层 condition Transformer；decoder 每层都能读取原始空间语义仍然存在的 memory。

12 层 decoder 分为三个连续的四层阶段，三个阶段使用同一个归一化和八维速度 readout，不存在独立中间头。第一阶段只根据 Flow 控制 token 和完整条件 memory 形成粗平均速度 `u¹`。对数据锚定时刻 `t`，其粗数据端估计为：

```text
x_hat¹=z_t-t·u¹
P_hat¹=Decode(x_hat¹)
```

从 `P_hat¹` 取 16 个固定弧进度锚点，直接在原始 `64×64×5` 场上做连续双线性查询。送入第二阶段的每个路径 token 为：

```text
position/3.6, sin heading, cos heading, arc progress,
(PointGoal-position)/3.6,
clearance/3.6, gradient_x, gradient_y, observed, forbidden
```

第八层再用同一个 readout 得到 `u²`，重建 `P_hat²` 并重复一次精确场查询；最后四层输出 `u³`。因此后续层修正的依据始终是当前网络实际准备生成的曲线，而不是独立 Gaussian `z_t` 解码出的随机曲线。三个阶段只是一个可微函数 `uθ` 内部的深度计算，MeanFlow 的 NFE 仍严格为 1；JVP 会穿过曲线解码和连续场查询，不存在训练/推理之外的投影、优化器或评价链。

中间轨迹不能是仅用来选择下一次几何查询位置的自由 latent：否则它会经曲线解码和场查询进入 JVP，形成无监督的高增益反馈。因此 `u¹,u²,u³` 都使用下节同一 Improved MeanFlow 恒等式监督，损失在三个阶段和八个欧氏坐标上直接取均值，没有人工阶段权重。概率路径的 JVP 切向仍唯一使用最终瞬时速度 `u³(z_t,t,t)`；推理只取最终平均速度 `u³(e,0,1)`。这是同一 readout 的深层监督，不是候选轨迹、评价头或第二推理链。

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
L_MF = 1/6 Σ_{k=1}^3 E[||vθᵏ-v_c||² + ||Vθᵏ-v_c||²]
```

每个 batch 先显式采样一次 `e~N(0,I)`，并用闭区间等距 collocation 覆盖 `[0,1]`。该源只属于当前优化器 batch；BF16 更新只执行一次，不重抽 source、不跳过 batch，也不复制/恢复整份 CUDA RNG 状态。不使用固定源训练、时间端点概率、课程开关或有限差分。推理选择训练高斯典型集内范数为 `√8` 的固定 latent：

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

对 64 个等弧长路径点查询配置空间场，只在已观测位置计算 footprint 外的 `0.10 m` 软裕度。硬 `max` 只把梯度传给一个最坏采样点，不足以训练整段扫掠路径；当前使用温度 `τ=0.10` 的稳定 smooth maximum：

```text
vj=observed(pj)·[relu((0.10-c(pj))/0.10)]²
L_safe=mean_batch τ[logsumexp_j(vj/τ)-log 64]
L=L_MF+L_safe
```

该式在全安全时严格为零，逼近最坏风险，同时让所有近危险点获得梯度；配置空间距离已经减去机器人半径，因此阈值只剩额外净空。该项不裁剪、不重规划、不拒绝模型输出，也不对不可见空间声称安全保证。

## 8. 相对公开工作的依据

| 来源 | 吸收 | CurveNav 的改进/取舍 |
|---|---|---|
| [SanD](https://arxiv.org/abs/2602.00923) | ResNet 视觉、小样本轨迹先验、cubic spline、配置空间净空 | 保留解析平滑曲线，但改用 softplus 正弧长和差分航向；完整配置空间在生成前进入唯一生成器，不复制候选 evaluator。 |
| [NavDP](https://arxiv.org/abs/2505.08712) | 轨迹 token 与视觉 memory 的深层交互、历史条件 | 保留深交互；以标定连续配置空间场替代要求 latent token 隐式恢复的碰撞几何。 |
| [X-NavDP](https://arxiv.org/abs/2607.28560) | 后训练用于恢复和分布外行为 | 当前先验证单生成器；不以 critic/RL 掩盖生成器的监督与几何错误。 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 合法随机高斯源和条件概率路径 | 禁止零源/固定源训练；确定性只在推理时选择典型 latent。 |
| [Improved MeanFlow](https://arxiv.org/abs/2512.02012) | 平均速度和 JVP 重参数化 | 同时监督瞬时边界与部署区间；只把 stop-gradient JVP 置于 FP32，普通主值与部署统一为 BF16。 |
| [Riemannian Flow Matching Policy](https://arxiv.org/abs/2412.10855) | 动作空间几何应进入 Flow | 先解析删除零切向奇点，再在标准化、良态的欧氏差分坐标中训练。 |

## 9. 数据、效率与验证合同

训练数据只来自 CurveNav 按 benchmark Dingo 配置生成的 HSSD 专家路线，不读取 SanD/NavDP 数据。当前 canonical dataset 为训练 `25,928`、验证 `6,087` 条；深度 bank 在编译时硬链接，不重复复制图像。

训练固定 global batch `1024`、每卡 micro-batch 上限 `342`、最大学习率 `2e-4`、`40` step/epoch、`200` epoch，共 `8,000` 次优化器更新。1--8 卡使用同一 DDP batch 分配；各 rank 份额最多差一个样本，并均衡拆成 micro-batch。`342` 是三卡 rank0 的实际最大值：24 GiB RTX 4090 上完整 `342` 样本前后向实测峰值约 `18.25 GiB`，因此三卡的 `342/341/341` 和四卡的 `256×4` 都只需一个 micro-batch。视觉、条件编码、瞬时边界主值和部署区间平均速度统一使用 BF16 Tensor Core 路径；只有已经 stop-gradient、不会参与反向图的精确 forward-mode JVP 使用 FP32 与数学 SDPA，因为十二层 Jacobian 连乘的动态范围不能可靠地放进 16 bit。JVP 的 primal 返回值被丢弃，可训练的平均速度另做一次 BF16 前向；这与 Improved MeanFlow 中 `Vθ=uθ+t·stopgrad(D_tuθ)` 完全等价，同时避免让数学 SDPA 的 JVP 激活驻留在梯度图中。训练和在线推理均使用 BF16，不存在 FP16/FP32 主值分支。EMA 与 optimizer/scheduler/RNG 都进入唯一 checkpoint。完整 MeanFlow/JVP 保持 eager；训练运行时只用 `torch.compile(fullgraph=True, mode="reduce-overhead")` 融合无参数、静态形状的度量几何投影器，因此 checkpoint 和推理数学图不变，也没有第二训练实现。每次反向仍在设备上检查 loss 和全局梯度范数是否有限，但用 CUDA 异步断言终止非法更新，不再为了读取布尔标量而强制每个 step 同步 CPU；不存在跳过 batch、降低 scale 或重抽 Flow source 的第二更新路径。

当前必要验证：

- `compileall`、`git diff --check`；
- 94 个数学、数据、前向、梯度和合同测试；
- 生产 39,804,232 参数图的 BF16 可训练主值、FP32 detached JVP 与反向检查；同一 RTX 4090、batch `256` 的交叉顺序 A/B 中，同步有限性检查为 `303.2/320.8 samples/s`，设备端异步断言为 `338.6/334.3 samples/s`，峰值显存均为 `13.89 GiB`，证明该修改只删除 host synchronization；
- 离线轨迹精度不按模型输出索引直接对齐，而是把所有模型和专家统一按绝对弧长重采样到前 `min(2 m, 专家局部长度)`；预测不足该距离时保持其终点继续计算误差，同时单列 horizon coverage，避免短轨迹靠少走获得低 ADE。安全评测独立以不大于 `0.025 m` 的间距覆盖前 `3.6 m` 局部轨迹，再查询与模型完全相同的四帧融合配置空间场；报告相对专家新增的 collision/margin violation，不把感知场自身对专家的误报归给模型；
- 6,087 条验证专家的四帧配置空间审计：当前帧单独识别专家 footprint collision `0` 条、裕度违例 `89` 条、直线裕度违例 `787` 条；四帧静态融合后分别为 `9`、`143`、`1,325` 条。9 条碰撞均可由具体单独历史帧复现，位姿为正常的约 `0.45/0.89/1.33 m` 后向平移，不是跨帧符号或偶然 observed 拼接错误；
- 被本节精度图替代的全 FP32 MeanFlow 主值在三张 RTX 4090 上按 `342/341/341` 分片、每 rank 两个 micro-batch 时约为 `0.72--0.96k samples/s`；分段剖析显示条件编码约 `26 ms`，而 JVP 约 `475 ms`，证明瓶颈在错误驻留于反向图的 JVP 主值，不在数据或四帧几何。新图单卡完整更新实测：batch `256` 峰值约 `13.85 GiB`、稳态约 `328--376 samples/s`；batch `342` 峰值约 `18.25 GiB`、稳态约 `408--442 samples/s`。这是单卡生产图结果；最终 3/4 卡端到端 DDP 吞吐仍以正式训练日志为准。
- 旧 decoder 的 V100S/4090 训练吞吐和推理延迟不适用于当前精确路径场查询图，已从当前结论删除。当前图的空闲态 4090 batch-1 延迟必须在本轮训练结束后重新实测，不用占卡争用数据代替；
- 新模型最终 200 epoch 墙钟时间和离线指标必须由本次训练实测，不沿用旧 checkpoint。

## 10. 正确性边界

代码保证：标定与坐标一致；历史变换同时用于障碍配准和因果状态；`L>0`；起点为原点；初始切向前向；路径正则且曲率连续；Flow 训练源合法；时间端点属于训练域；MeanFlow JVP 符号与一步推理一致；安全损失作用于实际生成曲线。

代码不保证：有限数据必然复现专家；深度可见空间等价完整地图；无长期记忆时一定走出迷宫或死胡同；软损失等价控制屏障函数；离线 ADE 必然转化为闭环成功率。这些必须由严格离线分层和固定协议闭环测评验证。
