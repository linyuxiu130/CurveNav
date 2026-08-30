# CurveNav 评测合同

当前 CurveNav 几何测评先使用训练深度的唯一相机合同。与 NavDP/X-NavDP 做正式对比，则三者必须直接使用官方仓库 `baselines/x-navdp/eval` 的同一套评测代码与同一套相机配置，不复制一份容易漂移的仿真器实现。

## 当前 CurveNav 深度口径

- 深度：`224×126`，光轴距离，最大值 `5 m`
- 内参：`fx=fy=166.80851, cx=112, cy=63`
- 外参：前移 `0.28618 m`、高度 `0.62532 m`、下俯 `10°`
- 模型视觉输入：四帧均按上述参数反投影并对齐到当前机器人系；局部表面法向坡度不超过 Dingo 合同 `45°` 的深度面视为可通行地形，不进入二维障碍集合

## 官方 wheeled PointGoal 口径

- 场景：`cluttered_easy`、`cluttered_hard`、`internscenes_home`、`internscenes_commercial`
- 机器人：Dingo
- 输入：PointGoal、RGB 历史、当前深度；CurveNav 当前策略只消费 PointGoal、四帧深度和逐帧相对变换
- 图像尺寸：224
- RGB 历史样本：easy/hard/commercial 为 8，home 为 7
- 当前深度样本：1
- 官方 Dingo depth 参考内参：`640×360`，`fx=fy=326.39856, cx=320, cy=180`；当前吞吐链保持相同 horizontal/vertical aperture 与 FoV，将渲染采样减为 `320×180`，因此四个像素内参分别按宽高乘 `1/2`。两种采样都重投影到同一个 `224×126` 模型相机；相机前移 `0.28618 m`、高度 `0.62532 m`、下俯 `10°`
- 到达阈值：1.0 m
- nominal speed：0.5 m/s
- MPC：`N=30, ref_gap=3, T=0.1, v_max=0.5, w_max=0.5`
- 规划：策略服务器与仿真控制异步运行
- success：episode 非 timeout 结束
- SPL：`success * initial_distance / max(path_length, initial_distance)`

## 公平比较规则

官方横向比较时，三种策略必须固定同一：

```text
官方代码提交 + scene USD + scene scale + episode index + sample index
+ 随机种子 + 相机 + MPC + controller + max episode time
```

每次结果至少保存逐 episode 的 `success`、`spl`、`distance` 和 `episode_idx`。先跑 10 条固定 episode 验证协议，再跑官方完整 episode；不得把旧 NavDP benchmark、修改后的成功阈值或不同 MPC 的结果混在同一表格。

当前 prepared dataset、checkpoint 合同和部署输入统一使用上述 Dingo D455 标定。旧 `0.40 m` 水平相机 checkpoint 与当前合同不兼容，必须直接拒绝，不能通过二维缩放或兼容分支继续评测。

## 严格离线协议

离线测评不生成随机目标，也不把 PointGoal 投影到局部专家终点。验证集中的每个 PointGoal 都是对应 HSSD 专家路线的真实任务终点；起终点均由同一份 Dingo 膨胀 NavMesh 采样，A* 连通，完整专家路线以 `0.025 m` 间隔验证连续净空。训练集和验证集按场景及 source family 隔离。

完整验证集只按已有物理情形分层，不删除困难样本：

- `forward`：PointGoal 在机器人前半平面；这是基础 PointGoal 跟随结果。
- `forward_direct`：前向目标，专家连续轨迹与通向局部专家终点的直线都不触及感知安全边界。
- `forward_detour`：前向目标，专家连续轨迹安全，而起点到局部专家终点的直线被安全场阻断；在 CurveNav 验证中安全场来自四帧可见深度，在跨模型比较中来自冻结 HSSD 配置空间真值。
- `rear_goal`：PointGoal 位于后半平面，单独报告，不与基础前向结果混淆。
- `expert_moves_away_from_goal`：专家局部段本身暂时远离任务目标，表示真实绕行或部分可观测情形，单独报告。

不同模型的输出点数、路径长度和采样方式不同，因此横向测评不按输出数组下标直接比较。每条预测与专家都按绝对弧长重采样到前 `min(2 m, 专家局部长度)` 的 81 个位置；预测短于该距离时保持预测终点继续计算误差，而不是只比较重叠段。统一报告固定距离 ADE/FDE、horizon coverage、PointGoal progress regret、曲率和曲率变化。这样短轨迹、长轨迹和不同离散点数不会获得不公平优势。

碰撞测量与训练共用四帧配准的 `64×64` 配置空间场。场值是障碍欧氏距离减 Dingo footprint radius；评测先把路径以不大于 `0.025 m` 的独立间距稠密化，再做双线性查询，负值为 footprint collision，小于额外 `0.10 m` 为安全裕度违例。安全结果同时报告专家在同一感知场上的基线和模型相对专家新增的风险，避免把遮挡、深度噪声或感知场误报归因于生成器。`observed_path_fraction` 单列，未知空间不能被解释为安全。

代表案例固定选择各层固定距离 ADE 中位样本，并额外显示 `forward_detour` 的最难样本，避免人工挑图。案例 JSON 直接保存产生安全指标的同一四帧融合 clearance/observed/forbidden 网格，不再用仅当前帧稀疏障碍点画一张与判定真源不同的图。

```bash
scripts/evaluate_policy.sh configs/base.yaml CHECKPOINT \
  --artifact-dir outputs/offline-evaluation
```

唯一主报告分四组，不合成主观加权总分：

- 精度：`fixed_horizon_ade/fde` 的 mean 与 P90，高转向 10% 单列；
- 任务性：`goal_progress_regret`、负进度比例和 `horizon_coverage`；
- 安全：稠密 footprint collision、`0.10 m` 裕度违例、最坏违例深度，以及相对专家新增风险；
- 可执行性：弧长、曲率 P95、RMS 曲率、单位长度曲率变化和切向反转率。

四帧模型额外用“只保留当前帧有效”的同网络消融测历史输入净增益；该项只用于 CurveNav 架构诊断，不参与 SanD/NavDP/X-NavDP 横向排名。旧的 batch 内 PointGoal/depth 随机错配会制造训练分布外条件，已经删除。

跨模型比较使用 `scripts/compare_offline.sh`。公共 NPZ 必须冻结 `axis=x_forward_y_left`、`reference_path`、`point_goal`、`scene_id`、`route_id`、`origin_xy` 和 `route_yaw`；每个映射结果 NPZ 必须显式写入相同 `axis`、`base_path`、`base_length`、模型名和 checkpoint。安全真值直接查询生成专家时冻结的 Dingo 膨胀 HSSD navigation grid，路径按 `0.025 m` 稠密化后映射回世界坐标；不再从缺少完整外参语义的 RGB-D 文件临时猜测安全场。旧公共集和四模型结果缺少显式 axis，只能作为历史产物，补齐协议字段后才能进入新排名。

比较开始前，公共集中的专家路径必须在该冻结地图上同时满足 `0` footprint collision、`0` 个 `0.10 m` 裕度违例以及 `100%` 地图覆盖；任一条件不满足就拒绝整份输入。这个门禁用于发现公共样本与地图版本、route pose 或坐标轴错配，禁止把协议错误计为模型错误。

```bash
scripts/compare_offline.sh configs/base.yaml COMMON.npz HSSD_SOURCE REPORT.json \
  CURVENAV.npz SAND.npz NAVDP.npz XNAVDP.npz
```

报告除全样本与物理分层外，还逐 held-out scene 输出同一指标，并给 scene-macro 与 worst-scene；样本数更多的场景不能淹没小场景上的完全失败。

CurveNav 用 8 维无界 Flow state 表示标准化 `softplus` 弧长预激活和 7 个 cubic B-spline 局部航向增量；单位切向积分保证轨迹正则、初始前向且曲率连续。训练为每个优化器 batch 显式采样一次随机高斯源并做闭区间时间 collocation，同时监督瞬时边界与数据锚定 improved MeanFlow；所有可训练主值与部署统一使用 BF16，只有 stop-gradient 的精确 forward-mode JVP 使用 FP32。部署从固定高斯典型 latent 只做一次平均速度输运。四帧配准的坡度感知配置空间场完整编码为 `8×8` 度量安全 token，并在生成前融合进条件记忆；不再只沿更新前的 Flow 源路径读取不足 1% 的场。历史位姿既用于障碍配准，也以三个因果 SE(2) token 提供近期运动；训练和部署使用同一变换定义，不读取未来专家状态。不存在 ODE solver、随机候选、learned critic、在线碰撞修补或 fallback。

模型内部 64 点路径的第 0 点是当前机器人原点。官方 evaluator 会统一在 policy 返回值前追加当前原点，因此 CurveNav 的部署边界只发送内部路径的 `1:64` 共 63 个未来点；MPC 最终仍接收 64 点路径，且只有一个原点。NavDP/X-NavDP 的累积位移输出本来就不含当前点。若 CurveNav 发送内部第 0 点，evaluator 会制造两个连续原点，使 MPC 的起始离散曲率退化。

官方 evaluator 在每个 scene worker 中只创建一次 Isaac 环境，episode 结束后原地 reset 对应 env；同场景 10 回合不得拆成 10 次 Isaac 启动。比较同一场景的一组 checkpoint 时，唯一 evaluator 进程继续常驻并让 USD、物理世界和渲染资源保持在 GPU；checkpoint 之间完整 reset 全部 env 并重启策略服务，但不重建 scene，列表结束后才关闭 evaluator。CurveNav 在线使用 eager BF16，并在每次策略服务的初始 `navigator_reset` 内按实际 `num_envs` 完成 CUDA kernel 预热；该过程必须在 episode 计时循环前完成，不得通过放宽 timeout 或首轮零动作来掩盖初始化开销。

上游评测真源是部署时固定 commit 的 benchmark checkout：

```text
general-navigation-benchmark/baselines/x-navdp/eval
```

## 当前终态离线结果（2026-08-30）

本节 checkpoint 是闭环结构重构前的 8,000-step EMA 基线。它用于定位旧生成器的信息瓶颈，不是正在训练的路径相对配置空间模型成绩；新模型只有完成训练、离线门禁和固定 10 回合闭环后才能替换本节结论。

最终 EMA checkpoint 为 step 8,000。完整自然分布验证集含 6,087 条 held-out HSSD 样本：固定 `2 m` ADE mean/P90 为 `0.05776/0.15277 m`，FDE mean/P90 为 `0.16284/0.42437 m`，覆盖率 `97.94%`，footprint collision `1.544%`，`0.10 m` 裕度违例 `4.436%`，负目标进度 `2.399%`。直接前向层的 ADE/collision 为 `0.03539 m/0.282%`；前向绕障层为 `0.12266 m/4.919%`；后向目标层为 `0.16808 m/10.00%`。只保留当前深度帧会把 ADE 从 `0.05776 m` 提高到 `0.06995 m`，碰撞从 `1.544%` 提高到 `2.776%`，说明四帧历史在同一模型上的净增益成立。空闲 4090 上纯模型 batch-1 FP16 延迟 P50/P95 为 `50.21/50.58 ms`，批量 32 的观测吞吐为 `596.59 obs/s`。

跨模型压力集含相同的 64 条样本和四个 held-out scene，刻意提高了绕障、后向目标和短时背离目标样本的比例，因此只用于比较能力边界，不代表自然场景频率。所有输出都按同一物理弧长、坐标轴和冻结 Dingo 配置空间地图重新计分；专家门禁结果是 `0` collision、`0` 裕度违例、`100%` 覆盖。

| 模型 | ADE mean/P90 (m) | FDE mean (m) | 路径覆盖 | footprint collision | 裕度违例 | 曲率 P95 中位数 (m⁻¹) | 最差场景 ADE / collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| CurveNav | **0.1181 / 0.2515** | **0.3160** | 98.33% | **32.81%** | **37.50%** | **0.908** | **0.1700 / 45.45%** |
| NavDP | 0.3660 / 0.6284 | 0.6270 | 97.78% | 64.06% | 68.75% | 3.297 | 0.4016 / 75.76% |
| SanD | 0.2339 / 0.4757 | 0.4805 | **99.66%** | 60.94% | 64.06% | 1.922 | 0.2997 / 75.76% |
| X-NavDP | 0.2812 / 0.5629 | 0.4375 | 93.95% | 40.62% | 45.31% | 1.721 | 0.3209 / 51.52% |

压力集不能仅按目标进度排名：NavDP、SanD 和 X-NavDP 的平均 progress regret 分别为 `-0.377/-0.252/-0.116 m`，但更直接地朝目标推进同时显著增加了碰撞；CurveNav 为 `+0.031 m`。因此任务性、几何拟合、安全和可执行性保持分组报告，不合成一个可以被激进直行投机的总分。

```text
final checkpoint: /DataDisk2/hsb/curvenav-f19ba8a/outputs/archive/train_policy-pre-path-relative-20260830/checkpoint.pt
full CurveNav:    /DataDisk2/hsb/curvenav-f19ba8a/outputs/offline-evaluation-8000-d56068c-20260830/offline-metrics.json
cross-model:      /DataDisk2/hsb/offline-cross-model/results/full-20260830-metric-12612de/comparison.json
common dataset:   /DataDisk2/hsb/offline-cross-model/data/offline-common-hssd-64-metric-12612de.npz
safety source:    /DataDisk2/hsb/offline-cross-model/source-metric-12612de
```

## 单场景闭环诊断（2026-08-30）

场景固定为 `home/MVUCSQAKTKJ5EAABAAAAABA8_usd`，episode `0--99`、seed `1234`、官方 Dingo、相机、异步 MPC、timeout 和 SPL 公式完全相同。当前这一轮使用 `num_envs=10` 加速收集配对轨迹，因此只称为单场景吞吐诊断，不冒充论文 20/40 场景均值或 `num_envs=1` 固定协议成绩。

| 模型 | 完成回合 | SR | mean SPL | 结果状态 |
|---|---:|---:|---:|---|
| CurveNav（结构重构前 checkpoint） | 100 | 32.00% | 0.297861 | 完整 |
| NavDP | 100 | 59.00% | 0.568184 | 完整 |

同一 episode 配对为：两者都成功 `27`，仅 CurveNav 成功 `5`，仅 NavDP 成功 `32`，两者都失败 `36`。因此至少 32 个 CurveNav 失败回合能由相同仿真、起终点和控制器下的 NavDP 完成，不能归因于场景普遍不可达或评测整体失效。

为排除“失败组只是目标更远或初始方位更难”，进一步只比较两个配对子集。
`NavDP-only` 的初始目标距离/绝对方位为 `6.091 m/0.668 rad`，与
`both-success` 的 `5.915 m/0.603 rad` 接近。CurveNav 首次局部规划的
起始切向与 PointGoal 余弦分别为 `0.706/0.730`，终点余弦为
`0.854/0.818`，弧长为 `3.573/3.446 m`；目标融合和轨迹长度并没有在
失败前先显著分离。最早的分离发生在空间可行性：首条轨迹中距官方
可行驶机器人中心集合超过 `0.15 m` 的点比例为 `19.58%/11.57%`，
最近距离均值为 `0.0885/0.0488 m`，最大值为 `0.3848/0.2205 m`；
前五次规划的超阈值点比例仍为 `19.55%/9.04%`。两者都失败组的
首条轨迹超阈值比例更高，为 `30.21%`。这一时序证据表明几何偏离在
机器人被阻滞和 PointGoal 转到侧后方之前已经出现，因果顺序不是“先跟踪失败，
再使规划看起来不安全”。

CurveNav 成功/失败轨迹的核心差异如下。可行驶距离使用本场景官方 `navigable.ply` 点集；每条预测先按机器人世界位姿转换，再查询最近可行驶机器人中心，`>0.15 m` 仅作为诊断阈值，不替代官方 success 或碰撞定义。

| 量 | 成功 32 | 失败 68 |
|---|---:|---:|
| 初始目标距离均值 | 6.016 m | 5.855 m |
| 最终目标距离均值 | 0.283 m | 4.575 m |
| 目标进展比例 | 94.96% | 21.45% |
| 目标距离回退 | 0.037 m | 0.874 m |
| 实际平均速度 | 0.315 m/s | 0.041 m/s |
| 有效命令期间停止比例 | 3.84% | 90.64% |
| 实际速度/线速度命令 | 91.97% | 10.29% |
| 平均局部规划弧长 | 2.105 m | 2.885 m |
| 路径曲折度 | 1.121 | 1.947 |
| 规划点离可行驶集合 `>0.15 m` | 2.66% | 49.15% |
| 含任一上述点的规划比例 | 13.49% | 97.69% |
| 实际机器人采样点离集合 `>0.15 m` | 0.66% | 16.10% |
| 局部起始切向与 PointGoal 余弦 | 0.914 | 0.232 |
| 局部终点与 PointGoal 余弦 | 0.962 | 0.487 |

这些数字支持一条闭环因果链，而不是几个独立补丁需求：旧模型先生成与障碍几何不一致的路径；MPC 仍发出约 `0.41 m/s` 的正向命令，但实体被障碍/不可行驶边界阻滞；机器人没有按计划移动后，PointGoal 转到侧后方，后续局部规划的目标一致性和单调进展继续恶化。失败回合的局部规划更长且曲率更低，排除了“3.6 m 不够长”或“曲率限制太严格”作为主因；成功与失败的初始目标距离也近似相同，排除了单纯任务长度差异。

逐层反证后的结构根因是旧生成器中的**路径位置—配置空间信息失配**。旧条件编码器先让 64 个度量配置空间 cell 被通用 goal/vision/motion token 查询并相加，随后丢弃作为独立 memory 的 cell；decoder 的 path token 又建立在当前 Flow state 解码的曲线上。部署从独立高斯源开始时，这条曲线不是待输出路径，因此生成器没有在“自己真正准备执行的位置”读取 clearance、梯度、可见性和禁行占据。训练期末端安全损失能降低平均风险，但不能恢复推理图中缺失的空间对应关系。

当前唯一重构直接修正该信息流：64 个配置空间 token 保持为一等条件 memory；第一阶段从完整条件估计数据端曲线，后两阶段沿该估计曲线的 16 个锚点连续查询原始配置空间场并细化同一 Flow velocity；三个阶段共用 readout，且都满足相同 Improved MeanFlow 恒等式监督。它不投影、裁剪或拒绝输出，也不增加候选评价头，因此是生成器内部的空间条件化修正，不是在线避碰补丁。其有效性仍须由新 checkpoint 的离线安全门禁与 10 回合闭环证明。

```text
CurveNav metric: /DataDisk2/hsb/eval-server-audit/runs/four-model-resident-b100-control-r2/pointgoal-v2/resident/20260830_002958/models/curvenav/00-curvenav-98d1e8d4d592/scenes/home/MVUCSQAKTKJ5EAABAAAAABA8_usd/metric.csv
NavDP metric:    /DataDisk2/hsb/eval-server-audit/runs/four-model-resident-b100-control-r2/pointgoal-v2/resident/20260830_002958/models/navdp/01-navdp-cc0246524765/scenes/home/MVUCSQAKTKJ5EAABAAAAABA8_usd/metric.csv
paired traces:   /DataDisk2/hsb/eval-server-audit/runs/four-model-resident-b100-control-r2/pointgoal-v2/resident/20260830_002958/trajectory_episodes.csv
```

## 历史基线与当前在线状态

以下 2026-08-29 数值来自已被当前 iMeanFlow 合同替代的瞬时 CFM 基线，只用于同数据问题定位，不能作为当前代码成绩：

```text
checkpoint: /mnt/data/huangshibo/H/navigation_three_projects/curvenav/outputs/archive/train_policy-cumulative-heading-baseline-20260829/checkpoint.pt
offline:    /mnt/data/huangshibo/H/navigation_three_projects/curvenav/outputs/strict-offline-state-20260829
```

该基线 checkpoint step 为 8,000，使用 EMA；严格离线覆盖全部 6,087 条验证样本。总体 `ADE=0.11556 m`，前向/前向开阔/可见绕行层分别为 `0.10328/0.10651/0.17563 m`；坡度感知障碍定义下 footprint collision 为 `3.083%`、安全裕量违例为 `5.252%`。P95 最大曲率为 `3.7250 m⁻¹`，专家为 `1.8148 m⁻¹`，切向反转保持 `0%`。PointGoal 与当前深度打乱分别令 ADE 增加 `0.56703/0.15140 m`。其 8-step eager FP16 延迟 P50/P95 为 `382.1/502.0 ms`。

当前主要失败层不是普通前向跟随，而是旧报告中的可见绕行层：其 footprint collision/safety violation 为 `12.17%/21.22%`。`rear_goal` 与 `expert_moves_away_from_goal` 的旧索引对齐 ADE 为 `0.4592/0.5987 m`，作为历史定位信息保留；新模型统一使用 `forward_detour` 和固定物理距离合同重新计算，不与这些旧口径数值直接横比。

同日公共离线对比中的 CurveNav 同样是该历史基线。公共集从修正后的 HSSD 专家路线重新渲染，包含 64 个样本、595 帧 RGB-D，并按专家累计转向分成四个等量难度层；四模型读取完全相同的历史观测和 PointGoal。统一按物理弧长比较前 `2 m`，并用当前深度的坡度感知 body obstacle 做逐段净空统计：

| 模型 | ADE / 高转向 ADE (m) | 覆盖 2 m | footprint collision | 安全裕量违例 | 曲率 P95 中位数 (m⁻¹) | 目标旋转响应 |
|---|---:|---:|---:|---:|---:|---:|
| CurveNav | **0.0817 / 0.1716** | 92.19% | 8.47% | 11.86% | **0.704** | 41.44° |
| NavDP | 0.2717 / 0.4062 | **95.31%** | **5.08%** | **8.47%** | 3.341 | 31.69° |
| SanD | 0.2342 / 0.3284 | 93.75% | 6.78% | **8.47%** | 1.929 | **50.13°** |
| X-NavDP | 0.2256 / 0.3004 | 76.56% | 8.47% | 11.86% | 1.827 | 1.02° |

该公共集的专家参考在单帧可见深度统计下本身为 `5.08%/8.47%`；这是可见点云遮挡、相机外区域与完整地图专家的观测合同差异，不等同于地图碰撞。因此安全结果必须相对专家基线解释：CurveNav 比参考多 2 个 footprint collision 样本和 2 个安全裕量违例样本；NavDP 与参考相同，SanD 多 1 个 collision 样本，X-NavDP 与 CurveNav 相同。当前 CurveNav 的优势是轨迹拟合和几何平滑，仍需重点降低可见绕行层的额外风险。完整数值、逐模型原始输出和交互对比位于 `outputs/offline-cross-model/full-20260829-safe-expert/`；各 runner 记录的总工作负载时间因 CurveNav 批处理而基线逐样本执行，不作为延迟横比。

旧固定高度障碍定义曾把同一段可通行坡面误报为 206 条专家碰撞。当前四帧坡度感知配置空间场用连续的 64 点标定射线覆盖可见栅格，并把已知障碍 `0.167584539 m` 机器人包络外再加 `0.10 m` 的已知风险域直接标为 observed。在 6,087 条验证样本中，当前帧单独计算的专家 footprint collision、裕度违例、直线裕度违例依次为 `0/89/787`；四帧融合后为 `9/143/1325`。9 条专家碰撞都能由具体单独历史帧复现，不是跨帧坐标拼接产生。后续新 checkpoint 必须使用这一合同，不能与旧 32 点射线、仅障碍中心 observed、稀疏点或固定高度统计横向混算。

4090 的 Isaac Sim 4.2 headless Vulkan/RTX/物理/深度 annotator 已用用户态 EGL ICD 与 NVIDIA 官方驱动校验开关通过 warm smoke；`64×64` 深度张量生成、world step 和清理均正常。唯一 benchmark checkout 位于 `/DataDisk2/hsb/general-navigation-benchmark-resident`，运行时参数在其 ignored `config/local.env`。锁定 commit `48e223e85f0408ebfd1d8c6d6fb0589e9c41b3aa` 的 acados 已在用户目录 Release 构建，`libblasfeo/libhpipm/libacados` 均从 Isaac Python 动态加载成功。Scene-N1 资产已同步并通过静态门禁。固定协议为同一 `home/MVUCSQAKTKJ5EAABAAAAABA8_usd`、seed `1234`、episode `0--99`、`num_envs=1`；不混入此前 `num_envs=10` 吞吐诊断：

| 模型 | 当前完成 | success | SR | mean SPL | 状态 |
|---|---:|---:|---:|---:|---|
| X-NavDP | 100/100 | 90 | 90.00% | 0.762712 | 完整 |
| SanD | 44/100 | 32 | 72.73% | 0.554173 | 四卡训练期间暂停，可续跑 |
| NavDP | 7/100 | 5 | 71.43% | 0.693114 | 四卡训练期间暂停，可续跑 |

X-NavDP 的完整 metric 位于 `/DataDisk2/hsb/eval-server-audit/runs/x-navdp-b100-numenv1-r3/pointgoal-v2/x-navdp/20260830_045722/models/x-navdp/00-x-navdp-981ae026d7f4/scenes/home/MVUCSQAKTKJ5EAABAAAAABA8_usd/metric.csv`。SanD 与 NavDP 的比例只是恢复断点事实，不作为最终模型横比；两者达到精确 100 回合并核验 trace 后再固化最终值。当前四张 4090 用于从头训练路径相对配置空间模型，训练完成后从上述 episode 断点恢复两个基线。
