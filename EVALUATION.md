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
- `forward_open`：前向目标、当前深度存在有效障碍点，但专家弦线不触及安全边界。
- `forward_visible_detour`：前向目标，专家连续轨迹安全，而起点到局部专家终点的直线被当前深度障碍阻断；这是不引入人造障碍的基础避障结果。
- `rear_goal`：PointGoal 位于后半平面，单独报告，不与基础前向结果混淆。
- `expert_moves_away_from_goal`：专家局部段本身暂时远离任务目标，表示真实绕行或部分可观测情形，单独报告。

碰撞测量与训练共用四帧配准的 `64×64` 配置空间场。场值是障碍欧氏距离减 Dingo footprint radius；64 个等弧长轨迹点做双线性查询，负值为 footprint collision，小于额外 `0.10 m` 为安全裕度违例。代表案例固定选择各层 ADE 中位样本，并额外显示 `forward_visible_detour` 的最难样本，避免人工挑图。

```bash
scripts/evaluate_policy.sh configs/base.yaml CHECKPOINT \
  --artifact-dir outputs/strict-offline
```

CurveNav 用 8 维无界 Flow state 表示标准化 `log` 正弧长和 7 个 cubic B-spline 局部航向增量；单位切向积分保证轨迹正则、初始前向且曲率连续。训练为每个优化器 batch 显式采样一次随机高斯源并做闭区间时间 collocation，同时监督瞬时边界与数据锚定 improved MeanFlow；FP16 overflow 重试复用同一源。部署从固定高斯典型 latent 只做一次平均速度输运。四帧配准的坡度感知配置空间场完整编码为 `8×8` 度量安全 token，并在生成前融合进条件记忆；不再只沿更新前的 Flow 源路径读取不足 1% 的场。历史位姿既用于障碍配准，也以三个因果 SE(2) token 提供近期运动；训练和部署使用同一变换定义，不读取未来专家状态。不存在 ODE solver、随机候选、learned critic、在线碰撞修补或 fallback。

模型内部 64 点路径的第 0 点是当前机器人原点。官方 evaluator 会统一在 policy 返回值前追加当前原点，因此 CurveNav 的部署边界只发送内部路径的 `1:64` 共 63 个未来点；MPC 最终仍接收 64 点路径，且只有一个原点。NavDP/X-NavDP 的累积位移输出本来就不含当前点。若 CurveNav 发送内部第 0 点，evaluator 会制造两个连续原点，使 MPC 的起始离散曲率退化。

官方 evaluator 在每个 scene worker 中只创建一次 Isaac 环境，episode 结束后原地 reset 对应 env；同场景 10 回合不得拆成 10 次 Isaac 启动。比较同一场景的一组 checkpoint 时，唯一 evaluator 进程继续常驻并让 USD、物理世界和渲染资源保持在 GPU；checkpoint 之间完整 reset 全部 env 并重启策略服务，但不重建 scene，列表结束后才关闭 evaluator。CurveNav 在线使用 eager FP16，并在每次策略服务的初始 `navigator_reset` 内按实际 `num_envs` 完成 CUDA kernel 预热；该过程必须在 episode 计时循环前完成，不得通过放宽 timeout 或首轮零动作来掩盖初始化开销。

上游评测真源是部署时固定 commit 的 benchmark checkout：

```text
general-navigation-benchmark/baselines/x-navdp/eval
```

## 历史基线与当前在线状态

以下 2026-08-29 数值来自已被当前 iMeanFlow 合同替代的瞬时 CFM 基线，只用于同数据问题定位，不能作为当前代码成绩：

```text
checkpoint: /mnt/data/huangshibo/H/navigation_three_projects/curvenav/outputs/archive/train_policy-cumulative-heading-baseline-20260829/checkpoint.pt
offline:    /mnt/data/huangshibo/H/navigation_three_projects/curvenav/outputs/strict-offline-state-20260829
```

该基线 checkpoint step 为 8,000，使用 EMA；严格离线覆盖全部 6,087 条验证样本。总体 `ADE=0.11556 m`，前向/前向开阔/可见绕行层分别为 `0.10328/0.10651/0.17563 m`；坡度感知障碍定义下 footprint collision 为 `3.083%`、安全裕量违例为 `5.252%`。P95 最大曲率为 `3.7250 m⁻¹`，专家为 `1.8148 m⁻¹`，切向反转保持 `0%`。PointGoal 与当前深度打乱分别令 ADE 增加 `0.56703/0.15140 m`。其 8-step eager FP16 延迟 P50/P95 为 `382.1/502.0 ms`。当前 1-NFE 新图在 V100S、batch 1、FP16 预热后实测纯模型 `33.89 ms`、完整 runtime step `40.18 ms`；该数值只说明执行开销，最终 EMA 效果与 4090 空闲态延迟仍以本轮训练完成后的独立测评为准。

当前主要失败层不是普通前向跟随，而是 `forward_visible_detour`：其 footprint collision/safety violation 为 `12.17%/21.22%`。`rear_goal` 与 `expert_moves_away_from_goal` 的 ADE 为 `0.4592/0.5987 m`，作为部分可观测困难层单列，不用随机后向目标稀释基础结果。

同日公共离线对比中的 CurveNav 同样是该历史基线。公共集从修正后的 HSSD 专家路线重新渲染，包含 64 个样本、595 帧 RGB-D，并按专家累计转向分成四个等量难度层；四模型读取完全相同的历史观测和 PointGoal。统一按物理弧长比较前 `2 m`，并用当前深度的坡度感知 body obstacle 做逐段净空统计：

| 模型 | ADE / 高转向 ADE (m) | 覆盖 2 m | footprint collision | 安全裕量违例 | 曲率 P95 中位数 (m⁻¹) | 目标旋转响应 |
|---|---:|---:|---:|---:|---:|---:|
| CurveNav | **0.0817 / 0.1716** | 92.19% | 8.47% | 11.86% | **0.704** | 41.44° |
| NavDP | 0.2717 / 0.4062 | **95.31%** | **5.08%** | **8.47%** | 3.341 | 31.69° |
| SanD | 0.2342 / 0.3284 | 93.75% | 6.78% | **8.47%** | 1.929 | **50.13°** |
| X-NavDP | 0.2256 / 0.3004 | 76.56% | 8.47% | 11.86% | 1.827 | 1.02° |

该公共集的专家参考在单帧可见深度统计下本身为 `5.08%/8.47%`；这是可见点云遮挡、相机外区域与完整地图专家的观测合同差异，不等同于地图碰撞。因此安全结果必须相对专家基线解释：CurveNav 比参考多 2 个 footprint collision 样本和 2 个安全裕量违例样本；NavDP 与参考相同，SanD 多 1 个 collision 样本，X-NavDP 与 CurveNav 相同。当前 CurveNav 的优势是轨迹拟合和几何平滑，仍需重点降低可见绕行层的额外风险。完整数值、逐模型原始输出和交互对比位于 `outputs/offline-cross-model/full-20260829-safe-expert/`；各 runner 记录的总工作负载时间因 CurveNav 批处理而基线逐样本执行，不作为延迟横比。

旧固定高度障碍定义曾把同一段可通行坡面误报为 206 条专家碰撞。当前四帧坡度感知配置空间场用连续的 64 点标定射线覆盖可见栅格，并把已知障碍 `0.167584539 m` 机器人包络外再加 `0.10 m` 的已知风险域直接标为 observed。在 6,087 条验证样本中，当前帧单独计算的专家 footprint collision、裕度违例、直线裕度违例依次为 `0/89/787`；四帧融合后为 `9/143/1325`。9 条专家碰撞都能由具体单独历史帧复现，不是跨帧坐标拼接产生。后续新 checkpoint 必须使用这一合同，不能与旧 32 点射线、仅障碍中心 observed、稀疏点或固定高度统计横向混算。

4090 的 Isaac Sim 4.2 headless Vulkan/RTX/物理/深度 annotator 已用用户态 EGL ICD 与 NVIDIA 官方驱动校验开关通过 warm smoke；`64×64` 深度张量生成、world step 和清理均正常，进程状态为 0。唯一 benchmark checkout 位于 `/DataDisk2/hsb/general-navigation-benchmark-resident`，运行时参数在其 ignored `config/local.env`。锁定 commit `48e223e85f0408ebfd1d8c6d6fb0589e9c41b3aa` 的 acados 已在用户目录 Release 构建，`libblasfeo/libhpipm/libacados` 均从 Isaac Python 动态加载成功。launcher 单场景 dry-run 也已正确解析 GPU、权重、scene、evaluator 与 Kit 参数。正式固定协议尚未启动的唯一已确认阻塞是 4090 缺少官方 Scene-N1 资产树；资产同步并通过静态门禁后，必须在原固定场景、episode 0--9、相机、MPC、timeout 和 metric 合同下运行。
