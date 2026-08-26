# CurveNav 评测合同

当前 CurveNav 几何测评先使用训练深度的唯一相机合同。与 NavDP/X-NavDP 做正式对比，则三者必须直接使用官方仓库 `baselines/x-navdp/eval` 的同一套评测代码与同一套相机配置，不复制一份容易漂移的仿真器实现。

## 当前 CurveNav 深度口径

- 深度：`224×126`，光轴距离，最大值 `5 m`
- 内参：`fx=fy=166.80851, cx=112, cy=63`
- 外参：前移 `0.28618 m`、高度 `0.62532 m`、下俯 `10°`
- 模型视觉输入：四帧均按上述参数反投影并对齐到当前机器人系

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

CurveNav 在固定 `y=8x` 无量纲空间用八个有序 curve query 直接预测一条条件有界曲率轨迹，生产坐标、解码后的 metric path 和 tangent 共同监督这个唯一输出。部署只做一次相同 Transformer 前向和一次解析几何解码；不存在 Flow/ODE、episode 随机状态、候选集、候选排序、历史重建打分、learned critic 或碰撞启发式。SanD 的随机候选由 ESDF 评价器选择，NavDP/X-NavDP 的多样候选由 critic/Q 机制约束；当前单专家阶段没有这些监督，因此路线选择由同一个 geometry-supervised 解码器直接承担。离线报告实际部署轨迹的 ADE、弧长、目标进展、终端航向误差、延迟，以及弧长域曲率 B-spline 直接给出的连续曲率；同时单独报告参考曲率最高 10% 样本的 ADE、终端航向、预测/参考曲率比和全体曲率相关系数，避免总体均值掩盖强转弯失败。ADE 仍按 1/2/3/4 帧有效观测分组，直接检查部署冷启动与完整历史条件下的轨迹质量。这些分组只读取现有标签或 `observation_valid`，不参与训练或推理；不再用 XY 控制多边形离散转角冒充轨迹曲率界。

模型内部 64 点路径的第 0 点是当前机器人原点。官方 evaluator 会统一在 policy 返回值前追加当前原点，因此 CurveNav 的部署边界只发送内部路径的 `1:64` 共 63 个未来点；MPC 最终仍接收 64 点路径，且只有一个原点。NavDP/X-NavDP 的累积位移输出本来就不含当前点。若 CurveNav 发送内部第 0 点，evaluator 会制造两个连续原点，使 MPC 的起始离散曲率退化。

官方 evaluator 在每个 scene worker 中只创建一次 Isaac 环境，episode 结束后原地 reset 对应 env；同场景 10 回合不得拆成 10 次 Isaac 启动。比较同一场景的一组 checkpoint 时，唯一 evaluator 进程继续常驻并让 USD、物理世界和渲染资源保持在 GPU；checkpoint 之间完整 reset 全部 env 并重启策略服务，但不重建 scene，列表结束后才关闭 evaluator。CurveNav 在线使用 eager FP16，并在每次策略服务的初始 `navigator_reset` 内按实际 `num_envs` 完成 CUDA kernel 预热；该过程必须在 episode 计时循环前完成，不得通过放宽 timeout 或首轮零动作来掩盖初始化开销。

上游评测真源是部署时固定 commit 的 benchmark checkout：

```text
general-navigation-benchmark/baselines/x-navdp/eval
```
