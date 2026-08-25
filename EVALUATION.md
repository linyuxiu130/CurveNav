# CurveNav 评测合同

当前 CurveNav 几何测评先使用训练深度的唯一相机合同。与 NavDP/X-NavDP 做正式对比，则三者必须直接使用官方仓库 `baselines/x-navdp/eval` 的同一套评测代码与同一套相机配置，不复制一份容易漂移的仿真器实现。

## 当前 CurveNav 深度口径

- 深度：`224×126`，光轴距离，最大值 `5 m`
- 内参：`fx=fy=166.80851, cx=112, cy=63`
- 外参：无前移、高度 `0.30 m`、水平安装
- 显式评价器：只使用当前帧，按上述参数反投影可见障碍表面

## 官方 wheeled PointGoal 口径

- 场景：`cluttered_easy`、`cluttered_hard`、`internscenes_home`、`internscenes_commercial`
- 机器人：Dingo
- 输入：PointGoal、RGB 历史、当前深度；CurveNav 当前策略只消费 PointGoal、四帧深度和逐帧相对变换
- 图像尺寸：224
- RGB 历史样本：easy/hard/commercial 为 8，home 为 7
- 当前深度样本：1
- 官方 Dingo depth：`640×360` D455，`fx=fy=326.39856`，相机前移 `0.28618 m`、高度 `0.62532 m`、下俯 `10°`
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

当前 `0.30 m` 相机结果不能与官方 Dingo D455 结果直接横向比较。正式对比前必须让 CurveNav、NavDP 和 X-NavDP 共同使用同一相机重新评测；仅对深度图做二维缩放不等价于修改相机外参。

CurveNav 每个规划周期生成十六条候选，并且只按当前深度构成的显式 `clearance + length + goal` 几何代价选择。闭环主结果不得混入 learned scorer、oracle ADE 选轨、标签轨迹或额外启发式碰撞 mask；离线同时报告 oracle ADE 是为了诊断生成覆盖，不参与实际选择。

上游评测真源位于：

```text
/mnt/data/huangshibo/H/navigation_three_projects/open_source_full/x_navdp/baselines/x-navdp/eval
```
