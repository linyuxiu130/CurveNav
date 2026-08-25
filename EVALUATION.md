# X-NavDP 对齐评测合同

CurveNav、NavDP 和 X-NavDP 必须直接使用官方仓库 `baselines/x-navdp/eval` 的同一套评测代码与配置，不复制一份容易漂移的仿真器实现。

## 官方 wheeled PointGoal 口径

- 场景：`cluttered_easy`、`cluttered_hard`、`internscenes_home`、`internscenes_commercial`
- 机器人：Dingo
- 输入：PointGoal、RGB 历史、当前深度；CurveNav 当前策略只消费 PointGoal、深度和机器人位姿
- 图像尺寸：224
- RGB 历史样本：easy/hard/commercial 为 8，home 为 7
- 当前深度样本：1
- 到达阈值：1.0 m
- nominal speed：0.5 m/s
- MPC：`N=30, ref_gap=3, T=0.1, v_max=0.5, w_max=0.5`
- 规划：策略服务器与仿真控制异步运行
- success：episode 非 timeout 结束
- SPL：`success * initial_distance / max(path_length, initial_distance)`

## 公平比较规则

三种策略必须固定同一：

```text
官方代码提交 + scene USD + scene scale + episode index + sample index
+ 随机种子 + 相机 + MPC + controller + max episode time
```

每次结果至少保存逐 episode 的 `success`、`spl`、`distance` 和 `episode_idx`。先跑 10 条固定 episode 验证协议，再跑官方完整 episode；不得把旧 NavDP benchmark、修改后的成功阈值或不同 MPC 的结果混在同一表格。

上游评测真源位于：

```text
/mnt/data/huangshibo/H/navigation_three_projects/open_source_full/x_navdp/baselines/x-navdp/eval
```
