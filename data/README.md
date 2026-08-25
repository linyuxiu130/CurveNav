# Data directory

Git 不保存数据。训练只读取 `data/policy_dataset`，其内容由 `scripts/build_dataset.sh DATA_ROOT` 通过唯一链路原子编译；运行时不再包含 SanD/HSSD 来源分支。SanD 与 HSSD 深度都按完整 route 存储，统一局部样本只保存深度索引。

统一样本包含四帧标定 224×126、5 m 有效范围的深度（3 帧过去观测 + 当前观测）、PointGoal、逐帧相对位姿与有效位、八个 B-spline 控制点和 64 点等弧长 `reference_path`。不同来源先按各自相机内参重投影，再进入同一个 prepared dataset。

SanD 与 HSSD 都由完整连续 route 切片：每个非终点时刻保留未来最多 24 个专家步，真正到达终点时允许更短；路径不缩放到固定长度。HSSD 包含 20 个 scene-disjoint 场景中的 500 条无扰动 route，沿轨迹以 0.15 m 采样。`observation_to_current` 统一表示每个观测帧到当前坐标系的 `(x,y,sin Δyaw,cos Δyaw)`，并用于深度 token 的平面反投影对齐。

HSSD 原始数据由唯一入口内部的 `scripts/generate_hssd_dataset.sh` 生成到 `outputs/hssd_policy_dataset`。场景资产、生成结果、日志和 compiled dataset 不提交 Git。
