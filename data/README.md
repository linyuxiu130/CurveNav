# Data directory

Git 不保存数据。训练只读取 `data/policy_dataset`，其内容由 `scripts/build_dataset.sh DATA_ROOT` 通过唯一链路原子编译；运行时不再包含 SanD/HSSD 来源分支。SanD 深度按 route 存储，HSSD 深度按物理重渲染样本存储，统一样本只保存深度索引。

统一样本包含四帧标定 224×126、5 m 有效范围的深度（3 帧过去观测 + 当前观测）、PointGoal、逐帧相对位姿与有效位、八个 B-spline 控制点和 64 点等弧长 `reference_path`。不同来源先按各自相机内参重投影，再进入同一个 prepared dataset。

SanD 按原始序列保留未来最多 24 个专家步，真正到达终点时允许更短；静止/重复步不得在截取前删除。HSSD 直接提供当前坐标系下 clearance-aware 重规划路径的最多 3 m 前缀，不伪装成 SanD route，也不将路径缩放到固定长度。`observation_to_current` 统一表示每个观测帧到当前坐标系的 `(x,y,sin Δyaw,cos Δyaw)`，并用于深度 token 的平面反投影对齐。

HSSD 原始数据由唯一入口内部的 `scripts/generate_hssd_dataset.sh` 生成到 `outputs/hssd_policy_dataset`。场景资产、生成结果、日志和 compiled dataset 不提交 Git。
