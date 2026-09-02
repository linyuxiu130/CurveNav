# Data directory

Git 不保存数据。训练只读取 `data/policy_dataset-source-cspace`，其内容由 `scripts/build_dataset.sh` 从已审计、冻结的 `outputs/hssd_policy_dataset` 通过唯一链路原子编译；运行时不包含来源分支。当前数据只来自 CurveNav 按 Dingo 配置空间生成的 HSSD 专家路线，不混入 SanD 或 NavDP 数据。

统一样本包含四帧标定 `224×126`、`5 m` 有效范围的深度索引（3 帧过去观测 + 当前观测）、PointGoal、逐帧相对位姿与有效位，以及七个二维 cubic B-spline 控制点的 14 个物理坐标值。首个控制点固定为机器人原点；64 点专家路径只在编译时用于最小二乘投影和审计，不作为训练数组重复保存，训练和推理都由同一个 codec 解码。

物理控制点是唯一写盘曲线定义；codec 在模型边界把相邻控制点差转换为 14 维标准化欧氏 Flow 坐标，推理后再以累积和精确恢复控制点。数据中不保存第二份增量标签。

完整连续 HSSD route 被切成局部样本：每个非终点时刻保留未来最多 24 个专家步，真正到达终点时允许更短；路径不缩放到固定长度。数据包含 20 个 scene-disjoint 场景中的 500 条无扰动 route，沿轨迹以 `0.15 m` 采样。`observation_to_current` 表示每个观测帧到当前坐标系的 `(x,y,sin Δyaw,cos Δyaw)`，同时用于深度 token 的刚体对齐和三个历史运动状态 token。

`outputs/hssd_policy_dataset` 是一次性上游场景、route 和深度缓存准备结果；它不是训练入口的一条备用生成路线。`scripts/build_dataset.sh` 只编译它为唯一 prepared dataset，并在结束时用 manifest 固定的 `endpoint_inclusive_max_spacing`、最大 `0.025m` 间隔规则完成 source C-space re-query certificate。场景资产、生成结果、日志和 compiled dataset 不提交 Git。
