# 训练数据

当前唯一训练入口为 `data/policy_dataset-depth-clearance/config.yaml`。
`bash scripts/build_dataset.sh` 生成并审计专家路线，然后直接编译为训练格式。

| 目录 | 用途 |
| --- | --- |
| `scene_assets/hssd-hab/` | HSSD 场景、网格和材质 |
| `../outputs/hssd_policy_depth_500_clearance/` | 500 条专家路线、10 Hz 深度与标定位姿、源配置空间和审计 |
| `policy_dataset-depth-clearance/` | 训练/验证标签、深度索引和训练集拟合的归一化配置 |

400 条路线用于训练，100 条用于验证，按场景隔离。深度直接保存为训练需要的
224×126 FP16 光学深度；四帧历史通过索引引用，不重复保存图像。
内参、外参、相对 SE(3) 位姿和时间戳使用统一观测合同。

专家曲线保存七个二维非原点 B-spline 控制点；原点固定。模型边界将控制点转换为
标准化控制增量，解码后恢复米制轨迹。源配置空间用于数据审核和测评，不作为在线输入。

生成完成需通过源几何审计与训练标签重新查询证书。旧 RGB-D / forward 数据集已清理；
数据、场景及模型权重留在本地，不提交 Git。
