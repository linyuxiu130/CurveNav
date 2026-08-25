# CurveNav

CurveNav 是高效的 PointGoal 条件二维局部规划器。模型读取三帧过去深度与一帧当前深度、对应逐帧相对位姿和当前 PointGoal，生成短距离平滑 B-spline，并通过条件兼容度选择一条执行轨迹。

当前目标只有一个：在 X-NavDP 官方 PointGoal 评测中，以完全相同的 episode、相机、异步 MPC 和指标口径对比 NavDP 与 X-NavDP。局部基线成立前不专项扩展长距离或脱困能力。

架构合同见 `ARCHITECTURE.md`，评测合同见 `EVALUATION.md`，保留的实验结论见 `EXPERIMENTS.md`。

## 唯一工作流

项目环境：

```text
../.venvs/curvenav
```

从固定上游 commit 下载 SanD/HSSD、生成 HSSD 观测并编译唯一训练集：

```bash
scripts/build_dataset.sh /path/to/data-root
```

该入口面向空的数据目录执行一次；内部阶段不提供历史版本、恢复模式或已有输出分支。训练只读取最终的 `data/policy_dataset`。

测试、过拟合、训练与离线评估：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ../.venvs/curvenav/bin/python -m pytest -q -p no:cacheprovider
GPU_ID=0
GPU_IDS=0,1
CUDA_VISIBLE_DEVICES="${GPU_ID}" scripts/overfit_policy.sh configs/base.yaml
CUDA_VISIBLE_DEVICES="${GPU_IDS}" scripts/train_policy.sh configs/base.yaml
CUDA_VISIBLE_DEVICES="${GPU_ID}" scripts/evaluate_policy.sh configs/base.yaml outputs/train_policy/checkpoint.pt
```

## 目录

```text
configs/base.yaml          唯一模型与训练配置
configs/hssd_dataset.json  唯一 HSSD 生成配置
scripts/build_dataset.sh   唯一数据构建入口
src/curvenav/data/         标定深度、统一数据编译与 loader
src/curvenav/data_generation/ HSSD 资产、几何、生成与正式审计
src/curvenav/encoders/     深度与 PointGoal 编码
src/curvenav/conditioning/ 多帧视觉压缩、逐帧位姿与目标融合
src/curvenav/models/       flow、轨迹兼容度评分器与 policy
src/curvenav/trajectory/   B-spline 和几何
src/curvenav/training/     DDP、AMP、EMA 与 checkpoint
src/curvenav/evaluation/   固定离线门禁
src/curvenav/deployment/   多帧观测状态与严格推理接口
tests/                     数学、数据、模型和部署合同
```

仓库不保存生成结果、权重或日志，也不提供旧协议分支。
