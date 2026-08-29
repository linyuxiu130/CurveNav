# CurveNav

CurveNav 是 PointGoal 条件二维局部规划器。模型读取三帧过去深度与一帧当前深度、对应逐帧相对变换和当前 PointGoal。当前帧经 SanD 风格 ResNet 形成 96 个视觉 token；四帧标定深度统一反投影、坡度分类并配准成连续机器人配置空间场，三个因果 SE(2) token 描述近期运动。生成器在标准化正弧长与七个局部航向增量中学习 boundary-complete improved MeanFlow：随机高斯用于训练，固定典型 latent 用于确定性 1-NFE 推理。每次求值把状态解码为 16 个物理路径 token，并连续查询 signed clearance、梯度、可见性与禁行占据。弧长不与 PointGoal 距离硬绑定，目标距离不截断，航向不经过 clip 或 `tanh`。不存在 ODE solver、候选集、评价头或推理修补。

当前目标只有一个：先在现有深度合同下验证 PointGoal 局部规划，再在完全相同的 episode、相机、异步 MPC 和指标口径下对比 NavDP 与 X-NavDP。局部基线成立前不专项扩展长距离或脱困能力。

架构与训练合同见 `ARCHITECTURE.md`，评测合同见 `EVALUATION.md`。

## 唯一工作流

项目环境：

```text
../.venvs/curvenav
```

从固定上游 commit 下载 HSSD、用官方 Dingo 相机生成观测并编译唯一训练集：

```bash
scripts/build_dataset.sh
```

该入口面向空的数据目录执行一次；内部阶段不提供历史版本、恢复模式或已有输出分支。训练只读取最终的 `data/policy_dataset`。

当前唯一相机合同与 X-NavDP Dingo 测评相机一致：参考深度内参为
`640×360, fx=fy=326.39856, cx=320, cy=180`；benchmark 可在保持光圈与 FoV
不变时等比例采样为 `320×180`，此时内参随宽高解析缩放。两者都严格重投影到
`224×126, fx=fy=166.80851`；相机在机器人系前向
`0.28618 m`、高度 `0.62532 m`、下俯 `10°`。HSSD 直接按该外参渲染；数据、
模型反投影与 checkpoint 共用该合同。旧 SanD `0/0.40/0°`
深度不再进入训练，因为单张深度无法无损改造成不同视点。

测试、训练与离线评估：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ../.venvs/curvenav/bin/python -m pytest -q -p no:cacheprovider
GPU_IDS=0,1
CUDA_VISIBLE_DEVICES="${GPU_IDS}" scripts/train_policy.sh configs/base.yaml
CUDA_VISIBLE_DEVICES=0 scripts/evaluate_policy.sh configs/base.yaml outputs/train_policy/checkpoint.pt
```

训练固定全局 batch 为 1024、每卡 micro-batch 上限为 256；1–8 张 GPU 都保持每次更新严格覆盖 1024 个不重复样本以及相同的 8000 个优化器更新。不能整除时只允许相邻 rank 相差一个样本，并按样本数缩放 loss 后再做 DDP 平均。每个 rank 的份额均衡拆成相同数量的 micro-batch；当前三卡训练为 `342/341/341`，各拆成两个约 B171 的前后向；四卡时每 rank 直接使用 B256。当前 prepared dataset 只包含我们在固定 HSSD 资产上生成的 Dingo 深度与专家轨迹，不混入 SanD/NavDP 数据。

## 目录

```text
configs/base.yaml          唯一模型与训练配置
configs/hssd_dataset.json  唯一 HSSD 生成配置
scripts/build_dataset.sh   唯一数据构建入口
src/curvenav/data/         标定深度、统一数据编译与 loader
src/curvenav/data_generation/ HSSD 资产、几何、生成与正式审计
src/curvenav/encoders/     深度与 PointGoal 编码
src/curvenav/conditioning/ 目标、当前视觉与因果运动状态融合
src/curvenav/models/       有序曲线 Transformer 与 policy
src/curvenav/trajectory/   专家/推理共用的正弧长度量航向场几何
src/curvenav/training/     DDP、AMP、EMA 与 checkpoint
src/curvenav/evaluation/   固定离线评测
src/curvenav/deployment/   多帧观测状态与严格推理接口
tests/                     数学、数据、模型和部署合同
```

仓库不保存生成结果、权重或日志，也不提供旧协议分支。
