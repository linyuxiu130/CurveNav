# CurveNav

CurveNav 是面向二维局部导航的 PointGoal-conditioned Rectified Flow。当前唯一训练链为 `v3`：从每条
SanD 专家 run 随机采样未来点作为 PointGoal，并监督机器人到同一未来点的完整可变长度路径。模型不使用
固定 2.1 m 轨迹，也不把预测终点硬投影到 PointGoal；B-spline/Flow 只固定机器人原点。

当前结构与数学协议见 [ARCHITECTURE.md](ARCHITECTURE.md)，实验结果、失败经验和 keep/discard 决策见
[EXPERIMENTS.md](EXPERIMENTS.md)。

```text
4-frame depth + sampled PointGoal + executed motion
  -> condition Transformer
variable-length expert local path
  -> 12-control planar cubic B-spline
PointGoal-directed prior + origin-conditioned RBF-GP
  -> Rectified Flow + 8-step Euler
  -> learned metric path / heading / curvature
```

## 当前固定合同

- `depth [B,4,1,168,224]`，从旧到新。
- `task_goal [B,2]`，采样未来点在当前机器人坐标系中的米制 XY。
- `motion_context [B,3] = [executed_unit_dx, executed_unit_dy, valid]`。
- target 为真实可变长度专家局部前缀，拟合成 `control_points [B,12,2]`。
- target 的末端对应 PointGoal；预测 `Q11` 仍属于完整 Flow 随机变量，不做终点硬投影。
- 不存在 2.1 m 或其他固定输出弧长。
- 推理固定 8-step Euler，并使用 EMA checkpoint。
- checkpoint format 为 11；旧 format-10/更早权重不会静默加载到 v3。

## 目录

```text
configs/                 唯一训练配置与数据配方
src/curvenav/
  data/                  packed depth bank 与 batch contract
  encoders/              depth / PointGoal / motion encoder
  conditioning/          condition Transformer
  models/                policy 与 trajectory field
  generative/            origin-fixed Rectified Flow
  trajectory/            planar B-spline、normalization、geometry
  training/              DDP/AMP、EMA、optimizer、checkpoint
  evaluation/            held-out trajectory evaluation
  deployment/            history、candidate selector、runtime
  data_generation/       独立 privileged expert data pipeline
scripts/                 数据、训练、评测与传输工具
tests/                   数学和端到端 contract 测试
```

## 环境

固定 Python 环境：

```text
/mnt/data/huangshibo/H/navigation_three_projects/.venvs/curvenav
```

SanD 公开数据：

```text
/mnt/data/huangshibo/H/navigation_three_projects/datasets/sandplanner
```

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
  ../.venvs/curvenav/bin/python -m pytest -q -p no:cacheprovider
```

正式训练前必须先通过固定单批过拟合：

```bash
mkdir -p outputs/train_v3_sand_goal_aligned
scripts/overfit_sand.sh configs/train_sand_official.yaml | \
  tee outputs/train_v3_sand_goal_aligned/overfit.log
```

## 训练

训练使用唯一双卡 DDP/FP16/compiled policy/fused AdamW/EMA 链。当前宿主的 NCCL P2P collective 已由
最小复现确认会自旋，因此启动脚本固定使用验证通过的 SHM collective。配置和 checkpoint contract 均为
v3/format-11；旧 checkpoint 不能用于 `--resume`。

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/train_sand.sh configs/train_sand_official.yaml
```

当前通过离线门禁的 format-11 EMA checkpoint：

```text
outputs/train_v3_sand_goal_aligned/checkpoint.pt
SHA256 410f184e64d0b25b5ad13efbeefb03fa45880376f981ae092b1d3787ae922b86
```

## 评测闭环

开发比较使用 9998 上固定的 100 episodes。目标是同协议达到 NavDP，而不是优化单一训练 loss。每个新
checkpoint 必须经历：contract/EMA 检查、1-episode smoke、固定 quick-100、matched failure 与 oracle
candidate diagnosis。

benchmark 现在只接受 `[B,H,W,1]` float32 米制 raw depth，NaN 保留，并只返回 NPZ。旧的有损请求链和
其结果已删除，不属于当前基线。数据生成保存同语义的无损物理深度。

旧 format-10 及更早权重不兼容当前源码，也不能作为 resume checkpoint。相关结论只保留在实验记录中。
