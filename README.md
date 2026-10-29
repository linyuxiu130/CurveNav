# CurveNav

唯一维护链路：真实深度与位姿 → 六帧采样写入持久空间记忆 → 16 条原生 B-spline 候选 → Q 选路 → 理想全向几何执行器。当前执行保持 yaw，每 0.5 秒最多走 0.25 米。

模型、监督和梯度公式见 [BSPLINE_MEMORY](docs/BSPLINE_MEMORY.md)，代码位置见 [CODE_MAP](docs/CODE_MAP.md)。源地图只用于离线专家、监督和测评。

## 数据与统计

唯一数据入口为 `data/curvenav_expert/dataset_summary.json`：3019 条训练状态、原 194 条验证状态。采集规则与验收见 [数据说明](data/README.md)。原验证文件及其哈希固定不动；部分原始文件保留原路径以维持这些绑定。

40 个任务逐任务等权，任务内部循环全部状态：K/L/D2 各 20%，J 40%。每个 epoch 完整训练集与验证集使用同一份 EMA、相同任务权重和固定评估噪声。逐步 raw loss 只表示优化时当前 batch 的数值。

全局 batch 176 时，每 epoch 为 110 步，覆盖最长任务的 482 条状态。配置为 50 epoch，即 5500 步；这是任务均衡后的完整覆盖周期，并非按原始行数直接除 batch。

## 运行

激活当前环境 `/shibo_huang/envs/curvenav-pcard`，在仓库根目录执行：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

训练与测评分别只有 `scripts/train_policy.sh` 和 `scripts/evaluate_policy.sh`；对应安装入口为 `curvenav-train`、`curvenav-evaluate`。显式设置已分配的 `CUDA_VISIBLE_DEVICES`；训练启动器使用 8 个进程。

新数据续训保留模型、AdamW、EMA 和 RNG，新数据 epoch 从 0 计；学习率从检查点保存的当前值继续余弦衰减，不重复预热：

```bash
bash scripts/train_policy.sh \
  --bundle outputs/supervision_audit_20261029/training_throughput/checkpoints/epoch_044/model.pt \
  --resume outputs/supervision_audit_20261029/training_throughput/checkpoints/epoch_044/training_state.pt \
  --new-data-stage --batch-per-rank 22 --global-batch 176 --output outputs/training
```

训练结束后，对该训练输出的 EMA 执行原 40 案：

```bash
bash scripts/evaluate_policy.sh --checkpoint outputs/training/ema_model.pt --output outputs/evaluation
```

测评默认使用数据目录内的原任务、标定与 Habitat 运行配置。输出目录必须新建，不能覆盖已有检查点或成绩。

采集入口为 `python -m curvenav.data.prepare_fixed40`；CPU 原始采集经 `python -m curvenav.data.prepare_six_frame_replay --device cuda` 准备后才进入训练。初始化使用 `python -m curvenav.deployment.bspline_bundle --config configs/base.yaml --output NEW.pt`。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```
