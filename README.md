# CurveNav

CurveNav 是 PointGoal 条件二维局部轨迹生成器。它读取三帧历史深度、一帧当前深度、
逐帧相对 SE(2) 变换和当前 PointGoal，以一次 improved MeanFlow 生成唯一平滑 B-spline。
模型内部使用四帧深度构成局部配置空间条件；原始 Dingo `navigation_grid.npz` 只用于
数据编译和物理安全评测，绝不进入部署网络。安全 query 固定为 `0.025m` 弧长采样、原生
cell lookup，OOB 一律不可执行。`64×64` 局部 raw C-space 是部署输入；它不补全
不可见区域，也不替代 source collision 真值。

架构、数学和安全合同见 [`ARCHITECTURE.md`](ARCHITECTURE.md)；离线和在线评测合同见
[`EVALUATION.md`](EVALUATION.md)。项目不存在候选评价、推理碰撞投影、ODE solver、旧模型
兼容、fallback 或第二条训练/推理链路。

## 唯一工作流

项目环境：

```text
../.venvs/curvenav
```

从已审计、冻结的 HSSD 路线与深度缓存编译唯一训练集：

```bash
scripts/build_dataset.sh
```

当前训练唯一读取路径是 `data/policy_dataset-source-cspace`。该 prepared dataset 保存
Dingo 深度索引、专家曲线、复制的 source grids 和逐样本 anchor provenance；写盘后必须通过
source re-query certificate，且 source-gated train 曲线的
FP64 Flow 坐标统计必须与模型配置一致，loader 才接受它。它不混入 SanD/NavDP 数据。

唯一相机合同与 X-NavDP Dingo 测评相机一致：参考深度内参为
`640×360, fx=fy=326.39856, cx=320, cy=180`；所有输入重投影到
`224×126, fx=fy=166.80851`。相机相对机器人前向 `0.28618m`、高度 `0.62532m`、
下俯 `10°`。HSSD 数据、模型反投影与 checkpoint 共用该合同。

测试、训练与离线评估：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ../.venvs/curvenav/bin/python -m pytest -q -p no:cacheprovider
# 本机 9999 的 V100 peer-DMA 实测不可靠；NCCL 必须走单机 SHM 路径。
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,3 scripts/run_training_runtime.sh \
  scripts/train_policy.sh configs/base.yaml
CUDA_VISIBLE_DEVICES=0 scripts/evaluate_policy.sh \
  configs/base.yaml outputs/train_policy/checkpoint.pt
```

训练固定全局 batch 1024、200 epoch/8000 optimizer step；所有 GPU 数都按真实样本数缩放
DDP loss，保持全局均值。四卡为 `256×4`，单卡微批上限为 342。支持 BF16 的 GPU 使用
BF16；V100 使用同一代码路径下的 FP16 + GradScaler，stopped JVP 始终使用 FP32。运行时
包装在独立 PID namespace 中执行唯一训练入口；强制
结束其 tmux 会话不会留下 DDP rank。

9999 的 GPU 0↔3 已通过 `NCCL_P2P_DISABLE=1` 的双 rank all-reduce；该变量仅选择其
已验证的 SHM 通信传输，不改变模型、优化器或训练入口。

## 目录

```text
configs/base.yaml          唯一模型、数据根和训练配置
scripts/build_dataset.sh   唯一数据构建入口
src/curvenav/data/         深度、source C-space query、编译与 loader
src/curvenav/data_generation/ HSSD 资产、几何、生成与审计
src/curvenav/encoders/     标定深度与观测配置空间编码
src/curvenav/conditioning/目标无关场景编码、PointGoal 与因果状态
src/curvenav/models/       MeanFlow trajectory Transformer
src/curvenav/trajectory/   专家/推理共用的 B-spline 坐标与重采样
src/curvenav/training/     DDP、AMP、EMA 与 checkpoint
src/curvenav/evaluation/   source-consistent 离线与跨模型评测
src/curvenav/deployment/   严格单步推理接口
tests/                     数学、数据、模型和部署合同
```
