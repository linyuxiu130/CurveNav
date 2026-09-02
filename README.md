# CurveNav

CurveNav 是 PointGoal 条件二维局部轨迹生成器。它读取三帧历史深度、一帧当前深度、
逐帧相对 SE(2) 变换和当前 PointGoal，以一次 improved MeanFlow 调用生成唯一平滑
B-spline。
四帧全部经过同一个 learned depth encoder，并在当前机器人坐标系中融合为目标无关的
observed C-space visual BEV。一次网络调用内，瞬时速度场先从目标走廊和场景记忆估计
clean 轨迹；平均速度场随后沿这条估计轨迹直接查询 observed C-space，并以候选控制点到
局部目标参考的相对位移保持任务方向。PointGoal 不决定第二次障碍查询位置，也不被直接
加到输出轨迹。最终仍只执行 `e* - u(e*,0,1,c)` 一次更新；不存在候选集、评分头或在线
迭代求解器。模型直接生成专家物理 B-spline 控制增量。
原始 Dingo `navigation_grid.npz` 只用于数据编译和物理安全评测，绝不进入训练
或部署网络。安全 query 固定为端点包含、最大 `0.025m` 间隔的原生 cell lookup，OOB
一律不可执行。

架构、数学和安全合同见 [`ARCHITECTURE.md`](ARCHITECTURE.md)；离线和在线评测合同见
[`EVALUATION.md`](EVALUATION.md)；测评吞吐与场景复用见
[`EVALUATION_ACCELERATION.md`](EVALUATION_ACCELERATION.md)。项目不存在候选评价、推理碰撞投影、通用 ODE solver、旧模型
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
下俯 `10°`。HSSD 数据、模型反投影与 checkpoint 共用该合同；部署 reset 必须传入模拟器
实际内参，并由同一预处理重投影到训练相机，不使用硬编码 fallback。

测试、训练与离线评估：

```bash
source scripts/common_env.sh
PYTHONDONTWRITEBYTECODE=1 "${CURVENAV_PYTHON}" \
  -m pytest -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_training_runtime.sh \
  scripts/train_policy.sh configs/base.yaml
CUDA_VISIBLE_DEVICES=0 scripts/evaluate_policy.sh \
  configs/base.yaml outputs/train_policy/checkpoint.pt
```

训练固定全局 batch 1792、200 epoch/4600 optimizer step；所有 GPU 数都按真实样本数缩放
DDP loss，保持全局均值。四卡为 `448×4`，单卡微批上限为 448，避免无意义的梯度累积。
神经算子在支持 BF16
的 GPU 使用 BF16；V100 在同一代码路径使用 FP16 + GradScaler。标定几何、Flow 状态、
MeanFlow/JVP、B-spline 解码和损失始终使用 FP32。运行时
包装在独立 PID namespace 中执行唯一训练入口；强制
结束其 tmux 会话不会留下 DDP rank。

正式训练前必须在目标机器用同一命令做一个编译后稳定区间的吞吐 smoke；吞吐只报告
完整 optimizer step 的全局 samples/s，不把首次静态编译计入。显卡拓扑若不支持可靠的
peer-DMA，可只通过 NCCL transport 环境变量选择 SHM，不改变模型或训练入口。

## 目录

```text
configs/base.yaml          唯一模型、数据根和训练配置
scripts/build_dataset.sh   唯一数据构建入口
src/curvenav/data/         深度、source C-space query、编译与 loader
src/curvenav/data_generation/ HSSD 资产、几何、生成与审计
src/curvenav/encoders/     四帧共享深度、observed C-space 与 metric BEV
src/curvenav/conditioning/目标无关场景记忆、PointGoal 度量查询与历史状态
src/curvenav/models/       MeanFlow trajectory Transformer
src/curvenav/trajectory/   专家/推理共用的 B-spline 坐标与重采样
src/curvenav/training/     DDP、AMP、EMA 与 checkpoint
src/curvenav/evaluation/   source-consistent 离线与跨模型评测
src/curvenav/deployment/   严格单步推理接口
tests/                     数学、数据、模型和部署合同
```
