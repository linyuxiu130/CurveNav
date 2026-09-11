# CurveNav 测评加速方案

本文件是 CurveNav 仓库唯一的测评加速方案。它只优化数据装载、GPU
批处理和场景生命周期，不改变 PointGoal 协议、控制器、成功阈值、SPL
公式或模型权重。测评合同和指标定义仍以
[`EVALUATION.md`](EVALUATION.md) 为准。

## 当前事实

- 离线测评只有 `curvenav.evaluation.offline` 一条入口；跨模型比较只有
  `curvenav.evaluation.compare` 一条入口。
- validation loader 使用固定 batch `32`，深度通过 `CudaPrefetchLoader`
  异步搬运到 CUDA；每个 observation 只执行一次确定性的单步 MeanFlow
  采样。
- native `navigation_grid.npz` 是唯一物理安全真值。source grid 和 depth
  bank 在启动时加载，source 查询按 batch 向量化；BEV 图和案例 SVG 在数值
  汇总之后一次性写出，不进入模型热路径。
- 在线测评由常驻 scene evaluator 复用已加载场景；每个模型只替换 policy
  服务，不重复构建同一资产。在线终止条件和 MPC 仍由固定 benchmark 实现。
- 当前训练配置为每卡 416 样本、累积 1 次、200 epoch；全局 batch 由
  每卡微批、GPU 数和累积次数计算。每 epoch 的样本预算向下取完整全局批次，
  实际更新次数写入训练日志和 checkpoint。
- 评测输出只保留 `offline-metrics.json`、`offline-cases.json` 和
  `offline-cases.svg`（或调用方指定的等价 artifact 目录），不生成临时
  checkpoint、第二套 launcher 或缩减版成绩。

## 唯一命令

在仓库根目录执行。`CUDA_VISIBLE_DEVICES` 必须只指向空闲设备；训练和测评
不得抢占其他任务。

```bash
# 代码与数学合同
source scripts/common_env.sh
PYTHONDONTWRITEBYTECODE=1 "${CURVENAV_PYTHON}" \
  -m pytest -q -p no:cacheprovider

# 单模型完整离线测评（唯一入口）
CUDA_VISIBLE_DEVICES=1 scripts/evaluate_policy.sh \
  data/policy_dataset-depth-v2/config.yaml outputs/train_policy-depth-forward/checkpoint.pt \
  --artifact-dir outputs/offline-evaluation

# 使用同一 common protocol 的跨模型比较
scripts/compare_offline.sh \
  configs/base.yaml \
  outputs/evaluation/common.npz \
  outputs/hssd_policy_dataset \
  outputs/evaluation/compare.json \
  outputs/evaluation/curvenav.npz \
  outputs/evaluation/sand.npz \
  outputs/evaluation/navdp.npz \
  outputs/evaluation/xnavdp.npz
```

正式成绩必须运行完整 validation split；`quick100`、随机子集和中途切换
源码只能用于调试，不能写入固定结果目录。批量吞吐以完整 batch 的总样本数
除以稳定区间墙钟时间，首轮 CUDA/Inductor 编译时间单独记录；延迟另用
batch-1 统计，不把两种数字混在一起。

## 加速原则

1. 数据只构建一次：使用已通过 source certificate 的 prepared dataset，
   复用 memory-map 深度 bank 和 native source grid；不在每个模型、每个
   case 或每次可视化时重新构建场景。
2. 模型只走一条 GPU 批处理路径：固定 batch、预取、单次 forward；不要
   添加候选集、评分器、后处理投影或 fallback 来“加速”。
3. 几何指标按 batch 计算并缓存中间查询；案例筛选和绘图放在 forward 之后，
   只处理紧凑案例集。
4. 在线使用一个常驻场景和一个统一 server/adapter 接口。多卡只按 scene
   shard 并行；同一 scene 不启动多个冷实例。当前 PointGoal B16 正式协议
   使用 `num-envs=16`，其他 batch 只用于诊断，不能混入正式结果。
5. 每次运行使用新的 artifact 目录，日志和指标不覆盖旧结果；不在运行中途
   修改代码或切换权重。

## 已知基准与验证

历史实验 E002 记录了同一评测链路的参考值：完整 6,064 条 validation
观测约 35.32 s，base-policy batch-32 约 920.7 observations/s，batch-1
模型延迟 P50 约 29.28 ms。这些数值是硬件和 checkpoint 相关的基准，不是
固定协议成绩；新的硬件报告必须同时给出 batch、worker 数、GPU 型号、稳定
区间和完整 artifact 路径。

2026-09-02 本机 9999 的四张 V100S 在 `NCCL_P2P_DISABLE=1` 下使用
`global_batch=1792 (448×4)` 实测稳定 `2766–2785 samples/s`，每卡显存约
`27.6 GiB/32 GiB`、利用率 `94–100%`。此前 `1368 (342×4)` 为约
`2690 samples/s`；第一次启用 P2P 的启动因 NCCL `ALLGATHER` 超时失败，
之后使用已验证的 SHM 传输成功。首轮静态 CUDA/Inductor 编译耗时单独计入
启动记录，不计入上述稳定吞吐。

本次整理完成后应至少验证：

- `bash -n scripts/*.sh`；
- `python -m compileall -q src tests`；
- 全量单元/合同测试；
- offline 模块 `--help`、唯一 launcher 的参数解析和一次最小 smoke；
- 离线数值输出与 artifact 文件彼此一致。

本次整理（2026-09-02）已完成 `bash -n scripts/*.sh`、
`compileall -q src tests`、训练/离线/比较入口 `--help`，以及全量合同测试：
`116 passed, 2 skipped`。两个跳过项仅因为当前虚拟环境没有 matplotlib，
不涉及训练、推理或几何数值合同。

## 维护记录

数据、训练与测评脚本共享 `scripts/common_env.sh` 的 Python、
TorchInductor 和 `PYTHONPATH` 初始化。新增加速必须先证明它不改变上述
测评合同，并在本文件补充命令、硬件、稳定吞吐、延迟和失败根因；不得另建
计划文档或第二条运行链路。
