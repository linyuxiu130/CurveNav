# 专家数据

当前六帧空间记忆模型使用原生 14 维 B-spline 专家标签。采集规则见
[BSPLINE_MEMORY](../docs/BSPLINE_MEMORY.md)，统一入口是
`python -m curvenav.data.prepare_fixed40`。

每条独立轨迹通过 `--partition train` 或 `--partition validation` 指定归属。
按实际运动采集并保留所有状态；目标、完成条件和任务失败定义沿用各任务本身。
源地图只供离线专家与标签使用，策略只接收真实因果观测和可见目标。
当前训练默认从 `curvenav_expert/dataset_summary.json` 读取清单；更换数据阶段保留模型、优化器、EMA和RNG，重新计epoch。
CPU 中间采集须经 `prepare_six_frame_replay` 在 CUDA 上重建记忆后才能用于训练。

唯一维护的数据目录为 `curvenav_expert/`，后续采集直接更新该数据集。训练接入使用其中通过验收后
发布的 `dataset_summary.json`；`HANDOFF.txt` 记录接入说明。原 194 条验证状态固定，
新增轨迹只进入训练集。该验证仍为原40任务内留出状态，不代表未见场景泛化。

`captures/` 保存真实执行、原始深度和专家标签，`native/` 保存 CUDA 重放后的训练输入，
`audit/` 保存准入、覆盖和当前验收结果；失败原因统一记在 `audit/maintenance.json`。
CPU 采集不保存可由原始观测和写入选择重建的记忆计划；训练所需计划保留在 `native/`。
清理记录与完整加载结果见 `audit/maintenance.json`。
数据、场景及模型权重留在本地，不提交 Git。

`inputs/` 仅保留任务、标定、采集计划和采集模型。所有可运行的采集与准备代码都维护在
`src/curvenav/data/`。准备器会用同一模型重建 CUDA 记忆和记录候选，并更新来源哈希。
`provenance/native_preparation.py` 是已准入数据所绑定的原始源码证据，不作为运行入口；
每条轨迹清单绑定的源码快照保留原路径，相同内容共享存储。部分原数据因冻结验证的绝对路径绑定保留在 `outputs/`。
