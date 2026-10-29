# 当前代码

| 链路 | 实现 |
|---|---|
| 原生模型、检查点、候选生成 | `deployment/bspline_bundle.py`、`models/policy.py` |
| 六秒采样、持久记忆写入与来源记录 | `data/six_frame_memory.py` |
| 当前 CNN 特征与时序／几何融合 | `training/bspline_image_batch.py`、`deployment/streaming_memory.py` |
| 单路几何场与环境 token | `encoders/temporal_geometry.py` |
| 独立专家采集、连续几何资格 | `data/prepare_fixed40.py`、`data/bspline_teachers.py`、`data_generation/geometry.py` |
| CUDA 原生记忆重放 | `data/prepare_six_frame_replay.py` |
| 数据入口、加载准入、任务与专家分组 | `data/bspline_dataset.py` |
| CFM／TRAJ／GF／LH／LQ | `training/bspline_objective.py`、`training/critic.py` |
| 任务均衡采样、预算、优化器恢复 | `training/bspline_replay.py` |
| 同权重、同口径训练／验证监控 | `training/convergence.py` |
| 原 40 案、全向执行与传感器标定 | `evaluation/run_bspline40.py`、`evaluation/bspline40.py`、`evaluation/bspline_execution.py`、`evaluation/bspline_scenarios.py` |
| Habitat 相机与位姿工具 | `data_generation/generate.py` |

路径均相对于 `src/curvenav/`。`factory.py` 直接构建当前网络，训练与在线推理共用记忆融合、候选生成和评分实现。数据准备器同时重建 CUDA 记忆与记录候选，原始观测和合格专家标签保持不变。
