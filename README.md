# CurveNav

CurveNav 是 PointGoal 条件的米制局部轨迹生成器。当前输入为深度图、逐帧内外参、
重力对齐的机体位姿和时间戳；四帧共享视觉骨干与 SE(3) 几何对齐构建 BEV，
局部世界坐标体素记忆保留已观测障碍。
生成器使用共享的有目标/无目标两步 Flow Matching，合成 32 条结构化 B-spline 候选，独立的单头路线评价器
选取最高分的一条，由固定测评 MPC 执行。
传感器历史与异步推理分离；当前二维观测场不提供绝对无碰撞保证。

唯一新观测契约拒绝旧深度数据、旧请求与旧 checkpoint，没有兼容或旧输入替代分支。
架构、数学、输入定义与已验证边界见 [ARCHITECTURE.md](ARCHITECTURE.md)，
重构原因见 [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)。源配置空间真值只用于
标签安全证书、评分头监督与测评，不进入在线场景记忆。

## 三条代码链路

主分支为 `main`。三条链路按 Python 包目录组织，运行脚本集中在 `scripts/`：

| 链路 | 实现目录 | 运行入口 | 产物 |
| --- | --- | --- | --- |
| 数据生成 | [data_generation](src/curvenav/data_generation/)；[data](src/curvenav/data/) 负责标签编译和读取 | [build_dataset.sh](scripts/build_dataset.sh)；资产下载用 [download_hssd_assets.sh](scripts/download_hssd_assets.sh) | 深度帧、专家路线、训练标签和拟合统计配置 |
| 训练 | [training](src/curvenav/training/) | [train_policy.sh](scripts/train_policy.sh) | 模型、EMA、优化器和恢复状态 checkpoint |
| 测评 | [evaluation](src/curvenav/evaluation/)；[deployment](src/curvenav/deployment/) 提供在线策略接口 | [evaluate_policy.sh](scripts/evaluate_policy.sh)；跨模型离线比较用 [compare_offline.sh](scripts/compare_offline.sh) | 离线指标、案例报告和比较结果 |

运行顺序：生成数据并完成审计 → 使用生成目录中的 `config.yaml` 训练 → 使用同一配置和
checkpoint 测评。三条链路共用 `encoders/`、`conditioning/`、`models/`、`trajectory/`
中的模型与几何实现。详细定义见 [架构说明](ARCHITECTURE.md) 和 [测评说明](EVALUATION.md)。

## 本地目录分工

唯一项目根目录为 `/shibo_huang/CurveNav`，不再保留独立 review 或旧测评仓库。

| 目录 | 分工 |
| --- | --- |
| `src/`、`configs/`、`scripts/` | 当前数据生成、模型和训练代码 |
| `data/` | HSSD 资产与当前训练数据 |
| `online_evaluation/` | 按模型组织的在线测评代码 |
| `online_evaluation/assets/`、`weights/`、`.runtime/` | 测评场景、基线权重与固定运行时，均在 online_evaluation 下 |
| `outputs/` | 当前专家源轨迹、训练 checkpoint、测评结果和实验依据 |
| `backups/recovery/` | 一份当前环境恢复包与恢复脚本 |

## GitHub 上传范围

本仓库已上传上述三条链路的源码、配置、入口脚本、测试和中文文档。
数据集、场景资产、模型权重、运行结果、Conda 环境和编译缓存保留在本地，由
[.gitignore](.gitignore) 排除。

`behavior_adapter/` 包含 R1Pro 数据采集、训练入口和独立仿真查看器，见
[适配器说明](behavior_adapter/README.md)。OmniGibson 的两处本地修复以
[补丁和重建脚本](behavior_adapter/vendor/README.md) 保存，运行前须准备对应的
BEHAVIOR v3.9.2 运行时。外部参考仓库 `Offroad-Path-Planning/`、`overseec/`
保留在本机，不作为 CurveNav 源码上传。

本仓库在线轨迹由模型生成并评分。数据生成及离线监督中的最短路工具用于历史
训练标签，不是在线导航的路径生成器；后续记忆探索实验单独在实验分支进行。

在线仿真测评的调度器、模型适配器、固定题目与运行时准备脚本也已纳入
[online_evaluation/](online_evaluation/README.md)，支持 SanD、NavDP、X-NavDP 和
CurveNav 接入，继续使用 `python -m navbench` 入口。各模型的启动适配器独立位于
`online_evaluation/navbench/adapters/`；NavDP 与 XNavDP 共用 `navbench/vision/` 视觉骨干。
本机测评缓存位于 `/shibo_huang/data/curvenav/cache/navbench`。
下文 `/shibo_huang/` 路径和 `outputs/` 验证记录均为本机位置，并非 GitHub 附件。

## 运行流程

重启后恢复：`bash /shibo_huang/CurveNav/scripts/restore_environment.sh`。
备份目录、资产位置和兼容性约束见 `/shibo_huang/CurveNav/backups/recovery/README.md`。

本地训练和测评共用 `curvenav-unified`（Python 3.11、PyTorch 2.7.0 / CUDA 12.6）：

```bash
cd /shibo_huang/CurveNav
source /opt/conda/etc/profile.d/conda.sh
conda activate curvenav-unified
python -m pip check
```

使用前执行 `conda activate curvenav-unified`。测评仓库的 `config/local.env`
已将策略服务和测评进程指向同一个 Python。
`scripts/common_env.sh` 使用当前 Conda 环境的 Python 和编译缓存；Python 头文件由
Conda 提供，系统需有 GCC。PyTorch 安装版本参见
[官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v260)。

实验结果保存在本地 `outputs/`，按模型、场景和运行批次分别统计；
旧批次、未完成批次和当前正式批次不能合并。在线成绩应同时记录权重和跟踪器版本。
数据生成以输出目录的 `audit/summary.json` 通过为完成标志，
准备目录使用实际拟合的 `config.yaml`。

在包含 Habitat-Sim 的已激活环境中，直接生产训练格式并编译标签：

```bash
scripts/build_dataset.sh
```

生成器读取 `configs/base.yaml` 的训练尺寸和深度编码，按 10 Hz / 0.1 s
采集；原生渲染仅驻留内存，经相同在线预处理直接顺序写入 224×126
深度 FP16。四帧窗口只存索引，不复制图像。没有原生帧落盘或独立缓存转换阶段。
默认保留 20 场景，每场景近/中/远路线数为 5/10/10，共 500 条（400 训练、100 验证），不设容量配额。
需要小批生产时，给 `scripts/build_dataset.sh` 传入配置，显式选择场景和路线配额；
保留训练/验证场景隔离。生成完成后审核全部帧、轨迹与训练标签。

新增 GRScenes/X-NavDP 59 场景使用同一生成器：

```bash
scripts/prepare_grscenes.sh configs/grscenes_dataset.json --seven-zip /shibo_huang/data/curvenav/cache/tools/7zip/7zz
scripts/build_dataset.sh configs/grscenes_dataset.json
```

资产准备复用已安装的 Isaac USD 运行时、`trimesh` 和共享模型库；分卷解压需要 `7zz`。
只导出场景几何，按 USD 单位、完整实例变换和轴约定转换为 Habitat 资产；不复制纹理。
官方 59 个训练场景中配置 49 个训练、10 个验证，共 6,000 条完整路线（训练 5,000、验证 1,000）。
`routes_per_split` 设置总数，`endpoint_sampling.bands` 设置距离区间、机器人坐标系下的目标方位角和权重；各场景数量最多相差一条。
按场景动态调度 8 个独立进程，在 GPU 1 渲染；日志逐条记录耗时、缓存命中和场景进度。
验证组按场景 ID 前 15 位保守分组（9 个住宅、1 个商业变体）；这不是独立户型数的保证。
官方 40 个测评场景不参与生产。近/中/远按每场景安全端点距离分位数定义，避免小场景
被固定米制距离排除。新采样分布与旧 500 条不完全相同，效果需单独实验比较。

`configs/grscenes_near_goal_dataset.json` 在同一生成链路补充 980 条 0.5–1.5 m
侧后方目标路线，只使用上述 49 个训练场景。目标相对方位在规划重试前固定，
每条路线重新规划和渲染；`base_route_roots` 合并原专家数据，保留原验证任务。
距离单位可选 `quantiles`（全场景覆盖）或 `metres`（定向补充），均复用相同的专家、审核和训练格式编译。
初始世界朝向独立均匀采样，并作为样条起始切向的边界条件；端点重采样不改变朝向。
端点与整条路线使用相同的本体膨胀后安全余量。平滑代价为
`∫ [1 + exp(1-d/r) + r²κ²] ds`，其中 `r` 是本体半径，`d` 是配置空间余量；
曲率是软代价，不设额外最小转弯半径。长度项使用分段高斯积分，曲率项自适应积分。

场景网格、导航网格与已完成路线缓存在 `/shibo_huang/data/curvenav/cache/expert-routes`，
缓存键包括生成代码和几何/传感器契约。增加场景或分档配额时复用已有路线；路线 ID
为“分档＋档内序号”。生成目录通过硬链接引用不可变路线，重启后再次执行同一配置即可
复用已完成内容。发布目标必须是新目录；若只需重做标签，使用 `curvenav-prepare-data --route-root ... --output ... --config ...`。合并数据源时重复传入 `--route-root`，编译器会在联合训练集上拟合 flow 坐标尺度。训练使用新数据目录生成的 `config.yaml`。
逐轨迹障碍记忆缓存在 `$XDG_CACHE_HOME/curvenav/route_memory`；仓库启动脚本默认使用 `/shibo_huang/data/curvenav/cache`。首次计算后重复编译可复用，深度或位姿文件修改、相机标定和记忆实现变化会生成新的缓存键。联合统计更新无需重新计算这部分几何；标签拟合和最终数组写入仍会执行。

当前训练读取 `/shibo_huang/data/curvenav/datasets/policy_hssd_grscenes7480`，保存深度帧索引、专家曲线、逐样本内外参、
SE(3) 相对位姿、观测年龄、因果障碍记忆和 source provenance。写盘后必须通过 source re-query certificate；
Flow 坐标统计只从训练集拟合，写入准备目录的 `config.yaml`。所有入口使用当前 Conda 环境，不再引用旧虚拟环境路径。

深度 的内参必须对应交付图像分辨率；缩放保持视场并更新 K。每帧实际相机外参参与几何计算，
没有固定相机参数的模型隐藏路径。生成数据和测评均使用相同的光学深度编码。

测试、训练与离线评估：

```bash
source scripts/common_env.sh
PYTHONDONTWRITEBYTECODE=1 "${CURVENAV_PYTHON}" \
  -m pytest -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES=1 scripts/train_policy.sh configs/base.yaml
CUDA_VISIBLE_DEVICES=1 scripts/evaluate_policy.sh \
  configs/base.yaml outputs/train-structured-exploration-7480/best.pt
```

训练配置直接指定 `per_device_batch_size` 和 `gradient_accumulation_steps`，全局 batch
由每卡微批 × GPU 数 × 累积次数计算，默认累积两次。每张卡始终执行完整固定形状的微批，
DDP loss 按全局均值缩放。`samples_per_epoch` 是每个逻辑 epoch 的样本预算，向下取完整
全局批次，实际样本数及更新次数记录在启动日志和 checkpoint 中。改变批次会改变优化器
更新频率；保持样本预算并不等于保持优化过程，学习率和收敛需要另行验证。
结构化探索默认每卡 64、累积 2 次：单卡全局 128，双卡全局 256。
批次显存和吞吐必须按当前评价架构实测，不沿用旧共享净空头的性能结果。
双卡使用同一配置，只需将启动命令中的 `CUDA_VISIBLE_DEVICES` 设为 `0,1`。
默认训练集为合并后的 7,480 条轨迹：1,491,638 个训练状态、260,340 个验证状态；归一化统计来自该训练集。每 epoch 预算 1,491,456 个样本，双卡 5,826 次更新。补充的 980 条近距离侧后方专家由 `configs/grscenes_near_goal_dataset.json` 生产并合并，四个方向各 245 条，验证集不增加补充场景。

每轮验证固定按场景抽取最多 512 个样本，使用 EMA 权重执行部署的 32 条生成与选择。
`best.pt` 按所选轨迹的真实地图效用最大值保存；同时记录碰撞、选择遗憾和轨迹误差。
训练 loss 用于观察拟合过程，不再作为最佳部署模型的选择标准。
训练批次、卡数和累积次数是 checkpoint 的恢复契约，恢复时必须保持一致。
训练和测评的神经算子统一 BF16，要求 GPU 原生支持 BF16。标定几何、Flow 状态、
Flow 积分、B-spline 解码和损失使用 FP32。
数据、训练与离线测评入口共用当前 Conda 环境；训练直接使用 `scripts/train_policy.sh`。 编译工作缓存使用 PyTorch 默认本地临时目录，不再强制指向共享存储中的 Conda 缓存；checkpoint、数据与日志仍写入持久目录。
本地训练停止时应向 torchrun 主进程发送 SIGINT 并等待各 rank 退出；不要强杀 tmux
来代替正常停止。

单独后训练评估头时，冻结生成器，复用部署的 32 条候选及现有地图效用监督：

```bash
python -m curvenav.training.evaluator_finetune configs/base.yaml outputs/train/checkpoints/best.pt \
  --output outputs/evaluator/checkpoint.pt --steps 8192 --batch-size 8
```

配置必须与基础权重对应；`--output` 是文件路径，不能传目录。输出旁的 JSON 记录
训练前后的离线选择指标。后训练权重仍需在线验证，不能仅凭 loss 下降替换正式模型。

正式训练前必须在目标机器用同一命令做一个编译后稳定区间的吞吐 smoke；吞吐只报告
完整 optimizer step 的全局 samples/s，不把首次静态编译计入。显卡拓扑若不支持可靠的
peer-DMA，可只通过 NCCL transport 环境变量选择 SHM，不改变模型或训练入口。

## 目录

```text
configs/base.yaml          数据准备的模型与训练模板
configs/base.yaml  含训练集拟合统计的实际训练配置
scripts/build_dataset.sh   唯一数据构建入口
src/curvenav/data/         深度、source C-space query、编译与 loader
src/curvenav/data_generation/ HSSD 资产、几何、生成与审计
src/curvenav/encoders/     共享 深度、时序几何与 metric BEV
src/curvenav/conditioning/目标无关场景记忆、PointGoal 度量查询与历史状态
src/curvenav/models/       Flow Matching trajectory Transformer
src/curvenav/trajectory/   专家/推理共用的 B-spline 坐标与重采样
src/curvenav/training/     DDP、AMP、EMA 与 checkpoint
src/curvenav/evaluation/   source-consistent 离线与跨模型评测
src/curvenav/deployment/   严格单步推理接口
tests/                     数学、数据、模型和部署合同
```

专家源数据使用前进式差速车曲线时钟：由同一解析样条计算位置、朝向和转弯减速，
按 10 Hz 渲染深度；不再对水平位置进行 NavMesh 修正。训练标签固定前向起始切线与
局部终点，中间控制点最小二乘拟合。控制增量采用每个控制点共享的 XY 尺度，
训练必须使用新数据目录中的统计配置；旧数据/检查点不兼容。
本地测评执行器也改为不倒车，比较成绩时需使用相同执行协议。
