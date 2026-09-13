# CurveNav

CurveNav 是 PointGoal 条件的米制局部轨迹生成器。当前输入为深度图、逐帧内外参、
重力对齐的机体位姿和时间戳；四帧共享视觉骨干与 SE(3) 几何对齐构建 BEV，
局部世界坐标体素记忆保留已观测障碍。
生成器使用两步条件 Flow Matching 输出 B-spline，由固定测评 MPC 执行。
传感器历史与异步推理分离；当前二维观测场不提供绝对无碰撞保证。

唯一新观测契约拒绝旧深度数据、旧请求与旧 checkpoint，没有兼容或旧输入替代分支。
架构、数学、输入定义与已验证边界见 [ARCHITECTURE.md](ARCHITECTURE.md)，
重构原因见 [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)。源配置空间真值只用于
标签安全证书与测评，不进入在线场景记忆。

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

在线仿真测评的调度器、模型适配器、固定题目与运行时准备脚本也已纳入
[online_evaluation/](online_evaluation/README.md)，支持 SanD、NavDP、X-NavDP 和
CurveNav 接入，继续使用 `python -m navbench` 入口。各模型的启动适配器独立位于
`online_evaluation/navbench/adapters/`；NavDP 与 XNavDP 共用 `navbench/vision/` 视觉骨干。
本机测评缓存位于 `/shibo_huang/.cache/navbench`。
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

训练读取 `data/policy_dataset-depth-memory`，保存深度帧索引、专家曲线、逐样本内外参、
SE(3) 相对位姿、观测年龄、因果障碍记忆和 source provenance。写盘后必须通过 source re-query certificate；
Flow 坐标统计只从训练集拟合，写入准备目录的 `config.yaml`。所有入口使用当前 Conda 环境，不再引用旧虚拟环境路径。

深度 的内参必须对应交付图像分辨率；缩放保持视场并更新 K。每帧实际相机外参参与几何计算，
没有固定相机参数的模型隐藏路径。生成数据和测评均使用相同的光学深度编码。

测试、训练与离线评估：

```bash
source scripts/common_env.sh
PYTHONDONTWRITEBYTECODE=1 "${CURVENAV_PYTHON}" \
  -m pytest -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES=1 scripts/train_policy.sh data/policy_dataset-depth-memory/config.yaml
CUDA_VISIBLE_DEVICES=1 scripts/evaluate_policy.sh \
  data/policy_dataset-depth-memory/config.yaml outputs/train_policy-depth-memory/checkpoint.pt
```

训练配置直接指定 `per_device_batch_size` 和 `gradient_accumulation_steps`，全局 batch
由每卡微批 × GPU 数 × 累积次数计算，默认累积一次。每张卡始终执行完整固定形状的微批，
DDP loss 按全局均值缩放。`samples_per_epoch` 是每个逻辑 epoch 的样本预算，向下取完整
全局批次，实际样本数及更新次数记录在启动日志和 checkpoint 中。改变批次会改变优化器
更新频率；保持样本预算并不等于保持优化过程，学习率和收敛需要另行验证。
4090 24 GiB 默认每卡 416、累积 1 次：单卡全局 416，双卡全局 832。
双卡使用同一配置，只需将启动命令中的 `CUDA_VISIBLE_DEVICES` 设为 `0,1`。
默认样本预算下，实际每 epoch 处理 40,768 个样本；单卡 98 次更新，双卡 49 次更新。
训练批次、卡数和累积次数是 checkpoint 的恢复契约，恢复时必须保持一致。
训练和测评的神经算子统一 BF16，要求 GPU 原生支持 BF16。标定几何、Flow 状态、
Flow 积分、B-spline 解码和损失使用 FP32。
数据、训练与离线测评入口共用当前 Conda 环境；训练直接使用 `scripts/train_policy.sh`。
本地训练停止时应向 torchrun 主进程发送 SIGINT 并等待各 rank 退出；不要强杀 tmux
来代替正常停止。

正式训练前必须在目标机器用同一命令做一个编译后稳定区间的吞吐 smoke；吞吐只报告
完整 optimizer step 的全局 samples/s，不把首次静态编译计入。显卡拓扑若不支持可靠的
peer-DMA，可只通过 NCCL transport 环境变量选择 SHM，不改变模型或训练入口。

## 目录

```text
configs/base.yaml          数据准备的模型与训练模板
data/policy_dataset-depth-memory/config.yaml  含训练集拟合统计的实际训练配置
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
