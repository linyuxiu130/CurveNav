# 在线仿真测评

本目录收录原 `general-navigation-benchmark` 的在线测评代码，入口保持
`python -m navbench`。数据生成、训练和离线测评仍使用仓库根目录的原有入口。

## 代码与资产

| 目录 | 内容 |
| --- | --- |
| `navbench/` | 场景调度、常驻仿真进程、原始观测传输、指标汇总和轨迹记录 |
| `baselines/` | 模型服务和适配器；CurveNav 仅保留协议接入，共享主仓库模型 |
| `suites/pointgoal-v2.json` | 40 场景、每场景 100 回合及固定输入的 SHA-256 |
| `assets/` | 固定 Dingo 定义、场景列表和 40 份小型起终点数组 |
| `config/` | 版本锁、运行时修改、模型清单和本机配置模板 |
| `scripts/` | 环境准备、权重下载、场景下载与运行时检查 |
| `tests/` | 协议、调度、指标及 CurveNav 接入测试 |

场景模型、导航网格、权重、运行时源码、缓存和结果均外置，不随 Git 上传。
上游代码保留其许可证与来源，见 [NOTICE.md](NOTICE.md)。
`docs/` 保留协议说明和历史实验记录；历史机器路径不是当前运行默认值。

## 使用现有环境

以下命令从本目录执行：

```bash
conda activate curvenav-unified
cd /shibo_huang/CurveNav/online_evaluation
python -m pip install --no-deps -e .. -e .
cp config/local.env.example config/local.env
```

按本机实际位置填写 `config/local.env` 中的 Python、场景、权重、X-NavDP
运行时和 acados 路径。训练和测评可以使用同一个 Conda Python。
当前服务器可继续引用原目录下的外部资产与运行时：

```bash
NAVBENCH_EVAL_PYTHON=/opt/conda/envs/curvenav-unified/bin/python
NAVBENCH_SERVER_PYTHON=/opt/conda/envs/curvenav-unified/bin/python
NAVBENCH_SCENE_ROOT=/shibo_huang/general-navigation-benchmark/assets/scenes
NAVBENCH_WEIGHT_ROOT=/shibo_huang/general-navigation-benchmark/weights
NAVBENCH_XNAVDP_ROOT=/shibo_huang/general-navigation-benchmark/.runtime/x-navdp-878740a20118/baselines/x-navdp
ACADOS_SOURCE_DIR=/shibo_huang/general-navigation-benchmark/.runtime/acados-48e223e85f04
NAVBENCH_CACHE_ROOT=/shibo_huang/.cache/navbench
NAVBENCH_GPUS=1
NAVBENCH_NUM_ENVS=16
```

配置不上传 Git。首次使用 Isaac Sim 的主机需接受 NVIDIA Omniverse EULA 后设置
`OMNI_KIT_ACCEPT_EULA=YES`。

新主机使用 `scripts/setup_evaluator_env.sh` 准备锁定的 Isaac Sim/Lab、acados 和
X-NavDP 运行时。完整统一环境脚本 `scripts/setup_unified_env.sh` 还需要预先将
Python 3.11 的 Magnum、Habitat-Sim wheel 和 `SHA256SUMS` 放到 `.runtime/wheels/`；
这些本地构建产物不包含在仓库中。通用 Python 包可设置
`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`，NVIDIA 与 PyTorch
专用包仍使用安装脚本指定的来源。

## 检查与运行

```bash
python -m pytest -q -p no:cacheprovider tests
python -m navbench --model navdp --gpus 1 --check-assets
python -m navbench --model navdp --gpus 1 --num-envs 16 --dry-run
```

缺少权重时执行 `bash scripts/download_weights.sh navdp sandplanner x-navdp`；
场景准备使用 `bash scripts/prepare_scenes.sh download`，需要对应数据集访问权限。
`--check-assets` 会报告缺失输入；`--dry-run` 仅检查解析后的运行计划，不代表仿真通过。

单模型在线测评：

```bash
python -m navbench --model navdp --gpus 1 --num-envs 16
```

SanD、NavDP、X-NavDP 共用一次场景加载：先创建两个模型清单目录，权重用绝对路径链接，
再交给现有固定模型集合入口。以下准备命令执行一次即可；已有清单直接复用。

```bash
mkdir -p cache/model-artifacts/sandplanner cache/model-artifacts/x-navdp
cp config/artifacts/sandplanner/artifact.json cache/model-artifacts/sandplanner/
cp config/artifacts/x-navdp/artifact.json cache/model-artifacts/x-navdp/
ln -s /shibo_huang/general-navigation-benchmark/weights/sandplanner/NoMax.pth cache/model-artifacts/sandplanner/checkpoint.pt
ln -s /shibo_huang/general-navigation-benchmark/weights/x-navdp/x-navdp_posttrain.ckpt cache/model-artifacts/x-navdp/checkpoint.pt
python -m navbench --model navdp --gpus 1 --num-envs 16 \
  --artifact-bundle cache/model-artifacts/sandplanner \
  --artifact-bundle cache/model-artifacts/x-navdp \
  --output-root runs/baselines-resident
```

每个场景内顺序启动模型服务，模型之间重置环境和策略历史，结果分别保存。
多个场景由现有任务队列调度；Kit/OptiX 缓存按 GPU 隔离，输入缓存使用稳定路径。
中断恢复使用 `--resume-root`，持续接收模型产物使用 `--checkpoint-queue`。

CurveNav 在线服务已包含在 `baselines/curvenav/`。其 `--model-config` 使用显式模型
bundle 的 `configs/config.yaml`，对应源码在该 bundle 的 `src/curvenav/`，
不能直接把任意数据目录中的配置当作模型 bundle。当前三基线批量命令不运行 CurveNav。

## 结果与验证边界

项目完整 B16 测评为每模型 4,000 回合；Home、Commercial 分别汇总。
`metric.csv` 保存逐回合指标，`episodes.csv`、`summary.csv` 保存汇总，紧凑轨迹保存在
`trajectory_traces/`。使用 `--episodes-per-scene 1 --num-envs 1 --scenes split/name`
只能作为单回合诊断，不能当作完整成绩。

本次迁入整理只运行 CPU 合同测试、输入校验和启动计划检查，没有重新运行仿真或训练。
先前服务器上三个基线的单 Commercial 回合测试不能代表完整 40 场景成绩。
本地运行时包含前进式控制约束；比较结果必须使用同一份配置、运行时修改和版本记录，
不能直接与不同控制协议的历史成绩混用。详细协议和历史记录见
[docs/POINTGOAL_BENCHMARK.md](docs/POINTGOAL_BENCHMARK.md)、[docs/ACCELERATION.md](docs/ACCELERATION.md)。
