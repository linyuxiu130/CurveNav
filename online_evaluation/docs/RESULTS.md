# PointGoal 测评成绩记录

本文只记录已经落盘并核对过的结果。固定协议、安全真值和统计口径见
[`FULL_PAPER_EVALUATION.md`](FULL_PAPER_EVALUATION.md) 与
[`POINTGOAL_BENCHMARK.md`](POINTGOAL_BENCHMARK.md)；吞吐和场景复用见
[`ACCELERATION.md`](ACCELERATION.md)。

## Result contract

Official project results use the frozen `pointgoal-v2` suite: 20 Home and 20 Commercial scenes,
100 released episodes per scene, seed 1234, FP32 policy inference and `num_envs=16`. Camera,
controller, termination and SPL remain those recorded in the suite manifest. This accelerated
project protocol is not a numerically exact reproduction of upstream single-environment execution.

Only runs with all 4,000 unique episode identities and complete Home/Commercial summaries enter
the table. Partial results and throughput calibrations remain outside it.

## 正式 B16 全场景结果

| Model | Checkpoint | Home SR | Home SPL | Commercial SR | Commercial SPL | Episodes | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SanD-Planner | `sandplanner/NoMax.pth` | — | — | — | — | 0/4000 | scene loading |
| NavDP | `navdp/navdp_pretrain.ckpt` | — | — | — | — | 0/4000 | queued per resident scene |
| X-NavDP | `x-navdp/x-navdp_posttrain.ckpt` | — | — | — | — | 0/4000 | queued per resident scene |

## Active 2080 run

```text
host: 8x RTX 2080 Ti
session: pointgoal-b16-sand-navdp-xnavdp-40x100
simulator GPUs: 0,1,2,3,4
policy GPUs: 5,6,7,5,6
num_envs: 16
session root: /mnt/data1/huangshibo/eval-server-audit/runs/sand-navdp-xnavdp-official-b16-40x100/pointgoal-v2/model-set/20260902_182456
SanD root: <session root>/models/sandplanner/00-sandplanner-5bfdbe772b02
NavDP root: <session root>/models/navdp/01-navdp-cc0246524765
X-NavDP root: <session root>/models/x-navdp/02-x-navdp-981ae026d7f4
run log: /mnt/data1/huangshibo/eval-server-audit/runs/pointgoal-b16-sand-navdp-xnavdp-40x100.log
```

The run is fail-fast. Each scene remains loaded while SanD, NavDP and X-NavDP are evaluated in
that order, then the worker releases the scene and takes the next one. The table is updated only
from validated `summary.csv` files.

## 2026-08-30：4090 单 Home 场景 100 回合

以下结果来自 4090 上同一 Home 场景、seed `1234` 的已完成 100 回合运行：

| 模型 | 回合 | `num_envs` | 成功率 | mean SPL | 运行性质 |
| --- | ---: | ---: | ---: | ---: | --- |
| CurveNav | 100/100 | 10 | 0.320000 | 0.297861 | resident 吞吐诊断 |
| NavDP | 100/100 | 10 | 0.590000 | 0.568184 | resident 吞吐诊断 |
| X-NavDP | 100/100 | 1 | 0.900000 | 0.762712 | 单场景 `num_envs=1` 运行 |

这些都是单场景结果，不是 40 场景、4,000 回合的论文复现成绩；`num_envs=10`
结果改变 simulator 调度，只用于吞吐和链路诊断，不能与 `num_envs=1` 结果直接
做模型排名。CurveNav/NavDP 的 resident 运行还包含同一场景的常驻生命周期，不能
按单回合冷启动时间外推。

共同元数据：suite digest
`8f2308e70c232d66e912b0611b4fec19f99e9289fdcc948bb27c6e452ef6e291`，runtime
revision `878740a2011856d0e3782dd6ccd880fd2eccd70f`，seed `1234`，场景
`home/MVUCSQAKTKJ5EAABAAAAABA8_usd`，GPU 为 4090。

| 模型 | checkpoint SHA256 | artifact SHA256 |
| --- | --- | --- |
| CurveNav | `b04706aee46d98f735df89ee9d408709ee86ceee5cffad96a02fa0eec1c8a905` | `98d1e8d4d5923509f8c24702a89d27669328042dd24227f1ad730cecc99f7983` |
| NavDP | `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947` | `cc02465247653a8787f7a7f4d0b24d694cb8286f5c18e470c6bcbf8c1cd49b93` |
| X-NavDP | `267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24` | `981ae026d7f46da90bfc085cb3e628850a2c541fbef5498ed50e274104a2a26d` |

### 原始 artifact

- CurveNav/NavDP resident 对比：
  `/DataDisk2/hsb/eval-server-audit/runs/four-model-resident-b100-control-r2/pointgoal-v2/resident/20260830_002958/`
- NavDP 汇总：
  `models/navdp/01-navdp-cc0246524765/summary.csv`
- CurveNav 汇总：
  `models/curvenav/00-curvenav-98d1e8d4d592/summary.csv`
- X-NavDP 单场景结果：
  `/DataDisk2/hsb/eval-server-audit/runs/x-navdp-b100-numenv1-r3/pointgoal-v2/x-navdp/20260830_045722/models/x-navdp/00-x-navdp-981ae026d7f4/summary.csv`

## 未完成或不可合并的运行

- NavDP 的另一轮 `num_envs=1` 运行只落盘 7/100 条记录，不计入上表。
- SanD 运行只落盘 44/100 条记录，不生成正式成绩。
- 没有 40 场景、4,000 回合的完整模型结果时，不填写 Home/Commercial 总表，
  也不把单场景结果称为论文成绩。

## 记录规则

新增成绩必须同时记录模型 checkpoint SHA、runtime revision、suite digest、seed、
scene 集合、episode 数、`num_envs`、GPU 和原始 artifact 路径。任何缺失 episode、
重复 ID、跨 `num_envs` 混合或未完成运行都只能放在“未完成”部分，不能进入正式汇总。
