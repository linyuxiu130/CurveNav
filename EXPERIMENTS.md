# CurveNav experiments

This is the sole experiment record. Each entry contains only the hypothesis,
authoritative run, result, root conclusion, and next decision. Architecture
details belong in `ARCHITECTURE.md`.




## 2026-09-17 — 双卡吞吐审核与数学等价共享 K/V

- 对齐同一早期区间（step500–800）：稳定v6中位1228.36样本/s，新探索1155.47样本/s，下降5.9%；新探索训练生成3对而非3条源曲线。双卡GPU利用率94–97%、PCIe Gen4×16；梯度只在最后微批同步。不能把GPU忙碌当作没有优化空间，也不能把启动冷缓存速度当作稳定吞吐。
- 修复重复数据搬运：交叉注意力按同场景拼接候选query行，共享K/V；softmax只沿场景维，候选仍独立。移除repeat_memory，无新参数、损失或候选分布变化。44项CPU检查通过，含前向及输入/参数梯度对照；CUDA编译前后向及部署采样检查通过。
- 隔离A/B（同4090，实际数据，正式384×12，batch64，未编译）：热批中位188.76→182.58ms，约+3.38%吞吐；单观测推理46.29→42.74ms，约+8.30%。这些不是已测得的DDP编译训练收益。
- 全局batch保持256的微批实测（同GPU，未编译，每rank128样本/更新）：64×累积2为328.55ms/10.50GiB；128×累积1为323.93ms/20.04GiB。仅快1.43%而显存接近翻倍，保留64×2，不修改checkpoint拓扑或随机数分组。
- 启动瓶颈证据：Conda torchinductor路径指向共享存储，检查进程出现cxiWaitEventWait及大量小文件读取；该编译检查13分38秒仍未完成，主动中止。改用已有本地PyTorch缓存后完整检查103.21秒通过（缓存历史不同，不宣称严格固定倍数）。脚本不再强制共享编译缓存；不删除其他任务可能使用的持久缓存。
- 已在step5826完整checkpoint落盘后停止旧launcher，保留原始checkpoint、优化器、EMA、scheduler及每rank RNG。新冻结bundle于 `outputs/train-structured-exploration-7480-10ep-sharedkv-20260917` 从同一checkpoint继续；先补第1轮验证，随后继续剩余9轮。未重新开始训练。
- 证据：`outputs/kv-throughput-audit-20260917` 的 before/after、micro64/micro128、编译检查日志和源码快照。正式恢复后第2轮step6460、loss0.5944，最近10个日志窗口吞吐中位1258.3样本/s；旧版新探索启动稳定区间为1155.5，观察到约8.9%提升。训练阶段和数据窗口不同，此处为运行观察，严格同批A/B以上述隔离测试为准。

## 2026-09-17 — 7480 数据结构化探索双卡正式训练，10 epoch（运行中）

- 运行目录：`outputs/train-structured-exploration-7480-10ep-20260917`；tmux `curve-exploration10`。源码、配置、dirty diff、环境与数据 manifest SHA256 已冻结于运行目录。
- 从头训练，seed42，双4090，每卡64、累积2、全局256；5826步/epoch，共58260步。学习率2e-4、warmup5epoch；神经BF16、几何/Flow FP32。目标/无目标单阶段 Flow＋Huber critic，每epoch验证并保存权重，按验证路线效用保存best。
- 新训练深度缓存81 GiB已完成，双rank共用；验证缓存14.09 GiB与旧验证深度逐文件同inode，使用硬链接复用，不重复复制。
- 已进入真实优化：step1 loss4.3793；step20 loss4.1124（Flow2.4032、critic1.7092），梯度范数有限，没有OOM或DDP错误。初始日志不作为收敛判断，尚无新验证/在线成绩。

## 2026-09-17 — 结构化候选探索接入与 7480 数据合并核验（尚未正式训练）

- 假设：原模型同方向候选过于接近，先扩展候选支持集，再由现有目标/安全教师监督评分；不把扰动轨迹当作专家，不重做此前失败的加权伪目标 Flow 实验。
- 实现：同一 Flow 用 20% goal dropout 学习无目标条件；共享场景编码/KV，批量生成 32 对目标/无目标曲线，以物理控制点执行 X-NavDP 的混合、直线基底、纯无目标、轴翻转、缩放与原样保留，评价最终 32 条。训练用同分布 3 候选＋专家，单阶段联合 Huber 评分。上轮 centered loss 未改善在线成绩，恢复稳定 v6 Huber；没有增加 RL、二次校准或曲线修补。
- 数学边界：线性变换与物理 B-spline 解码可交换；不能在带非零均值的标准化增量上直接翻转。无目标条件是整块目标嵌入被移除，不是零距离目标。保持 C2 连续不代表动力学/安全保证；新增倒车候选沿用当前 MPC 协议，不能把它们当成已验证安全专家。
- 数据：已有 980 条补充专家完整通过审核，四方向各 245，原始 SHA256 `5ed99de99b73315b3ed317459f885bf902434f000b1724f05158d74171e51793`；未重复渲染。合并 7480 轨迹，训练 1,491,638、验证 260,340 状态。正式入口更新数据路径及训练集归一化统计。验证全部数组逐元素等同 6500_v6，1100 个深度文件是同一硬链接，未引入验证数据。
- 检查：47 项 CPU、2 项 CUDA（含 compiled fullgraph 前向/反向）通过；checkpoint 固定采样库严格恢复后的候选与分数一致。正式 384×12 模型在 GPU0、实际补充数据 batch64 三步前后向和教师标签检查通过，全部参数梯度有限。峰值 10.50 GiB，热批次约 185–206 ms，未编译单观测推理 43.28 ms。此为单批短检查，不代表全数据吞吐或收敛；随机初始化短检查 loss 有波动，不作为训练趋势。
- 证据：`outputs/structured-exploration-20260917/verification.json`、`gpu-check.json` 与可复现 `gpu_check.py`。旧权重/正式成绩保留；新无目标能力尚未经过正式训练，不能宣称已提高成功率。
- 补充数学审核：4 个针对性检查通过；强制有/无目标时 Flow 共享生成器及深度梯度均非零，无目标时目标 MLP 梯度为零；完整两步无目标 Flow 对替换目标逐元素不变；同候选改变目标会反转教师排序，零目标则偏好停止。本次未修改生产损失或推理语义。连续自主探索仍缺覆盖状态/收益/部署任务入口，不能称已完成。
- 范围收敛为简单无目标候选生成：新增 `sample_nogoal`，不加入连续自主探索、地图覆盖或新 reward。复用 20% mask 的专家 Flow loss，输出32条原生候选；目标替换不影响输出、固定起点、独立噪声及目标/无目标梯度检查通过。没有据此宣称训练后安全性或多样性已达标。
- 收敛后的消融：同一 7480 数据、10 epoch、seed42、全局 batch256、相同验证任务，只比较稳定 v6＋数据与本次结构化探索＋数据。对照代码使用原 v6 冻结 bundle，不在生产路径加入算法切换分支。先检查候选安全覆盖、侧后方方向覆盖、选择遗憾和推理时延，再决定在线对照；不将“轨迹更分散”直接解释成导航改善。

## 2026-09-17 — 居中评分损失并入正式联合训练，10 epoch

用户要求完整合并训练再测。使用原v6正式数据、seed42、双4090、每卡64、累积2、
全局batch256、5535步/epoch、10epoch共55350步，从头训练；学习率2e-4、warmup5epoch及
其他配置保持原10轮训练设置。唯一算法改动是原评分SmoothL1改为状态内居中MSE；
不加入Reflow、不冻结感知/生成器、不追加后训练阶段。
评分训练仍为1专家+3条当前两步生成候选，部署32选1；这检验联合训练的可迁移性，
不把先前固定32候选、冻结主干的后训练成绩直接当作本次预期。
已有数据和深度缓存复用，不重新生产或编译全量数据。

损失 `mean((e-mean_candidates(e))²)` 在FP32计算；神经网络仍使用原BF16区域。
训练目标写入原checkpoint契约，避免把旧Huber优化器状态误认为新目标续训。
既有测试增加一个分差等价、逐状态偏移不变与解析梯度检查；使用现有模型前后向检查。
每epoch保存，仍按原验证集部署32候选效用选best；结束后自动离线512状态/source和同场景100回合在线评估。
在线继续使用修复后的渲染隔离、16环境、原任务/控制协议，GPU1运行模拟器与策略。

实验：`outputs/train-centered-joint-10ep-20260917/`；独立源码快照、完整配置、dirty diff和协议均保留。
上一轮完整结果已确认：原95%/0.859417；centered两个种子96%/0.877030、93%/0.849319；
普通MSE91%/0.835341；Reflow95%/0.872266；Reflow+配套居中评分97%/0.887106。
后训练居中收益尚不稳定；本次是用户授权的正式联合训练验证，不是已经证明优于基线。

验证：现有critic/checkpoint/policy检查共47项通过，含GPU编译前后向与部署有限值检查；
数学测试覆盖两两分差等价、逐状态偏移不变及解析梯度。无新增测试框架。
`tmux: curve-centered-joint10` 已启动双卡单阶段正式训练，后接离线及在线评估，不进行二阶段微调。
已确认真实优化到step20：Flow loss=2.36660、居中critic loss=2.44336，损失与梯度有限。
初始20步吞吐约240样本/s，与原训练同期236样本/s接近；尚不能用启动阶段估算稳态速度。

完整结果：10epoch/55350步及离线、在线均完成。训练约3小时11分钟，稳态约1300样本/s；
按原验证规则选中epoch9/step49815。相同7168离线状态：本体碰撞8.3147%→7.8683%，
前1m碰撞0.8510%→0.7254%，选中效用0.07741→0.08768，但安全候选覆盖95.8705%→95.2567%。
同场景同100任务在线却从95%/SPL0.859417降至90%/0.817741；seed、16环境、任务、
runtime revision/patch和执行协议均一致。新增成功52/78/91，新增失败6/8/11/33/37/38/77/85。
结论：单阶段居中监督的离线局部改善没有转化为本场景在线改善，本轮不替换原模型。
不能由这一次结果证明所有单阶段训练均不成立，也不能归因于某个未验证的具体故障。

## 2026-09-16 — 目标与空间几何交互：连续受控消融（详情）

本节是本轮唯一实验账本。正式模型保持 v6 epoch8 EMA；实验源代码与检查点隔离在
`outputs/goal-spatial-search-20260916/`。双卡各运行一个不同假设，不以重复运行充当新证据。
结论限定为已检查的数据、预算和评估协议，不能宣称全局最优。

当前状态：已完成的同场景100回合中，原v6为95% / SPL 0.859417；
S1-Huber为92% / 0.837225（淘汰）；S1-centered为96% / 0.877030（保留确认，未替换正式模型）。
居中损失的独立种子、普通MSE对照以及两步Reflow正在同一常驻场景队列确认。
已完成的目标交互分支、曲线度量、独立噪声效用重加权、完整评分器解冻均未可靠胜过原模型，
不进入生产代码。所有新训练均使用固定子集，不重跑全量训练。

### 已有证据与不再重复的实验

| 实验 | 证据 | 当前结论 |
| --- | --- | --- |
| 显式目标重参数化 | 224 个验证状态，短训安全性/候选覆盖没有明确提高 | 不重复相同重参数化短训 |
| Flow 2→4→8 步 | 4 个窄路案例中 3 个没有解决 | 不把增加积分步数当成已证实方案 |
| 单摘要目标→BEV、解码器整体短训 | 1024 步；普通组碰撞 34→32/448，但原 v6 是 31/448；侧后目标 19→19/164 | 微弱收益，不合入；不能否定所有目标几何交互 |
| 上述分支机制检查 | 607 状态注意力熵 0.99967；反向目标特征变化 2.22%；128 状态关闭分支候选点均移 1.60 mm，同模型重复输出严格相等 | 学成弱场景摘要；需区分训练不足与交互结构不足 |

上述证据分别位于 `outputs/goal-reparameterization-20260915/`、
`outputs/v6-change-audit-20260916/` 和 `outputs/goal-bev-gated-20260916/`。
当前初始参考在普通 448 状态：安全候选 429、选中整曲线碰撞 31、前 1 m 碰撞 3、
平均 teacher utility 0.113773；侧后近目标 164 状态：安全候选 153、整曲线碰撞 21。
这些是离线状态统计，不是在线回合成功率；相邻帧不能视作独立回合。

### 本轮公共协议与数学约定

- 同一个 v6 EMA、原数据归一化、原训练划分；16,640 个已缓存训练状态、607 个验证状态。
  保持原数据集，不同时混入扩充数据。感知/生成主干/评分器全部冻结。
- 只学习新增模块；Flow MSE、两步积分、32 个固定候选噪声、评分器和控制协议不变。
  BF16 神经计算、FP32 物理几何；不增加碰撞筛选、强制多样性、gate 下限或目标吸引力规则。
- 第一轮每组 batch256、LR=3e-4、4096 步、seed20260916；共同样本顺序及训练噪声。
  512/1024/2048/4096 步检查保存的 EMA，使用固定噪声计算验证 Flow loss。
- 初始化必须复现原模型的速度场、候选、评分和选择。首两步检查 gate/分支梯度；
  零 gate 时分支首步梯度为零是乘法结构的正常结果，不同时零初始化分支权重。
- 选择依据同时看安全候选覆盖、执行前缀碰撞、选中/最优候选效用、路径长度和进展；
  不凭训练 loss 或单个样本判胜。微弱收益再用独立状态/种子确认，避免反复调同一验证集。

| ID | GPU | 假设与唯一架构变量 | 状态/结论 |
| --- | --- | --- | --- |
| G1-global | 0 | 原单摘要模块，冻结主干后给予独立适配训练，检验之前是否学习不足 | 4096 步完成；1024 步有微弱收益，继续训练退化，转独立轨迹确认 |
| G1-spatial | 1 | 逐 BEV 格子融合自身特征与 `[p/H,(g-p)/H]`，零门控残差写入生成器 K/V；复用原注意力 | 4096 步完成；未胜过原模型，最优候选效用随训练下降 |
| G2-intent | 0 | 不汇总场景；米制目标经 MLP 编码，以每层独立零门控持续注入原 12 层，让原注意力融合几何 | 已完成；未稳定胜过原模型，不宣称复现 NavDP |
| G2-multi | 1 | 将 G1-global 单查询改为与 7 个增量 token 对应的 7 个可学习查询；其他结构不变 | 已完成；未稳定胜过原模型，不指定物理锚点或转弯模式 |
| G3-bias | 0 | 仅微调原 12 层几何相对注意力 MLP，LR=1e-4；匹配对照 | 已完成；4096 步效用微增，但最优候选效用下降 |
| G3-goalbias | 1 | 同一 MLP 的原7维几何增加目标到格子的相对位移/距离3维，新增权重列零初始化 | 已完成；没有胜过匹配对照或原模型 |
| G4-channel | 0 | G1 的每 token 标量 gate 改为每特征 gate，其他部分不变 | 已完成；4096步普通碰撞32/448，效用0.10116，最优候选效用0.20171，未胜过原模型 |
| G4-channel-scene | 0，接前组 | 与 G4-channel 相同，但新增查询不接目标；原模型的目标条件保留 | 已完成；4096步普通碰撞31/448，但前1m碰撞5次、效用0.10189，仍不优于原模型 |

G1-spatial 的数学形式为 `m'_j=m_j+tanh(a)F(m_j,[p_j/H,(g-p_j)/H])`。
只修改生成器的神经记忆；运动 token、物理坐标、可见性及评分器输入不改。
原始 token 的 FP32 恒等通路保留，归一化只在分支内。各格子的残差不必共线，
因此取消了单摘要 `ΔH=αcᵀ` 的秩一注入限制，但这不构成性能保证。

论文参考只作为假设来源：
[OccPlanner](https://arxiv.org/html/2608.14160) 的目标/占据交互；
[DAGR](https://arxiv.org/html/2607.13731) 的门控和特征变换消融；
[Flamingo](https://arxiv.org/html/2204.14198) 的冻结主干、零初始化残差适配。
不照搬像素目标回归损失，不将论文的其他任务成绩当作本模型的证据。

后续决策：先读第一轮阶段趋势；若单摘要在充分适配后仍弱而空间模块有效，
才深入空间模块；若两者均无收益，先检查训练/验证差异及条件敏感性，避免盲目堆层。

### G1 阶段结果（普通 448 状态；侧后组另列）

| 组/步数 | 整曲线碰撞数 | 前1m碰撞数 | 选中效用 | 最优候选效用 | 侧后碰撞数/164 | 固定噪声验证 Flow MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 v6 | 31 | 3 | 0.11377 | 0.20477 | 21 | 0.21726 |
| global/512 | 32 | 4 | 0.10833 | 0.20134 | 20 | 0.21839 |
| global/1024 | 30 | 3 | 0.11770 | 0.20402 | 21 | 0.21962 |
| global/2048 | 34 | 3 | 0.09325 | 0.20335 | 21 | 0.22315 |
| global/4096 | 34 | 4 | 0.09378 | 0.20027 | 21 | 0.22651 |
| spatial/512 | 35 | 4 | 0.09461 | 0.19525 | 21 | 0.22280 |
| spatial/1024 | 32 | 4 | 0.10381 | 0.19077 | 20 | 0.22416 |
| spatial/2048 | 33 | 4 | 0.09496 | 0.18458 | 21 | 0.22754 |
| spatial/4096 | 31 | 3 | 0.10181 | 0.18510 | 21 | 0.23090 |

- 原模型初始输出在两组均逐元素一致。首步 gate 梯度非零、分支梯度零，次步分支梯度非零，符合链式求导。
- global 约 5900 samples/s；spatial 约 3040 samples/s，训练峰值约 10–11 GiB。
  空间分支需要反传各层 K/V 投影，训练更慢；不能将训练速度直接等同于推理速度。
- global/1024 仅少一次碰撞、候选最优效用未提高，可能是选择边界变化，尚无绕路改善证据。
- spatial 的训练 loss 下降但验证 loss/候选效用退化，反驳“逐格几何交互一定更好”。
  该结果仍限定于冻结主干、16,640 状态子集，不能外推为此结构永远无效。
- 已一次性构建第二验证集：1016 状态（普通896、侧后127，部分重叠），436 个深度轨迹文件，
  与第一验证集涉及的392个文件零重叠；同14个验证场景，不是全新场景。
  数据在 `outputs/goal-spatial-search-20260916/holdout.pt`，用于确认微弱收益，不用于反复挑检查点。

### G2 / G3 结果与被反驳的解释

| 组/步数 | 普通碰撞/448 | 前1m碰撞 | 选中效用 | 最优候选效用 | 侧后碰撞/164 |
| --- | ---: | ---: | ---: | ---: | ---: |
| intent/512 | 31 | 3 | 0.11249 | 0.20119 | 20 |
| intent/4096 | 31 | 4 | 0.11167 | 0.20071 | 22 |
| multi/512 | 32 | 4 | 0.10771 | 0.20130 | 20 |
| multi/4096 | 31 | 4 | 0.10831 | 0.20368 | 21 |
| bias/4096 | 31 | 3 | 0.11402 | 0.19986 | 20 |
| goalbias/4096 | 32 | 3 | 0.10909 | 0.20009 | 21 |

- 更深地重复注入目标、增加查询数量、显式提供目标相对几何，都没有在本协议下获得稳定收益。
  因此不再把“目标注入不足”当作已经证实的主要根因；也不凭这些子集实验否定所有类似架构。
- G3 在首次 optimizer step 前遇到本机 PyTorch 2.7 的 `LSE is not correctly aligned (strideH)`。
  用单个原 cross-attention 模块复现：冻结 Q/K/V、仅训练 bias MLP 时原 CUDA 路径反传失败。
  两组统一使用 PyTorch 的标准 math SDPA 完成训练，推理保持原 native SDPA；没有修改模型以绕过错误，
  没有升级环境或加自动回退。初始候选/速度/评分/选择仍逐元素匹配原模型。
  失败启动单独保存在 `initial-native-sdpa-error/`，没有重复已经完成的优化步骤。
  G3 吞吐约 4200 states/s、峰值约14.6 GiB。

### G1 微弱收益的独立确认与因果追踪

| 第二验证集，普通896 / 侧后127 | 普通碰撞 | 前1m碰撞 | 普通效用 | 侧后碰撞 |
| --- | ---: | ---: | ---: | ---: |
| 原 v6 | 86 | 14 | 0.04536 | 4 |
| global/1024，seed20260916 | 87 | 15 | 0.04558 | 3 |
| global/1024，seed20260917 | 85 | 15 | 0.04998 | 3 |

两个独立种子都修复第二集的状态136977，但此例不能算生成能力提升：
原32条中已有23条安全；原评分选择27号（clearance=-0.05m），两组改选25号（0.15m）。
原模型的25号本来就安全，且 teacher utility=0.1830，高于新25号约0.1692；
安全候选数依旧23，原27号也依旧碰撞。收益来自选择变化，并非发现新的通路。
该状态的新旧选中轨迹均移约10cm，全部候选均移仅6–7mm。
第二集已被用于此机制分析，后续不能再称为完全盲测；在线回合另作确认。

### S1 — 固定生成器，检查评分监督是否与32选1一致

触发证据是上述同一候选集合中的重复选错，不是泛泛怀疑所有绕路都来自评分器。
保留原感知、生成器、评分器前两层及池化，仅训练最后 MLP。
对16640训练状态一次性生成部署时完全相同的32个固定噪声候选、全局教师标签、FP32池化特征。
两卡各缓存8320状态，约89/111秒完成；验证候选直接复用保存文件，未重复生成。
共有532480条训练候选。三组相同初始化、batch256状态、LR1e-4、2048步；256/512/1024/2048步检查EMA。

- S1-huber：原 SmoothL1，控制使用32条训练候选带来的影响。
- S1-mse：平方误差。Huber 的条件最优值通常不是条件期望，因此可能弱化少数严重负效用。
- S1-centered：`mean((e-mean(e, candidates))²)`，其中 `e=score-teacher`。
  这是全部候选对误差差值平方的 `1/(2K²)` 倍；去掉对 argmax 无意义的共同偏移。
  在无限表达能力下，最优分数差等于条件期望效用差；不是 softmax 教师分布的另一种风险目标。
  最后截距在该损失下不可辨识，保持预训练值；不加温度、阈值或推理筛选。

选择检查点只用第一集普通组平均选中 teacher utility，再读第二集一次；没有用第二集挑步数。

| S1，选择步数 | 第一集普通碰撞/448 | 第一集侧后碰撞/164 | 第二集普通碰撞/896 | 第二集前1m碰撞 | 第二集侧后碰撞/127 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 v6 | 31 | 21 | 86 | 14 | 4 |
| Huber，512 | 33 | 17 | 83 | 14 | 6 |
| MSE，256 | 33 | 16 | 83 | 15 | 6 |
| centered MSE，256 | 33 | 19 | 84 | 15 | 4 |

结果是混合的，尚不能替换原模型。原候选安全覆盖完全不变，不把评分收益写成生成收益。
Huber 与 centered MSE 使用原生 checkpoint loader 在4个固定状态完成部署检查，原生成器输出逐元素不变。
部署检查在 `score-deployment-parity.json`，优化器可恢复的 head 状态与完整部署权重均已保存。

### 在线确认

`tmux: curve-architecture-search / online-score100`：GPU1复用同一场景，依次运行 Huber / centered MSE 各100回合。
场景为 `commercial/MV4AFHQKTKJZ2AABAAAAADQ8_usd`，16环境、同原协议与源码 bundle，保留已修复的渲染隔离。
不是宣称这两个模型离线已胜出，而是确认混合指标是否转化为闭环表现。
参考原v6的100回合为95成功、SPL=0.859417；本轮未更换控制器或任务定义。
路径：`outputs/goal-spatial-search-20260916/online-score100/`。
GPU0另有 Anyverse 任务约6GiB，未终止；本轮GPU0只安排已测得显存约10–11GiB的受控小模型实验。

Huber已完成：**92/100，SPL=0.837225**，低于原v6。恢复任务21，但丢失38/77/85/99。
91个共同成功任务的SPL仅+0.001344，平均路程仅短0.01038m；没有明显绕路改善。
成功任务单独SPL=0.910027高于原0.904650，主要比较对象发生了变化，不能据此判胜。
centered MSE完成：**96/100，SPL=0.877030**，原v6为95/100、0.859417。
新增成功21/52/78，新增失败38/89；93个共同成功任务的SPL+0.010685、平均路程-0.11760m。
任务17路程11.33695→4.80756m，SPL 0.41816→0.98608，绕路确有改善；任务72少走1.35m。
100个配对任务SPL差的bootstrap 95%区间[-0.02144,0.05725]仍跨零，不宣称统计确定或跨场景最佳。
Huber不合入；centered MSE保留为首个在线有收益的候选，不替换正式基线。
表格与配对结果在 `online-comparison.json`，路径对照在 `online-score-trajectories.png`。
生成器与控制器未变，但闭环分叉后输入也变化；仅凭执行轨迹不能证明每个分叉处32候选完全相同。

预登记独立训练种子确认：S1-centered改用seed20260919，除此之外复用缓存、预算、
256/512/1024/2048阶段和第一集选点规则，第二集只读最终选点。此项确认训练顺序敏感性，
不是重复同种子实验；不根据在线任务17调权重。若有效，复用常驻场景进行同100任务确认。

独立种子完成，仍按预定规则选择256步：第二集普通碰撞83/896、效用0.05547、
前1m碰撞15；侧后碰撞5/127。与首种子的84/0.05313/15/4接近，仍是混合离线结果。
首种子的在线收益值得独立确认，完整artifact已原子入队
`online-reflow100/queue/centered-seed20260919.ready`，与Reflow共用常驻场景。
不根据该种子离线成绩另调超参数或选在线任务。

补充数学与机制边界：令候选误差 `e=s-u`、`P=I-11ᵀ/K`，本损失为
`eᵀPe/K`，梯度 `2Pe/K`；`P` 是正交投影，唯一丢弃的是所有候选相同的评分偏移。
平方损失在条件分布下拟合 `P E[u|input,candidates]`，故理论argmax与期望效用一致，
但有限模型、教师信息不完全、离线与闭环差异仍会导致错误，不保证绝对安全或全局最短。
它没有改变候选、增加避障阈值或手工目标项，部署结构、算力成本均保持原样。
第二集普通组居中误差0.11232→0.09529，差值加权两两错序0.30931→0.30752，
仅后者是微弱改善；最高两分并列比例42.63%→40.74%，不能宣称量化并列已解决。
原v6与新模型在线每回合平均规划延迟的中位数约220.0/220.9ms，未见明显算力代价变化。
分析复用现有缓存：`score-order-audit.json`；没有重新做已完成的FP32读出实验。

在线归因仍缺一个必要对照：S1-MSE已经训练完成，但此前只做了离线确认。
因此将其同样的256步选点加入常驻队列；不重新训练。
只有与普通MSE相比仍有收益，才能把优势归因于“居中去掉共同偏移”，否则可能只是平方损失或重新校准的作用。
S1-Huber、MSE、centered的数据/训练预算/选点规则一致，此比较不更换生成器或控制器。

第二集普通896状态的平方误差精确分解：`mean(e²)=mean((e-mean_i e)²)+mean((mean_i e)²)`。
原模型总MSE=0.49643，其中共同偏移0.38410（77.37%）、相对误差0.11232；
centered总MSE反而为0.53933，但相对误差降至0.09529；普通MSE组相对误差0.10023。
这支持关注相对监督的动机，但共同偏移是否挤占模型容量仍是解释，不是由分解自动证明的因果结论。
该比例描述平方误差诊断，不能当成原Huber训练loss的组成比例。
文件：`score-common-mode-decomposition.json`。

### S2 — 固定候选与教师，比较独立评分和集合条件评分

S1 的混合结果尚不支持替换正式模型。进一步区分：是原特征缺少可用信息，
还是单候选读出不能利用同一次规划的候选之间的差别。仍复用同一532480条缓存候选；
不改变生成器、教师、安全定义或控制器，不重新采集候选。

- point_residual：冻结原评分，增加约14.8万参数的独立候选MLP校正。
- context_residual：相近参数量，令 `μ=mean_i(φ_i)`，
  `s_i=s_i_old+a(μ)ᵀ(φ_i−μ)/sqrt(D)`。`a` 是384→192→384的MLP。
  集合均值对排列不变，逐候选输出对排列等变；共同特征可以改变相对比较方向。
  校正的候选均值为零，只固定评分的共同偏移自由度，不约束轨迹或碰撞行为。
- 两组均仅将校正输出层零初始化，完整初始评分必须与原模型逐元素一致；
  首步只更新输出层、随后梯度进入前层，符合链式求导。
  都用S1相同的centered MSE、数据顺序、seed20260918、LR1e-4、2048步与检查点选择规则。
- 此处只验证固定部署32候选的独立实验模块。若有效，再将候选维度显式接入正式评分接口；
  不把临时flatten/unflatten约定当作生产架构，也不把失败分支留下兼容。

新组即使优于其他新组，只要未可靠胜过原v6，也不晋升为最佳模型。

结果（上述设计在训练完成前登记）：

| 组，第一集选中的步数 | 第一集普通碰撞/448 | 第一集侧后碰撞/164 | 第二集普通碰撞/896 | 第二集前1m碰撞 | 第二集侧后碰撞/127 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 v6 | 31 | 21 | 86 | 14 | 4 |
| 独立校正，512 | 32 | 17 | 84 | 14 | 5 |
| 集合条件校正，256 | 32 | 18 | 85 | 17 | 6 |

集合条件校正的第二集效用0.04222，低于原0.04536；独立校正为0.05307，但近目标安全退化。
此轮未支持新增集合条件评分结构，不合入正式链路，也不继续盲目延长训练。
独立校正2048步在第一集侧后组21→15次碰撞，普通组却31→33；这是局部收益与整体退化并存，
不能仅挑有利子组宣称成功。未据此改换检查点选择规则。

### 统计与复现范围

`summarize.py` 只读取已有结果，生成 `generator-results.csv`、`paired-scene-analysis.json`、
`experiment-summary.png`，不重复推理。配对bootstrap以14个场景为单位，而非把相邻状态独立采样。
S1第二集效用变化：Huber +0.00712，95%区间[-0.01222,0.03181]；
centered +0.00778，区间[-0.01488,0.03751]。均跨零，尚无稳定改进证据。
这是探索性区间，未做多重比较校正，且该验证集已参与机制分析，不能称为最终盲测。

### F1 — 曲线诱导的 Flow 误差度量

复用已经完成的原始解码器微调对照 `goal-bev-gated-20260916/baseline`，不重跑对照。
已有 `v6-change-audit-20260916/objective-audit.json` 证明：同范数控制增量误差所造成的曲线RMS误差
最大相差55.69倍；这是有限容量训练的误差权重问题，不是原Flow目标有错误最优解。
本组不再添加目标模块，只将坐标MSE换成由原样条线性映射导出的曲线位置MSE：
`L=E[eᵀJᵀJe]/trace(JᵀJ)`，`e=vθ−(noise−clean)`，J包含原控制增量尺度。
归一化使单位各向同性误差的期望损失仍为1，避免仅改变整体梯度尺度。
J满列秩，故该正定常数度量仍以条件均值速度为无限容量最优解；不改变Flow路径、噪声或积分。
位置误差本身不能保证低曲率，结果必须同时检查安全、候选效用与轨迹质量，不能预设成功。

与已完成对照严格相同：冻结感知和评分、训练原解码器，seed20260916、batch128、1024步、LR2e-5、
原样本顺序和噪声；GPU0运行。唯一变量为上述训练误差度量。正式模型保持不变。

1024步已完成，峰值11.26GiB、约2293 states/s；初始速度/候选/评分/选择与原对照逐元素一致。

| 组 | 普通碰撞/448 | 前1m碰撞 | 普通效用 | 最优候选效用 | 侧后碰撞/164 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原v6，不微调 | 31 | 3 | 0.11377 | 0.20477 | 21 |
| 已有坐标MSE微调 | 34 | 8 | 0.08903 | 0.18449 | 19 |
| 曲线度量微调 | 32 | 8 | 0.10027 | 0.19653 | 21 |

曲线度量比匹配微调对照少退化，但没有超过原v6，前1m安全性也未恢复。
数学上合理不等于经验上更好；不合入，不将“坐标损失有bug”作为结论。

### 安全与绕路不能混为一个收益

复用已有候选，按原教师加法公式恢复进展分量，与保存的原进展逐元素核对误差<3e-6。
`safety-efficiency-decomposition.json` 区分全部32条安全和混合安全的状态，不再重新推理。
第二集普通组中722个状态的32条均安全；全地图教师最优选择相对原选择平均仅短0.01821m，
进展多0.02668m。各新评分头的平均长度变化在约-0.00016至+0.00210m，未展现明显缩短路径。
第一集373个全安全状态也类似，教师最优平均短0.02010m。
这支持“普通安全状态中，当前候选集合留给评分的局部效率空间较小”，
不能推出在线总绕路仅2cm，也不能覆盖分岔、近目标和分布外状态。

### 新论文交叉核对与未采用项

- [NeurRAFT，2026-08](https://arxiv.org/html/2608.24026)：两步Flow、稀疏锚点、Jacobian损失与偏好微调相关。
  但其推理仍做重建网格碰撞筛选，不能当作“纯模型无筛选”的证据；其偏好阶段将Flow MSE作为
  负对数似然的替代量，并非精确似然。只最大化间隙也可能加重本项目绕路，不直接照搬。
  本轮F1使用固定线性样条Jacobian，与该文依赖真值关节构型的Jacobian加权不是同一目标，
  前者的条件均值证明不能直接套用于后者。F1已独立验证失败，不因论文宣称有效而合入。
- [Slow Brain, Fast Planner，2026-06](https://arxiv.org/html/2606.20458)：困难语义场景中候选选择存在较大空间，
  但其设定先保证候选可行，再用VLM与延迟融合选择。我们现有纯深度候选并非全部可行，且普通安全状态
  的选中/最优差距小；不由其结论推导“我们必须加VLM评分”。
- [Hydra，2026-09](https://arxiv.org/html/2608.28995)：离散意图与连续Flow分工可作为长期方向，
  但引入世界模型、码本和额外规划代价会同时改变多个因素，本轮没有对应必要性证据，暂不实施。

论文用于产生可反驳假设；现阶段没有证据支持将上述系统整体移植进正式模型。

### F2 — 将已有较好候选的概率质量学回生成器（预登记）

由“安全候选已经存在、换读出未稳定改善”转向生成分布本身。参考上述NeurRAFT的训练阶段对齐思路，
但不把Flow MSE称作精确对数似然；采用可直接定义的经验目标分布。
对同一训练状态的32个原候选，原教师效用为u，定义 `q_i=exp(u_i)/sum_j exp(u_j)`。
q恰是 `max_q E_q[u]−KL(q||Uniform32)` 的解；无碰撞过滤、阈值或新增推理分支。
Flow训练从Uniform32抽候选i，使用 `32*q_i*||v−(noise−candidate_i)||²`，
这是q下Flow损失的无偏估计。其理想速度场对应这32个候选形成的经验加权分布，
不是声称恢复连续原策略的精确指数倾斜，更不保证有限容量/两步积分后的闭环改进。
权重改变的是端点分布；不在积分后修补轨迹。

F2-uniform为同样自生成数据上的普通Flow对照；F2-weighted只改上述权重。
共享原16640状态、样本顺序、候选索引、噪声、1024步、batch128、LR2e-5，感知/评分冻结。
原缓存已有532480个教师分数，但未存控制增量，因此只补算并保存对应14维生成坐标，
复用教师标签，不重复昂贵几何查询；首批复核与旧缓存评分特征一致。
两组依次用GPU0，GPU1保持当前在线测评。若两组均退化，不再叠加模块挽救此分支。
指数加权回归也与[AWR](https://arxiv.org/abs/1910.00177)的基本思想相关；本实验不声称复现其完整RL算法。

两组已完成。补存坐标耗时51秒；两个旧缓存分片各16个状态的评分池化特征逐元素相同。
加权经验分布的平均有效候选数30.95/32，训练集经验期望效用比均匀分布高0.07136，
但这不等于训练后策略能实现该收益。

| 组 | 普通碰撞/448 | 前1m碰撞 | 普通效用 | 最优候选效用 | 侧后碰撞/164 | 普通1m候选间距 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原v6 | 31 | 3 | 0.11377 | 0.20477 | 21 | 约0.015m |
| 自生成均匀Flow | 32 | 4 | 0.09904 | 0.16897 | 21 | 0.01111m |
| 效用加权Flow | 33 | 7 | 0.09605 | 0.16820 | 21 | 0.01148m |

加权目标在经验分布上提高效用，却未转化成模型收益；两组候选覆盖与最优效用都退化。
不合入。不能只根据经验加权目标的理论性质宣称真实两步策略已优化。

### D1 — F2候选收缩的积分机制检查（预登记）

对一维零均值目标N(0,σ²)与独立源N(0,1)，线性概率路径的精确条件速度为
`v(x,t)=[t−(1−t)σ²]/[t²+(1−t)²σ²] * x`（本项目t=1→0）。
两步Euler给出输出标准差 `σ²/(1+σ²)`，而不是σ；例如σ=0.1时约0.0099。
这证明“Flow训练目标正确”不保证“两步采样分布正确”，但不能据此直接断言实模型的收缩全由积分造成。
旧实验只检查4个困难在线状态的2/4/8步，3例未解决；本次为F2机制分析扩展到固定607个验证状态，
只补充原v6的8步结果，复用原2步输出，不重跑旧4例，不修改在线部署协议。
同时比较已有F2结果的候选方差；只有实模型证据支持，才考虑将较准确传输蒸馏回两步。

检查已完成，出现值得深入的候选层面收益，但不是部署收益：

| 验证集 / 积分 | 有安全候选的状态 | 选中碰撞 | 前1m碰撞 | 最优候选效用 | 选中效用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一集普通448 / 2步 | 429 | 31 | 3 | 0.20477 | 0.11377 |
| 第一集普通448 / 8步 | 434 | 32 | 3 | 0.24053 | 0.09217 |
| 第二集普通896 / 2步 | 851 | 86 | 14 | 0.20422 | 0.04536 |
| 第二集普通896 / 8步 | 867 | 100 | 14 | 0.25133 | -0.00280 |

第一集侧后组：安全候选153→157/164，选中碰撞21→18，前1m碰撞16→12。
第二集侧后组：安全候选127→127/127，选中碰撞4→4。8步确实增加普通组候选覆盖，
但原评分器没有利用它，整段选中安全性反而退化；不能直接上线8步。
普通组候选标准差（总体方差开根号）8步为原1.376倍；F2-uniform/weighted为0.668/0.707倍。
理论例子与这些结果一致，但尚不能将所有F2退化唯一归因于离散误差。

### F3 — 保留噪声—轨迹配对的Reflow（预登记）

根据D1两套状态中的候选覆盖增益，验证能否将8步教师的输出分布学回原两步学生。
方法参考[Rectified Flow](https://arxiv.org/abs/2209.03003)，不加网络模块：
保留每条教师轨迹生成时的源噪声z，学生沿 `x_t=(1−t)x_teacher+t*z` 回归 `z−x_teacher`。
F3-reflow2用已有2步教师坐标作匹配对照；F3-reflow8只换教师为8步。
两组学生部署仍2步，原感知和评分均冻结，相同1024步、batch128、LR2e-5和候选索引顺序。
首轮限定于当前部署固定的32个源噪声，研究的是这个经验源分布到教师输出分布的传输；
不能声称验证了任意新噪声的连续高斯分布泛化。

此处不叠加效用权重：在确定的配对上加权会同时改变源边缘分布，而部署仍均匀使用32个源，
会破坏源分布一致性。若后续需优化候选概率，应另行构造边缘分布匹配的耦合，不能直接拼接F2和F3。
判据先看教师候选覆盖是否被保留，再看原评分是否正常；生成增益不得冒充最终选择增益。

两组已完成，学生均2步：

| 第一集普通448 | 安全候选覆盖 | 选中碰撞 | 前1m碰撞 | 最优候选效用 | 选中效用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原v6 | 429 | 31 | 3 | 0.20477 | 0.11377 |
| reflow2 | 428 | 35 | 4 | 0.19125 | 0.08509 |
| reflow8 | 433 | 35 | 6 | 0.23210 | 0.07966 |

| 第二集普通896 | 安全候选覆盖 | 选中碰撞 | 前1m碰撞 | 最优候选效用 | 选中效用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原v6 | 851 | 86 | 14 | 0.20422 | 0.04536 |
| reflow2 | 847 | 88 | 11 | 0.18626 | 0.03544 |
| reflow8 | 864 | 93 | 13 | 0.24219 | 0.01546 |

reflow8在两套数据上保留了8步教师大部分候选覆盖收益，但原评分器的最终选择仍不如原v6，尚不晋升。
侧后组的选中碰撞：第一集原21、reflow2为20、reflow8为21；第二集原4、reflow2为3、reflow8为4。

### F3收益是否真的能被当前输入识别

为避免把全地图教师上限误当成可部署收益，检查选错安全候选时，碰撞点在实际缓存的完整历史几何中是否有证据。
`reflow-visibility.json` 复用当前输入场与源地图逐点对应：

| 普通状态中的“有安全候选却选碰撞” | 状态数 | 碰撞点有观测覆盖 | 碰撞点有已识别负间隙 |
| --- | ---: | ---: | ---: |
| 第一集原v6 | 12 | 2 | 0 |
| 第一集reflow8 | 20 | 2 | 0 |
| 第二集原v6 | 41 | 1 | 0 |
| 第二集reflow8 | 61 | 3 | 0 |

这不能支持“评分器忽视已识别障碍”的结论；大部分错误涉及当前显式几何未覆盖的位置。
也不能据此证明神经场景特征完全没有预测线索。后续不直接添加硬碰撞筛选或全地图推理输入。
评分变化的现有MPC几何诊断也未显示明显初始速度收益：第二集平均期望速度原0.34050m/s、
Huber 0.34070m/s；这些是控制参考值，不是闭环速度。没有证据支持立即给教师追加曲率惩罚。

### S3 — 评分器表征是否是瓶颈（预登记）

仅校正最后MLP的在线Huber已失败。再做一个范围明确的对照：保持原v6全部候选与感知不变，
训练现有评分器的全部层，使用S1-centered相同的32候选标签、centered MSE、LR1e-4、
seed20260918和每步256状态；以microbatch8累积梯度，不增加网络模块或目标。
先完成512步，检查256/512步，与已完成S1相同阶段比较；第一集选步数后只读第二集。
此实验检验最后MLP之前的表征适配是否有作用，不预设隐藏障碍能够被完全预测。
若仍无稳定收益，不追加更多评分头。物理安全/候选覆盖不因这个实验改变。

S3完成：第一集选择256步，普通碰撞31/448、前1m为5（原3）、效用0.10867（原0.11377），
侧后碰撞17/164。第二集普通碰撞84/896、前1m为14、效用0.03757（原0.04536），
侧后碰撞8/127（原4）。512步第一集效用进一步下降。
因此，完整解冻评分器未优于只训练末端MLP，也未稳定胜过原v6；不送入在线队列、不合入。
初始607状态评分逐元素匹配原缓存；吞吐约209.5状态/s、峰值12.90GiB。
结果文件：`outputs/goal-spatial-search-20260916/full_critic_centered/holdout.json`。

已排队 `online-reflow100/run.sh`：等待当前两组评分在线评估结束后，在GPU1按原16环境/100任务协议评估reflow8学生。
使用仓库已有 `--checkpoint-queue` 常驻场景服务，后续符合条件的完整模型可复用这次加载；
没有改写正在运行的评估器或给生产代码新增队列机制。此项是检验局部重规划是否能利用候选收益，
不是宣称其离线最终选择已经胜出。

### S4 — Reflow候选分布变化后的评分校准（预登记）

只追踪F3已经证实的候选覆盖收益，不扩展其他结构。固定reflow8学生和原感知，
用其实际两步/32候选重新缓存训练候选及原全地图教师效用，再按S1-centered完全相同协议
仅拟合现有末端MLP。相对于F3，唯一变量是匹配其候选分布的评分校准；不新增头或教师项。
验证仍复用已保存的reflow8候选，原critic重放必须逐元素匹配；训练候选仅生成一次。
同时用原S1-centered权重离线重评分同一候选集合，区分直接组合与分布适配的作用。
只按第一集阶段选点，第二集用于机制确认（已不是盲测）；是否在线确认取决于这些结果及F3闭环结果。
实现复用 `prepare_scores.py --generator reflow8` 与 `train_scores.py centered_mse --generator reflow8`。

S4完成，按第一集规则选256步。固定Reflow候选的比较如下：

| 评分器 | 第一集普通碰撞/448 | 前1m | 普通效用 | 侧后碰撞/164 | 第二集普通碰撞/896 | 前1m | 普通效用 | 侧后碰撞/127 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原评分器 | 35 | 6 | 0.07966 | 21 | 93 | 13 | 0.01546 | 4 |
| 直接移入S1-centered | 32 | 7 | 0.09334 | 18 | 87 | 15 | 0.04090 | 4 |
| Reflow候选上校准末端 | 30 | 5 | 0.10615 | 22 | 84 | 16 | 0.05486 | 2 |

候选分布适配确实改善离线选中效用，但前缀和近目标指标仍混合；不宣称已胜过原v6。
两套缓存原评分重放严格一致。作为与原Reflow评分器的配套闭环对照，将校准模型也加入常驻队列，
分别报告生成器单独改变与生成器/评分校准联合改变，不把联合收益归给单个模块。
本轮在此停止增加新训练分支，等在线确认和独立种子结果再决定是否保留；不因GPU闲置而重复训练。

<!-- online-round-live-start -->
### 本轮在线队列自动汇总

同一场景、同100任务、16环境、相同控制/渲染协议。只列已完整结束的模型；不是跨场景泛化结论。

| 模型 | 成功/100 | SPL | 共同成功任务SPL差（相对原v6） |
| --- | ---: | ---: | ---: |
| 原v6 | 95 | 0.859417 | — |
| huber | 92 | 0.837225 | +0.001344 |
| centered_mse | 96 | 0.877030 | +0.010685 |
| reflow8 | 95 | 0.872266 | +0.007848 |
| centered_seed19 | 93 | 0.849319 | +0.001634 |
| mse_control | 91 | 0.835341 | +0.004088 |
| reflow_centered_calibrated | 97 | 0.887106 | +0.011422 |

待完成：无；该队列完成，释放本次常驻服务。
新增/丢失成功任务与配对区间见 `outputs/goal-spatial-search-20260916/online-queue-comparison.json`。
不自动替换正式模型；单场景最佳值仍需独立种子与更多场景确认。
<!-- online-round-live-end -->

## E001 — separated PointGoal with compressed scene memory

- Hypothesis: four learned metric depth frames, target-independent scene memory,
  and PointGoal used only as a cross-attention query are sufficient for one
  deterministic MeanFlow B-spline to imitate safe experts.
- Model/data: 41.5M parameters; 14 scalar Cartesian-control tokens; 64 learned
  scene slots; sole improved-MeanFlow loss; source-C-space-certified HSSD data
  with 25,777 train and 6,064 validation samples.
- Training: 4x RTX 4090, global batch 1024, 8,000 steps / 200 epochs; stable
  throughput about 5.1k samples/s; final loss 0.00873; no OOM, NaN, or restart.
- Artifact: `outputs/offline-evaluation-geometry-first-step8000-20260901` on
  `go-4x4090`, checkpoint step 8000, EMA weights.
- Strict offline result: ADE 0.03393 m; source collision 14.97%; margin violation
  19.90%; forward-direct collision 3.60%; forward-detour 34.56%; rear-goal
  71.90%; expert-moves-away 73.99%. Predicted curvature variation is 191.64
  1/m2 versus 7.27 for the expert. Batch-1 inference is 45.18 ms P50.
- Root conclusion: training converged, but the architecture did not establish a
  metric correspondence between a 2-D control point and raw depth surfaces.
  Independent scalar control tokens plus positionless learned scene compression
  fit common goal-directed paths while underfitting obstacle-conditioned modes.
  Cartesian B-splines are not by themselves the root cause: SanD uses the same
  representation but explicitly samples and geometrically selects candidates.
- Decision: reject this checkpoint for online scoring. Preserve the one-loss,
  one-step MeanFlow contract; replace scalar trajectory tokens and positionless
  scene compression with a direct metric depth-to-2-D-control interaction, then
  run a short source-truth gate before another full training.

## E002 — direct metric depth-to-plan interaction

- Hypothesis: retain all four frames as metric-positioned visual tokens and use
  shared 2-D control tokens so every predicted control point can directly attend
  to obstacle geometry, while PointGoal remains a separate intent query.
- Model/data: 33.8M parameters; seven 2-D Cartesian B-spline control tokens;
  384 depth tokens plus three history-state tokens; one deterministic improved
  MeanFlow step and MeanFlow imitation as the only training objective. Same
  25,777/6,064 source-certified train/validation data as E001.
- Training: 4x RTX 4090, global batch 1024, 8,000 steps / 200 epochs; stable
  throughput about 4.05k samples/s; final loss 0.01508; no OOM, NaN, restart, or
  skipped update.
- Final artifact: `/home/hsb/navigation_three_projects/curvenav/outputs/offline-e002-step8000`
  on `go-4x4090`, checkpoint step 8,000, EMA weights. Full 6,064-sample audit
  took 35.32 s including three diagnostic interventions; base-policy batch-32
  throughput was 920.7 observations/s and batch-1 inference was 29.28 ms P50.
- Final source-truth result: ADE 0.03514 m; full-path collision 13.98%; first
  0.5/1.0 m collision 0.066%/0.676%; mean first-collision distance 2.740 m.
  Forward-direct collision is 2.79%, forward-detour 33.26%, rear-goal 70.0%,
  and expert-moves-away 73.99%. Fixed-MPC mean desired speed is 0.443 m/s and
  curvature limits 5.80% of plans.
- Causal audit: swapping the complete four-frame depth condition between paired
  real samples changes the path by 0.1225 m and raises full/first-1m collision
  to 32.49%/3.68%; swapping only real PointGoals changes it by 0.1373 m and
  raises collision to 31.13%/1.81%. The policy uses both inputs; depth removal
  is more damaging to immediate safety, so “depth is ignored” and “PointGoal is
  the sole override” are both rejected.
- Failure localization: 848 predicted curves collide in source truth, but only
  528 (62.26%) contain a collision point that the four-frame raw depth field
  itself recognizes; recognized points are 38.30% of all source-collision
  points. Together with 0.676% first-metre collision, this separates visible
  generation errors from an unobservable distant tail. Full-path collision is
  a strict risk metric, not a prediction that the receding-horizon robot will
  immediately collide.
- Visibility/terminal audit on the same 6,064 outputs: among the 848 colliding
  curves, 457 (53.89%) hit an obstacle already recognized by the current frame,
  71 (8.37%) are recognized only after adding history, and 320 (37.74%) remain
  unrecognized by all four frames. The physical endpoint collides in 458
  curves; at the endpoint itself 125/22/311 are
  current-visible/history-only/unrecognized.
  The final 0.25 m contains a collision in 544 curves, but only 110 collisions
  are confined to that terminal window. History is causally active but weak:
  it changes the path by 0.0413 m on average, lowers collision from 15.09% to
  13.98%, and makes 22/74 (29.73%) current-only trajectories safe when their
  collision obstacle is visible only through history.
- Root conclusion: training after step 3,200 reduced loss from 0.08861 to
  0.01508 but changed full collision only from 14.07% to 13.98%. The remaining
  failure is therefore not insufficient optimization. Low ADE also hides a
  derivative error: predicted curvature variation is 235.01 1/m2 versus 7.27
  for experts, and full-path P95 maximum curvature is 21.74 1/m versus 2.53.
  Absolute Cartesian control regression learns position closely but does not
  preserve the strong neighbouring-control correlations that define a stable
  tangent field; detour/rear modes remain underfit. The fixed MPC sees only the
  first 12 points, where P95 curvature is 1.22 1/m and only 5.80% of plans are
  curvature-limited. Thus the derivative defect is real but is not yet proven
  to lie in the executed prefix; only a real closed loop can determine whether
  distant errors repeatedly enter it.
- Decision: do not add a clearance loss, scorer, or hard curvature projection.
  Keep E002 frozen until a fixed online trace is available. If online confirms
  replanning instability, the next single architectural experiment is an
  invertible Euclidean control-increment Flow coordinate, whose linear
  cumulative decode directly supervises B-spline tangent geometry without a
  constraint or second objective.
- Online status: not yet measured. On 2026-09-01 the 3090 endpoint was
  unreachable from both the restored local aTrust route and `go-4x4090`; this
  is infrastructure state, not a CurveNav result. Do not transfer any older
  checkpoint's closed-loop score to E002.

## E003 — geometry-first path-relative incremental MeanFlow

- Hypothesis: E002's remaining visible collisions and derivative error come
  from two coupled representation defects: absolute controls are poorly
  conditioned, and PointGoal changes scene retrieval before obstacle geometry
  is grounded. Replacing both boundaries inside the one generator should lower
  visible/history-only collision without a safety loss or selector.
- Pre-implementation evidence: standardized absolute-control covariance has
  condition number `23800.8`; standardized increments reduce it to `278.2`
  (`85.6x`). E002 has 528/848 colliding curves with raw observable collision
  evidence, so partial observability alone cannot explain its failure.
- Architecture: standardized unconstrained control increments; goal-independent
  depth cross-attention; multiplicative PointGoal modulation only after geometry;
  the exact MeanFlow estimate `e_hat=z_t-t*v_theta` supplies physical control
  anchors for path-relative attention in the interval-average field. One
  improved-MeanFlow loss and one deterministic inference call remain.
- Rejected alternatives: DPNet dead-end head/virtual wall/primitive library;
  SanD/NavDP candidate scorers; clearance penalties; hard horizon, curvature,
  or collision projection. They add a second decision mechanism without fixing
  the generator's representation boundary.
- Acceptance gates: full mathematical and gradient suite; deterministic compile
  smoke; at least `3000 samples/s` steady 4x4090 throughput; then the complete
  6064-sample source-truth audit. Primary comparisons are current-visible,
  history-only, first-1m collision, detour collision, derivative quality, ADE,
  and arc-length/progress. Fixed 10-episode online evaluation remains mandatory.
- E003a rejection before full training: `100 passed, 4 skipped`; remote compiled
  CUDA smoke passed and four-RTX-4090 training was numerically stable at
  approximately `3.65-3.88k samples/s`. A step-2400 audit exposed a violated
  intent contract before spending all 8,000 steps. PointGoal distance is greater
  than the `3.6m` local horizon in 60.26% of train samples and reaches `23.31m`.
  The raw Cartesian goal MLP produced validation intent norms up to `56.08`.
  Across all intent channels/blocks, factors `1+scale(intent)` were negative for
  0.22% / 8.25% / 18.49% of samples in distance bins `(0,3.6]`, `(3.6,7.2]`,
  and `(7.2,inf)`. Thus distant mission scale could flip or amplify a local
  grounded update. The step-2400 checkpoint is retained as a rejected numerical
  artifact and is not a navigation result.
- Root correction for E003b: encode PointGoal as unit Cartesian direction plus
  `log1p(distance/3.6m)`, then RMS-normalize the intent embedding. This preserves
  direction and continuous distance without clipping or a trajectory-length
  constraint, while removing distance-proportional feature gain. No loss, head,
  inference branch, or model selection mechanism is added.
- E003b validation/start: full CPU regression is `101 passed, 4 skipped`; the
  remote compiled CUDA forward/backward/JVP/optimizer/deployment test is
  `1 passed in 32.71s`. Four-RTX-4090 training restarted from step zero with the
  same global batch 1024. At step 80 the loss is `0.91346`; post-compile
  throughput is `3.57-3.71k samples/s`, all four cards hold about `18.31GiB`,
  and no NaN, OOM, skipped update, or restart has occurred. The complete
  step-8000 source-truth and fixed online evaluations remain pending.
- E003b step-800 intent audit on all 6,064 validation goals passes the root
  numerical gate. Intent-norm min/median/max is `19.699/19.705/19.726`. For
  goal-distance bins `(0,3.6]`, `(3.6,7.2]`, `(7.2,inf)`, median absolute
  modulation is `0.532/0.518/0.480`, P99 is `1.966/1.777/1.651`, and negative
  `1+scale` fractions are `10.32%/8.92%/6.31%`. Unlike E003a, modulation no
  longer grows with global goal distance. Negative learned feature modulation
  still exists, but it is bounded by normalized intent and is not a far-goal
  numerical shortcut; navigation safety remains an empirical evaluation gate.
- The same audit at step 2,400 remains stable: intent norm is approximately
  `19.72`; near/mid/far median absolute modulation is
  `0.683/0.651/0.582`, P99 is `2.614/2.147/1.932`, and negative-factor fractions
  are `16.85%/14.85%/9.72%`. Learned modulation strengthens during training but
  still decreases rather than explodes with mission-goal distance.
- Design limitation to test rather than conceal: the same-block cross-attention
  is goal-independent before its PointGoal modulation, but a previous block's
  modulated residual is the next block's query. E003 is therefore a local
  geometry-before-intent ordering bias, not a formal non-interference theorem.
  Persistent current-visible collisions would directly reject this factorization.
- E003b completed all `8,000` updates on four RTX 4090s with global batch 1024,
  final loss approximately `0.01465`, steady-state throughput about
  `3.66-3.73k samples/s`, and no NaN, OOM, skipped update, or restart. The EMA
  checkpoint is `541,696,099` bytes and the strict evaluator independently
  reports `checkpoint_step=8000`.
- Complete 6,064-sample source-C-space evaluation rejects the safety part of
  the hypothesis. E003b versus E002 is: ADE `0.03580` versus `0.03514 m`, full
  collision `14.17%` versus `13.98%`, detour collision `32.92%` versus
  `33.26%`, rear-goal collision `77.62%` versus `70.00%`, and moves-away
  collision `78.03%` versus `73.99%`. The `0.18 pp` aggregate change is below
  one binomial standard error and is not an improvement. Current-visible
  colliding trajectories remain `450`; history-only collisions fall from 71
  to 46 while unrecognized collisions rise from 320 to 363. PointGoal and
  depth swaps still change paths by `0.1359 m` and `0.1230 m`, so neither input
  is disconnected.
- The incremental coordinate is useful but insufficient: curvature variation
  falls `235.01 -> 159.71 1/m2`, P95 maximum curvature `21.74 -> 15.29 1/m`,
  and first `0.5/1.0 m` collision falls `0.066/0.676% -> 0.016/0.511%`.
  Endpoint collision nevertheless rises `7.55% -> 7.97%`. This local-prefix
  improvement may matter under receding-horizon execution, so fixed online
  evaluation is still required, but it cannot be reported as full-path safety.
- A separate read-only geometry audit rules out the seven control anchors as
  the dominant error. On the validation expert distribution, control-to-curve
  nearest distance is `0.016 m` mean, `0.049 m` P95, and `0.096 m` P99, below
  the model's local metric-token scale. Densifying this attention would add
  substantial cost without addressing the observed failure.
- Root conclusion: marginally normalizing PointGoal and ordering one block's
  depth read before its goal modulation fixes the far-goal numerical shortcut,
  but does not create a target-independent traversability decision. Goal-
  modulated residuals become the next block's scene query, and the goal-shaped
  instantaneous proposal also defines the path-relative lookup. The surviving
  current-visible shortcuts and severe rear/moves-away failures are direct
  evidence that the generator still fits progress more easily than rare safe
  detours. Do not add DPNet virtual walls, a scorer, more candidates, a
  clearance penalty, or dense path stages in response. First obtain the fixed
  closed-loop trace; any E004 must replace this factorization as one coherent
  architecture rather than layer another safety mechanism on top.
- Final artifacts are `outputs/offline-e003b-final/offline-metrics.json`,
  `offline-cases.json`, `offline-cases.svg`, and `offline-cases.png`. Fixed
  online evaluation is still unavailable: the 3090 has no route from the 4090,
  and the local aTrust process is not running. No older checkpoint's online
  result is assigned to E003b.

## E004 — metric goal-reference path interaction

- Hypothesis: E003b fails because a learned PointGoal modulation repeatedly
  biases the hidden state and later scene queries. Replacing that semantic
  branch with a physical goal-reference corridor should reduce already-visible
  collisions while retaining the well-conditioned incremental B-spline and the
  sole improved-MeanFlow objective.
- Architecture delta: remove the PointGoal MLP and every per-block FiLM. The
  seven Greville controls of the exact straight B-spline from the origin to
  `min(||g||,3.6m)` provide the first field's metric path-relative queries. Flow
  transports standardized expert-minus-reference control increments, so its
  learned signal is route-shape deviation rather than common goal progress. The
  instantaneous field's exact clean estimate provides the second field's
  queries. The affine residual coordinate is unbounded and does not constrain
  output length or heading. No loss, head, candidate, scorer, projection, or
  inference pass is added.
- Final pre-run audit also removed a provisional learned embedding of the
  reference controls before it produced any optimizer step. That embedding
  would have recreated a goal-only hidden-state path. In the accepted graph,
  PointGoal affects learned computation only through scene-relative metric
  query geometry; its other role is the analytic affine coordinate origin.
- Pre-training identifiability audit on all 6,064 validation samples: the exact
  goal-reference spline violates the `0.10 m` source margin in 45.66% of cases.
  Safe-reference versus blocked-reference expert residual RMS is `0.329` versus
  `1.027` (`3.13x`); lateral residual RMS is `0.171` versus `1.083` (`6.35x`).
  Although blocked references are 45.66% of samples, they carry 91.26% of the
  standardized residual-coordinate squared energy. Thus the sole Flow target
  now gives obstacle-conditioned detours the dominant learning signal rather
  than letting common straight progress swamp them. This is evidence of
  identifiability, not a claim of guaranteed collision freedom.
- Pre-training verification: source dataset recompilation preserves all expert,
  depth, goal, and provenance arrays while replacing only the coordinate
  contract; `102 passed, 4 skipped` locally. A compiled RTX 4090 BF16
  forward/backward/JVP/optimizer/deployment smoke passes with finite loss and a
  finite `[4,64,2]` path.
- Run status: E003b's completed checkpoint was preserved as
  `outputs/train_policy-e003b-step8000` on `go-4x4090`. E004 started from zero
  on four RTX 4090s with global batch 1024. The first checkpoint is step 800;
  training will be stopped there for the complete source-truth gate before any
  continuation to step 2,400 or 8,000.
- Evidence gate: before a full run, tests must prove that scene memory is target
  independent, the target enters only through metric references, both fields
  are path-relative, and one-step/JVP gradients are finite. A step-800 then
  step-2400 complete source-truth audit must show a substantial reduction in
  current-visible, rear-goal, and moves-away collision; otherwise reject E004
  without spending 8,000 steps. The final unchanged physical target is at most
  1% full source-C-space collision, reported without excluding OOB, endpoints,
  unobserved regions, or short trajectories.
- Rejection: the step-800 gate had loss `0.486` and source collision `33.53%`
  (first 0.5/1.0 m `4.39%/12.48%`, detour `70.93%`, rear-goal `92.38%`).
  This checkpoint was under-trained and is not used as a performance verdict.
  The architecture was rejected before continuation for a structural reason:
  its codec always added the straight PointGoal reference back to generated
  residuals. That makes model error physically default toward the exact unsafe
  corridor the model is supposed to reject, contradicting the intended
  geometry-before-goal factorization.

## E005 — observed C-space BEV with goal-indexed physical increments

- Hypothesis: a single expert-imitation generator can learn visible avoidance
  only if robot-footprint-aware geometry is explicit, target independent, and
  upstream of goal selection, while its output coordinates contain no analytic
  goal trajectory.
- Architecture: all four frames share one ResNet-18-style encoder and calibrated
  SE(2) projection. Learned visual evidence is metric-splatted and fused with a
  64x64 observed Dingo C-space into one 16x16 BEV. Unknown EDT extrapolation is
  masked. Seven PointGoal Greville anchors are relative-attention queries only.
  Flow generates standardized physical B-spline increments. A four-block
  shared trunk feeds parallel four-block instantaneous and average iMF fields.
- Objective/inference: unchanged sole improved-MeanFlow imitation objective,
  exact deployment-boundary quarter and fixed typical latent; one deterministic
  call. No clearance loss, candidate, critic, scorer, projection or fallback.
- Root bug fixed during preflight: normalized max-range depth was previously
  excluded from visibility even though it is calibrated negative ray evidence.
  It is now observed-free up to 5 m while still excluded from obstacle/surface
  classification. Validation expert-point support rises to `81.86%`, fully
  observed expert curves to `51.10%`, and median contiguous observed prefix to
  `2.31 m`.
- Pre-training gate: the complete local suite passes `104 passed, 4 skipped`
  (two optional plotting imports and two local-CUDA tests). The RTX 4090
  compiled BF16 forward/backward/JVP/optimizer/deployment test passes in
  `153.61 s`. The prepared manifest is rebuilt with the exact physical-increment
  statistics. The final 4x4090 run started from step zero with global batch
  1024; steps 20/40/60 report `4.079/4.265/4.278k samples/s`, about
  `16.7 GiB/GPU`, finite loss and no skipped update. Its first physical
  checkpoint is step 800.
- Acceptance: first-1m and current/history-visible source collision below 1%,
  with ADE, arc length, progress and fixed-MPC curvature non-regressing. Full
  3.6m collision remains reported, but local depth alone cannot guarantee its
  unobserved tail; no score may hide OOB, endpoints or unrecognized collisions.
- Final run: step 8,000 completed without a skipped update. Final loss is
  `0.02051`; the last logged four-GPU throughput is `4.054k samples/s`.
  Batch-one model latency on one RTX 4090 is `38.66 ms` median / `38.96 ms`
  p95 (`25.86/25.67 Hz`), and batch-32 policy throughput is `710.90
  observations/s`.
- Final source-truth result: ADE is `0.03395 m`; collision in the first
  `0.5/1.0 m` is `0.033%/0.297%`. Forward-detour first-1m collision is
  `0.907%`. These execution-prefix gates pass and are materially better than
  E003. Full 3.6m collision is still `12.24%` (`27.99%` forward-detour,
  `66.67%` rear-goal), so the complete architecture is not accepted as a
  sub-1% full-horizon generator.
- Failure location is now explicit: mean distance to first collision is
  `2.76 m`; endpoint and terminal-0.25m collision are `7.40%` and `8.58%`.
  Of 742 colliding trajectories, 399 are unrecognized by all four depth frames,
  328 have current-frame evidence and 15 only historical evidence. Depth and
  PointGoal swaps change the path by `0.128 m` and `0.126 m`, respectively;
  the model no longer exhibits a simple PointGoal-only shortcut. The remaining
  problem is predominantly far-tail observability and long-horizon shape
  accuracy, not failure to generate a safe immediate prefix.
- Executability audit: the fixed controller's first-12-point curvature p95 is
  `1.153 m^-1` and only `4.90%` of plans are speed-limited, but full-curve max
  curvature p95 is `17.48 m^-1` versus `2.53 m^-1` for experts. This tail-shape
  gap must be distinguished from immediate collision before changing the
  trajectory coordinate system. The next evidence gate is the unchanged
  fixed online 10-episode protocol; no safety loss, selector or hard curvature
  constraint is added from this offline result alone.
- Fixed online result: checkpoint step 8,000, Home scene, seed 1234 and ten
  episodes produced `SR=1/10` and mean `SPL=0.092984`. Nine failures timed out.
  The policy and MPC did not output zero: plans averaged about `2.82 m`, MPC
  desired speed averaged about `0.26 m/s`, and curvature limiting was active on
  91% of plans. Yet actual motion repeatedly stalled. Closed-loop plans had
  p95 peak curvature about `310 m^-1`, far beyond both teacher-forced prediction
  (`17.48 m^-1`) and expert (`2.53 m^-1`), while adjacent replans disagreed by
  only `5.3 mm` over the first metre. The model was stably generating an unsafe
  out-of-distribution shape, not randomly jittering or being actively stopped.
- Evaluation audit: official success, timeout, SPL, PointGoal transform, output
  origin insertion and MPC axes are internally consistent and cannot turn a
  successful run into this 1/10 result. Two bugs existed only in the added map
  diagnostic: PLY samples were shifted by half a cell, and the current robot
  origin was included in every future-plan collision query. They inflated the
  diagnostic collision percentage but did not change official SR/SPL.

## E006 — Flow-candidate-grounded observed geometry

- Root defect: E005's function named `_path_relative_geometry` was anchored at
  the straight PointGoal reference, not at the current Flow state. The decoder
  therefore read obstacles around the desired corridor but never queried
  whether its own noisy/candidate B-spline occupied them. This breaks the
  state-conditioned geometry principle used by NavDP's noisy-trajectory
  queries and leaves an easy PointGoal shortcut.
- Single architectural correction: decode every Flow state into its physical
  64-point B-spline and query the deployed observed C-space along that curve.
  Aggregate path geometry to the seven controls using the fixed positive
  B-spline basis. Scene relative attention is also anchored at candidate
  controls. PointGoal appears only as the relative vector from each candidate
  control to its metric local goal reference.
- Mathematical contract: the decoder is now
  `u_theta(z,r,t,D,g)=u_theta(z,r,t,M_D,H(z,D),G(z,g))`. The stopped JVP includes
  the differentiable `z -> B-spline -> bilinear C-space query` path almost
  everywhere. Unknown field corners contribute zero geometry and explicit zero
  coverage, not extrapolated free space.
- Scope: one Transformer, one improved-MeanFlow loss and one deterministic
  inference call remain. No clearance loss, hard constraint, candidate set,
  critic, selector, refinement pass or fallback was added. The architecture and
  checkpoint type change, so E005 weights are intentionally incompatible.
- Verification/start: focused forward, condition-causality and gradient tests
  pass; a compiled RTX 4090 BF16 forward/JVP/backward/optimizer/deployment smoke
  passes with finite loss and a finite `[4,64,2]` path. Training started from
  zero on free RTX 4090 GPUs 0--2 with exact global batch 1024; GPU3 belongs to
  another user and was not touched. Steady steps 40--100 reach
  `3.19--3.25k samples/s`, about `21.7--21.8 GiB/GPU` and 95--100% sampled GPU
  utilization; loss falls from `2.54` at step 1 to `1.07` at step 100. No E005
  score is attributed to E006.
- Step-800 source-truth gate: loss is `0.314`; ADE `0.04783 m`; full collision
  `14.87%`; first `0.5/1.0 m` collision `0.000/0.841%`; forward-detour first-1m
  collision `1.586%`; full detour/rear/moves-away collision
  `33.65/71.90/74.57%`. Full-curve curvature p95 is `14.37 m^-1` and fixed-MPC
  curvature limiting `5.24%`. This early checkpoint is not yet better than the
  converged E005 and its loss is far from convergence, so it does not establish
  the hypothesis; training resumed exactly from step 800 for the next gate.
- Final run: step 8,000 is readable with checkpoint SHA256
  `f3025da23a7ee67ece9eb302bc30118ab1623cfa3c7ff4de7e9c2e6bbc71a35b`.
  Three RTX 4090s sustain about `3.1k samples/s` with approximately
  `21.7--21.8 GiB/GPU`; the final logged loss is approximately `0.014`.
  The source-truth result is ADE `0.03448 m`, full collision `12.78%`, first
  `0.5/1.0 m` collision `0.016/0.396%`, and forward-detour first-1m collision
  `0.963%`. Full forward-direct/detour/rear/moves-away collision is
  `2.52/29.07/75.71/79.77%`. Endpoint and terminal-0.25m collision is
  `7.85/9.10%`; full-curve maximum-curvature p95 is `15.61 m^-1` versus
  expert `2.53 m^-1`. Batch-one latency is `43.90 ms` median / `45.46 ms`
  p95 (`22.78/21.997 Hz`).
- Offline decision: E006 did not improve teacher-forced source safety. Relative
  to E005, full collision changes `12.24 -> 12.78%`, first-1m collision
  `0.297 -> 0.396%`, detour first-1m collision `0.907 -> 0.963%`, and ADE
  `0.03395 -> 0.03448 m`. Querying the current Flow state is therefore not a
  direct final-trajectory/geometry contract: at one-step deployment that state
  is still the fixed source latent.
- Fixed online result: E006 reaches `5/10` success and mean SPL `0.450821` under
  the same Home scene, seed 1234 and `num-envs=1` protocol where E005 reached
  `1/10` and `0.092984`. Candidate-relative C-space is therefore retained as a
  real closed-loop improvement even though its teacher-forced safety did not
  improve. The five failures are timeouts. Failed and successful episodes have
  comparable nonzero command magnitude, but failed actual speed collapses to
  about `0.023 m/s`; replans remain stable and repeatedly point through blocked
  corners. This localizes the next defect to generated-trajectory geometry, not
  random planning, zero commands or the official metric.
- Online evaluator root audit: the real persistent-scene implementation passes
  raw `distance_to_image_plane` depth and the simulator `3x3` intrinsic matrix;
  PointGoal is `R^T(goal-world - robot-world)` and CurveNav consumes its body
  `x/y` directly; the runtime returns 63 future points and the evaluator inserts
  exactly one origin before the unchanged MPC. `EvalTerminationsCfg` contains
  only arrival and timeout, while trace aggregation independently recomputes
  every SR/SPL row. Thus no metric, axis, transport or termination bug can turn
  a genuine success into the E005 `1/10` result. The YAML field
  `arrival_threshold: 1.0` is unused: the executable fixed condition is
  `<0.5 m`, `<0.25 m/s`, then 40 accumulated Dingo control steps. Its upstream
  timer latches once started, which can only inflate success and remains
  unchanged for cross-model comparability. CurveNav's server now uses the
  supplied intrinsic matrix. Its separate map diagnostic now uses native 5 cm
  cell centres, a half-open raster extent, and excludes the current origin from
  future collision; these corrections do not alter official SR/SPL.
- Paper falsification before the final gate: NavDP grounds denoising tokens in
  the current noisy trajectory, which motivates E006, but it updates that
  trajectory over multiple denoising steps. The contemporaneous FlowPilot
  result instead couples a future-depth stream and a polynomial-action stream
  through shared attention; its reported depth-frozen/action-only intervention
  raises collision from `4%` to `26%`, and deployment normally uses three Euler
  steps. This is evidence that explicit future-geometry/action co-training can
  matter; it is not evidence that a loss term should be appended to E006. If
  the converged E006 still fails on currently visible prefix collisions, the
  next architectural hypothesis must compare one-step candidate grounding
  against a single jointly trained geometry-action process. No future-depth
  head, extra denoising step or safety penalty is introduced mid-run.

## E007 — single-call clean-proposal-grounded improved MeanFlow

- Rejected pre-run draft: a parameter-shared three-step MeanFlow was briefly
  implemented but never trained. Its training still gave exact fixed-source
  coverage to `(r,t)=(0,1)`, while deployment called the distinct intervals
  `(2/3,1)`, `(1/3,2/3)`, `(0,1/3)` on policy-updated states. CPU tests proved
  execution, not that statistical contract. It was removed when the production
  requirement was reaffirmed as single-step; no checkpoint or score belongs to
  that draft.
- Root hypothesis: E006's one C-space query is performed on its fixed Gaussian
  source. A one-call generator needs an estimate of its clean endpoint before
  the average field can read output-relevant obstacle geometry.
- Architecture: retain improved MeanFlow's supervised instantaneous field.
  Six proposal blocks use target-independent scene memory and the metric goal
  corridor to predict `v_theta`. The exact linear-interpolant form supplies the
  learned endpoint estimate `x_tilde=z_t-t*v_theta`. It is decoded immediately;
  the next six blocks query observed C-space and BEV at this proposal and use
  candidate-relative PointGoal displacement to predict `u_theta`. Deployment
  remains exactly `x_hat=e*-u_theta(e*,0,1,c)` with one decoder call.
- Objective: unchanged single improved-MeanFlow scalar, the equal mean of
  instantaneous velocity regression and reparameterized interval-average
  regression. The JVP differentiates through proposal decode and geometry
  query. There is no clearance penalty, future-depth loss, critic, candidate
  set, selector, projection, rollout loss or inference loop.
- Mathematical gate: tests must prove one deployment call, distinct reference
  and learned-proposal geometry queries inside that call, structural
  independence of `v_theta` from `r`, fixed-source `(0,1)` parity, finite JVP
  and full-model gradients, and strict checkpoint incompatibility with E006.
- Performance gate: compile BF16 on RTX 4090, measure parameters/latency and
  sustain at least `3000 samples/s`; stop first at step 800 for all 6,064
  source-truth samples. Continue only if first-1m, visible-detour, progress and
  curvature move together. Final evidence remains the unchanged ten-episode
  closed loop, followed by a larger protocol before any `80%` claim.
- Pre-training verification: the complete CPU suite is `108 passed, 4 skipped`;
  the skipped cases require local CUDA or optional plotting. A real RTX 4090
  compiled BF16 forward/JVP/backward/optimizer/deployment smoke passes in
  `256.57 s`, including a finite one-step `[4,64,2]` output. The instantiated
  policy/decoder contain `33,898,916/29,018,980` parameters. Training started
  from zero on the only free RTX 4090 GPU2 with global batch 1024; other users'
  GPUs 0, 1 and 3 were not touched. No offline or online score is claimed yet.
- Step-800 gate: training is finite and sustains `1.00--1.02k samples/s` on one
  RTX 4090 at about `22.1 GiB`. Versus E006 at the same step, ADE changes
  `0.04783 -> 0.04223 m`, full collision `14.87 -> 13.49%`, first-1m collision
  `0.841 -> 0.511%`, forward-detour collision `33.65 -> 31.50%`, and rear-goal
  collision `71.90 -> 63.33%`. Fixed-MPC first-12 curvature p95 improves
  `1.35 -> 1.13 m^-1` and limiting falls `5.24 -> 4.53%`.
- The gate is not yet accepted as a safe model. Full-curve maximum-curvature
  p95 regresses `14.37 -> 21.79 m^-1` and complete comparison-horizon coverage
  falls `97.79 -> 85.32%`, exposing an under-converged short/sharp tail. The
  exact step-800 checkpoint is preserved; training resumes to step 2,400 to
  test whether both defects converge away before any online evaluation.
- Case evidence prevents attributing all remaining collision to unseen space.
  The selected current-visible failure first collides at `0.725 m` and contains
  11 collision points recognized by the current depth frame; the hard detour
  case first collides at `1.30 m` with 10 current-visible and 73 unrecognized
  collision points. E007 therefore still has both a learned visible-geometry
  error and an unobservable-tail error at step 800. Step 2,400 distinguishes
  under-training from a surviving architecture defect; no new loss is added
  while that distinction is unresolved.
- The first continuation from step 800 reached approximately step 1,140 at
  about `1.0k samples/s`, then its remote tmux session exited before the next
  checkpoint. The preserved checkpoint is still exactly step 800. The local
  aTrust tunnel subsequently disappeared, so kernel/tmux exit evidence has not
  yet been retrieved and no later checkpoint is claimed. The next launch must
  retain the dead tmux pane and exit status before continuation resumes.
- Because the 4090 aTrust route remained unavailable, the unchanged E007 code
  was validated on local V100 GPU6: compiled FP16 forward, MeanFlow JVP,
  backward, optimizer step and deployment sample passed in `212.19 s`. A fresh
  two-rank run briefly started on otherwise empty V100 GPUs 6/7 with the same
  global batch 1024. It was stopped cleanly before producing a checkpoint once
  the 4090 route recovered, avoiding two independent trainings of the same
  experiment; its retained tmux pane records signal 2.
- A second, bounded V100 6/7 throughput probe was made after the 4090 resumed.
  Both ranks remained in CUDA/NCCL initialization for more than seven minutes
  with only about `0.63 GiB` allocated per GPU and never emitted
  `training_start`; the probe was stopped cleanly. In the same interval the
  single 4090 advanced from step 800 to beyond 1,060 and sustained
  `1013--1019 samples/s`. For this checkpoint and remaining wall time, the
  measured faster route is therefore the single 4090; no model or launcher
  branch is introduced for the V100 host.
- The 4090 route was restored through the already-running user-namespace
  aTrust `utun7`; authentication had not expired. Remote GPU2 was empty and
  showed no kernel OOM or NVIDIA Xid around the prior exit. Training resumed
  from the exact step-800 checkpoint in `curvenav-e007`, with tmux now retaining
  the dead pane and exit status. The unchanged watcher stops after the step
  1,600 and 2,400 checkpoints.
- Closed-loop rendering was refined without changing SR, SPL, timeout, MPC or
  proxy metrics: episode and scene maps now crop to the actual task region,
  generated plan samples outside frozen robot-center free space are overlaid
  in red, and executed positions outside that proxy are marked separately.
  The complete CPU regression after the evaluation diagnostics is `110
  passed, 4 skipped`.
- Step-2,400 final gate: the resumed single RTX 4090 run sustained
  `1013--1021 samples/s`, saved SHA-256
  `45079517e3a3e6f45d05ab8b3fc9e1a599d73c622b28ddcffc28f6f351bb7ddc`,
  and stopped cleanly. Loss reached `0.04254`. On all 6,064 source-truth
  validation items, ADE is `0.03559 m`, full collision `12.269%`, first
  `0.5/1.0 m` collision `0.016/0.363%`, forward-direct/detour collision
  `2.470/28.045%`, and rear/moves-away collision `70.476/75.145%`.
  Continuing from step 800 improves aggregate collision by only `1.22 pp` and
  worsens rear-goal collision by `7.14 pp`; training duration is not the root
  safety variable.
- Proposal/final falsification: proposal collision is `13.391%`, final
  collision `12.269%`, mean path change is `0.01493 m`, proposal-safe to
  final-collision is `0.792%`, and proposal-collision to final-safe is `1.913%`.
  Hence the PointGoal-conditioned final update is not destroying an already
  safe proposal. The proposal itself never learned a safe manifold. Of 744
  final colliding curves, `45.56%` contain current-frame recognized collision,
  `4.03%` history-only recognized collision and `50.40%` no four-frame
  recognized collision; visible learning failure and unobservable tail are
  both real.
- E007 is rejected as the final architecture. Candidate-relative geometry is
  retained because E006 improved closed loop, but querying geometry while both
  Flow heads optimize only expert coordinate error is not a feasibility
  objective. No additional E007 epochs or online score will be used to hide
  this result. The 3090 VPN route is present but SSH port 22 remains unreachable.

## E008 — deployment-curve observed-feasibility MeanFlow

- Root hypothesis: safe expert labels identify the desired route, but finite
  regression error has no mathematical implication for signed clearance. With
  source-safe expert `p+`, `d(p+)>=0.10 m`, safety would follow only from a
  uniform path error below the margin; E007 has mean FDE `0.0947 m`, so low ADE
  cannot supply that bound. SanD resolves this with ESDF selection, NavDP with
  a learned critic and privileged negative paths, and ViPlanner/iPlanner use a
  differentiable trajectory cost. A single-output CurveNav that rejects those
  inference mechanisms requires direct final-path feasibility training.
- Unique change: keep one E007 decoder call and one output, but evaluate the
  exact deployment-quarter final path against its own raw-depth C-space at
  2.5 cm. The sole added scalar is strict-observed squared clearance deficit
  below `0.10 m`. It has no network parameters and is absent at inference.
  Every bilinear support cell must be observed, so unknown EDT extrapolation
  cannot become free-space evidence. Fixed-grid normalization and detached
  radial scale remove coverage normalization and uniform-shortening shortcuts.
- This is one constrained generator objective `L_MF+L_vis`, not a safety head,
  candidate scorer, hard projection, privileged teacher, fallback or second
  launcher. PointGoal and depth remain joint conditions; “safety first” is the
  feasible-set semantics of the output, not an unsupported depth-only policy
  that must guess one route before seeing the goal.
- Pre-run CPU gate: policy/checkpoint/evaluation tests are `54 passed, 1
  CUDA-skipped`; tests explicitly prove strict unknown masking, clearance
  gradient direction, zero uniform-shortening gradient, exact loss
  decomposition and full-model finite gradients. CUDA compile, throughput and
  the step-800 source-truth gate remain required before any E008 score.
- Step-800 gate: single-4090 steady throughput is `970--994 samples/s` at
  `22.1 GiB`, only about 4% below E007. The frozen checkpoint SHA-256 is
  `a5e2fdce942c490635a01ec31e4f28f811311739b911a20fda16c154f5521dd0`.
  Against E007 at the same step, source collision changes
  `13.489 -> 11.412%`, forward-direct `3.155 -> 1.859%`, detour
  `31.501 -> 28.215%`, rear `63.333 -> 56.190%`, and moves-away
  `64.162 -> 56.647%`. Current-frame-recognized colliding curves fall
  `418 -> 265` (`-36.6%`), direct evidence that final-path geometry gradients
  work.
- The gate is not yet an online candidate. ADE changes `0.0422 -> 0.0505 m`,
  first-1m collision `0.511 -> 0.742%`, first-12 curvature p95
  `1.127 -> 1.341 m^-1`, and mean progress regret grows to `0.0302 m`.
  Unrecognized colliding curves rise approximately `375 -> 400`. A repeated
  support audit measures strict raw-depth coverage `82.57%` for predictions
  versus `84.59%` for their experts (`-2.02 pp`; 33.2% below reference).
  This is a warning, not yet proof of a dominant unknown-space shortcut.
- Decision: resume only to step 1,600, then repeat the complete gate. Accept
  the simple objective only if visible collision keeps falling, ADE/curvature
  recover and the support gap does not expand. Do not append an unknown-space
  penalty from one under-converged checkpoint.
- Step-1,600 gate passes that decision. Versus step 800, full source collision
  changes `11.35 -> 9.96%`, detour `28.10 -> 22.66%`, first-1m
  `0.742 -> 0.462%`, ADE `0.0505 -> 0.0430 m`, progress regret
  `0.0302 -> 0.0091 m`, and first-12 curvature p95 `1.333 -> 1.289 m^-1`.
  Strict observed support recovers from `-2.02` to `-0.36 pp` relative to each
  expert, disproving a growing unknown-space shortcut. Current-visible
  colliding curves fall again from `264 -> 193`; unrecognized curves remain
  nearly fixed at about `388`, exposing the local sensor's remaining limit.
  Checkpoint SHA-256 is
  `e9652616c46b63bccd2f0547f66ef86450fbc86217112b351435860862c6169f`.
- Proposal/final strata also reject the claim that the final PointGoal update
  breaks a safe path: all/detour/rear proposal collision is
  `11.18/25.78/67.14%`, while the final is `9.96/22.66/64.76%`.
  Rear-goal failure is already present in the proposal and is dominated by
  unrecognized collision (`53.81 pp` of `64.76%`); the model also advances
  `0.280 m` more toward the mission goal than the locally retreating expert.
  Without topology memory, this is not identifiable from visible clearance.
- Decision: keep E008 unchanged and continue only to step 2,400 for the final
  convergence gate. Do not add a PointGoal gate, unknown penalty, critic or
  extra stage. The fixed online ten-episode run remains required, but the 3090
  VPN currently reaches the route while SSH port 22 remains unavailable.
- Step-2,400 rejects further blind training. The single RTX 4090 sustained
  `973--979 samples/s`; checkpoint SHA-256 is
  `333b3a0aff2c09b8ef13e595a63111d4a1d456f87d721272be4b05ae400b94cc`.
  ADE improves `0.0430 -> 0.0368 m` and first-1m collision improves
  `0.478 -> 0.379%`, but full collision regresses `9.96 -> 10.41%`, detour
  `22.66 -> 23.40%`, rear `64.76 -> 69.05%` and moves-away
  `65.32 -> 69.36%`. Step 1,600 remains the safer checkpoint; lower imitation
  error is not evidence of better geometric feasibility.
- The earlier current/history/unrecognized trajectory labels meant that any
  collision point on a curve had that evidence, not that its first collision
  did. The corrected first-hit audit partitions every colliding curve exactly.
  At step 1,600, only `52/604=8.61%` first hits are recognized by the current
  frame, `12/604=1.99%` only by history and `540/604=89.40%` by neither. Across
  all 6,064 observations, first-hit current-visible collision is `0.857%` and
  history-only is `0.198%`; the remaining `8.905%` is unrecognized. Step 2,400
  shifts further toward unrecognized first hits (`583/631=92.39%`). Thus E008
  did learn visible avoidance below the 1% observation-level target, while its
  full-path score is now dominated by geometry absent from all four inputs.
- Root decision: do not increase the clearance weight, append an unknown-space
  penalty or add decoder stages. Those cannot identify hidden geometry. At the
  step-2,400 gate, step 1,600 was the provisional safer checkpoint; the later
  completed-run audit below supersedes that checkpoint choice. Treat the first
  `0.5/1.0 m` source-safe prefix as the local receding-horizon execution gate.
  A future architecture change is justified only if online traces show
  first-hit collisions in observed space; hidden long-horizon topology belongs
  to the planned topology-map input, not to a local-depth patch.
- The user-directed complete run reached step 8,000 without changing code,
  data or objective. Single-4090 throughput stayed near `970--980 samples/s`;
  the frozen EMA checkpoint SHA-256 is
  `20972dc0c7f28b6654ec16ada03c6239e46708e4b4ebb8493233dc2ec8b82e73`.
  The full 6,064-item source-truth audit gives ADE `0.03105 m`, full collision
  `11.560%`, first `0.5/1.0 m` collision `0.000/0.396%`, and mean first-hit
  distance `2.764 m`. Relative to step 1,600, imitation and the executable
  prefix improve, while full collision regresses `9.960 -> 11.560%`.
- The regression is localized rather than a new visible-avoidance collapse.
  Step 8,000 has 701 colliding curves: first hit is current-visible for 57
  (`0.940%` of all observations), history-only for 14 (`0.231%`), and absent
  from all four frames for 630 (`10.389%`). Terminal-0.25 m collisions rise
  from about 428 at step 1,600 to 516, accounting for 88 of the 97 added
  collisions; endpoint collision reaches 453. The correct conclusion is that
  the fully predicted 2.9 m tail increasingly enters unobserved space, not
  that the model forgot visible obstacles near the next executed prefix.
- The execution-prefix attribution closes the remaining ambiguity. Of the 24
  curves colliding within the first `1.0 m`, the first hit is current-visible
  for `0`, visible only through history for `1`, and unrecognized by all four
  frames for `23`. Thus no validation example drives the executed prefix into
  a source obstacle that the current raw depth C-space already recognizes.
  The one history-only case is a genuine temporal-fusion residual; the other
  95.83% are not identifiable from this local observation. This metric is now
  reported directly by the evaluator rather than inferred from selected cases.
- Geometry refinement remains real but small: proposal collision is `12.632%`,
  final collision `11.560%`, and their mean path difference is `9.16 mm`.
  Training also reduces curvature variation `594.9 -> 130.4 m^-2` and
  full-curve maximum-curvature p95 `21.46 -> 14.52 m^-1`, but both remain far
  above the expert's `7.27 m^-2` and `2.53 m^-1`. The controller-facing first
  12 points are much better behaved: curvature p95 `1.118 m^-1`, with `4.58%`
  speed-limited. Closed-loop execution, not another offline penalty, must now
  determine whether the remaining tail is harmless under replanning.
- Rear-goal and locally retreating-expert collision remain `70.48/71.68%`.
  The model changes its path by nearly the same amount under a depth swap
  (`0.1322 m`) and PointGoal swap (`0.1328 m`), so neither condition is absent;
  the unresolved ambiguity is that local depth cannot identify when global
  progress requires temporarily moving away from the goal. Do not encode that
  topology as a PointGoal heuristic or an unknown-space penalty.
- The fixed online ten-episode candidate is step 8,000 because it has the best
  executed-prefix collision, ADE and smoothness, despite its worse unobserved
  full tail. Online traces must record first-metre plan feasibility at every
  replan and overlay plans, executed motion and frozen C-space. If unsafe
  observed prefixes precede stalls, the generator remains wrong; if prefixes
  stay safe while the vehicle stalls, investigate deployment coordinates/MPC.
  The 3090 VPN route currently exists but port 22 is unreachable from both the
  workstation and the 4090 host, so this fixed online run has not started.
- Operational note: a one-off watcher used tmux prefix matching and confused
  `curvenav-e008-full-gate` with the completed `curvenav-e008-full` session.
  The step-8,000 checkpoint itself was complete and valid. The watcher was
  stopped and the unchanged evaluator was launched directly; future lifecycle
  checks must use exact tmux session targets rather than prefixes.
- Fixed online result: the exact step-8,000 EMA checkpoint reaches `6/10` SR
  and `0.582788` mean SPL on Home episode `0--9`, seed 1234 and `num-envs=1`.
  Successful episodes are `1,2,4,5,6,8`; failures are `0,3,7,9`. The identical
  first ten episodes give official NavDP `5/10` and X-NavDP `8/10`, so E008 is
  a credible supervised baseline but does not inherit X-NavDP recovery.
  In the four failures, adjacent first-metre replans differ by only
  `0.9--5.1 mm`, desired speed remains `0.36--0.50 m/s`, and stalled-step
  fractions are `66.4/80.3/95.9/98.4%`: stable unsafe intent, not stochastic
  switching or a zero-command controller, is the dominant signature. Frozen
  map overlays and numeric traces are stored under
  `outputs/online-e008-step8000-online10/20260901_115641`.

## E009 — Flow-state-grounded one-call MeanFlow

- Root defect: E008 places the first six blocks' C-space and BEV queries on the
  analytic straight PointGoal corridor. The final six blocks query the learned
  clean proposal, but offline they alter it by only `9.16 mm`; online failures
  repeat one wrong route for the entire timeout. The goal therefore still
  chooses where the first obstacle lookup occurs, contrary to the intended
  target-independent geometry contract.
- Paper/source basis: SanD's conditional U-Net and NavDP's Transformer denoise
  the current noisy trajectory variable while attending to depth/goal context;
  neither substitutes a straight PointGoal path for that variable. Their
  candidate scorer/critic remains deliberately excluded.
- Unique change: decode the affine standardized Flow state `z_t` to its physical
  controls and B-spline, and use those positions for the first C-space/BEV
  query. The learned clean estimate remains the second query. PointGoal is only
  relative intent from each physical control. Model size, block count,
  objective, source distribution, one-call inference and FLOP order are
  unchanged; no loss, head, candidate, projection or fallback is added.
- Mathematical gate: affine coordinate decode commutes with the linear Flow
  interpolant, so every `z_t` has an exact physical query curve and the JVP
  differentiates through both state and proposal queries. The fixed deployment
  source decodes to a finite `2.563 m` forward B-spline. Tests must prove that
  changing PointGoal changes goal features but cannot relocate the Flow-state
  geometry query before CUDA training starts.
- Pre-training verification passed on the current worktree: `114 passed, 4
  skipped` on CPU, followed by a compiled V100 FP16 forward, MeanFlow JVP,
  backward and deployment smoke (`2 passed` in `261.08 s`). A three-V100
  topology on local GPUs 0--2 entered NCCL initialization but made no training
  step; it was stopped before any checkpoint and all ranks exited. The same
  unmodified launcher immediately reached `training_start` on GPUs 0--1 with
  exact global batch 1024, 512 samples per rank, two 256-sample microbatches,
  FP16 and 8,000 target updates. This is a hardware communication-topology
  fact, not a model branch or a CurveNav loss result; throughput and model
  quality remain unreported until real updates are observed.
- Completed run: four V100S GPUs used global batch `1792`, `200` epochs and
  `4,600` optimizer updates, processing approximately the same number of
  examples as E008. Final loss was `0.10050`; the EMA checkpoint SHA-256 is
  `34b899677f32698a785fcd1ec5c8c0a1c0395a1cb344a95bde7ff69371320d2b`.
  Source-truth offline ADE/FDE are `0.03622/0.09933 m`, full collision is
  `11.807%`, and first `0.5/1.0 m` collision is `0.049/0.577%`. These are only
  slightly worse than E008 and do not predict the closed-loop regression.
- Fixed online rejection: on the exact resident Home scene, episodes `0--9`,
  seed `1234` and `num-envs=1`, E009 reaches only `3/10` SR and `0.296271`
  mean SPL, versus E008's `6/10` and `0.582788`. It loses E008 successes 1, 2
  and 8. Across all ten traces, actual-position free fraction falls
  `69.6 -> 41.2%`, commanded low-speed stall fraction rises
  `35.4 -> 64.1%`, free-origin first-metre plan collision rises
  `60.3 -> 69.9%`, and full-plan collision rises `81.0 -> 94.7%` on the same
  frozen benchmark proxy. This is stable wrong-route behavior, not a server,
  MPC, seed or episode mismatch.
- Rejected inference: exact affine decode proves only that `z_t` denotes a
  finite curve. Near the one-step deployment boundary it is still the fixed
  artificial Gaussian source, not an estimate on the clean navigation
  manifold. SanD/NavDP can repeatedly replace a noisy variable and select among
  candidates; that does not license a path-local C-space lookup on CurveNav's
  single-call source. E009's first lookup is mathematically defined but has the
  wrong navigation semantics.

## E010 — goal retrieval, clean-proposal geometry

- Root correction: restore the metric PointGoal reference as the first six
  blocks' spatial retrieval coordinate. It is explicitly an attention prior,
  not an executed trajectory, additive residual or hard goal corridor. The
  instantaneous field still receives the complete Flow state and predicts the
  clean endpoint. Only that learned clean proposal receives candidate-path
  C-space/BEV queries before the interval-average field produces the final
  B-spline.
- Scope: no extra block, head, candidate, critic, loss, projection, inference
  iteration or fallback. The four-frame encoder, target-independent observed
  BEV, improved-MeanFlow identity, deployment-boundary clearance objective and
  one-call runtime are unchanged. E009 code is removed rather than retained as
  a switch.
- Causal training comparison: keep E009's global batch `1792`, sample budget,
  FP16/FP32 precision contract and launcher for the first E010 run. This
  isolates the retrieval-location correction from the separate E008/E009
  optimizer-topology difference. Evaluate by source-truth validation and the
  same resident ten episodes before changing batch or learning rate.
- Completed run: four V100S GPUs reached `200` epochs and `4,600` optimizer
  updates at a final throughput of `2,820 samples/s`. The final loss was
  `0.06510`; the EMA checkpoint SHA-256 is
  `57a7bbabaeeb39f1632089915624acdde05f0d191aeb9dca70e7f0ccee0af3c9`.
  Source-truth offline ADE/FDE are `0.03758/0.10204 m`, full collision is
  `12.187%`, and first `0.5/1.0 m` collision is `0.000/0.495%`. Relative to
  E009, only the immediate execution prefix improves slightly; full collision
  and imitation error do not. The fixed resident ten-episode online run below
  remains the acceptance test.
- Fixed online rejection: on the same resident Home scene, episodes `0--9`,
  seed `1234` and `num-envs=1`, E010 reaches `3/10` SR and `0.298922` mean SPL.
  Successful episodes are `4,5,6`; E009 also reaches `3/10` and `0.296271`,
  while E008 reaches `6/10` and `0.582788`. E010 improves median final goal
  distance (`3.64 -> 2.54 m`), progress fraction (`0.412 -> 0.512`) and online
  plan peak-curvature p95 (`80.66 -> 36.63 m^-1`) relative to E009, but does
  not recover any additional success. The retrieval-order correction is
  therefore insufficient and must not replace E008 as the accepted baseline.

## E011 — target-independent global geometry, one proposal query

- Root diagnosis: E009 and E010 both use a non-proposal as the first
  path-local obstacle query. E009 uses the fixed Gaussian Flow source and
  reaches `3/10`; E010 uses the straight PointGoal ray and also reaches `3/10`.
  The former has no clean navigation semantics, while the latter lets target
  intent choose where geometry is read. Neither is repaired by more epochs.
- Paper/source basis: LoGoPlanner first extracts task-specific metric geometry
  and state context, then fuses goal intent into its diffusion policy. Its
  released runtime still uses ten denoising steps, sixteen samples and a critic;
  those mechanisms are explicitly excluded. CurveNav already has calibrated
  depth and known SE(2) history, so learned localization and point-cloud heads
  would duplicate more accurate inputs.
- Unique change: the first six blocks attend to the complete BEV/motion memory
  with metric bias relative to the robot origin. PointGoal remains in query
  content but cannot relocate memory coordinates, and no C-space path lookup
  occurs. The instantaneous field produces one clean proposal; only that
  proposal is decoded and queried against observed C-space before the final six
  blocks produce the MeanFlow average field.
- Complexity contract: one deterministic source, one decoder call, one clean
  proposal, one candidate C-space query and one final B-spline. No new block,
  head, candidate, critic, loss, projection, recurrent inference or fallback.
  Removing the E010 reference-ray decode/query slightly reduces work.
- Acceptance gate: preserve or improve E008's `6/10` fixed online result while
  reducing source-truth first-metre and forward-detour collision. E009/E010's
  `3/10` result rejects the version even when ADE, progress or curvature looks
  better. Training and measurements must remain bound to the E011 code commit.
- Pre-training verification: the complete CPU suite passes (`114 passed, 4
  skipped`; two visualization skips lack Matplotlib and two are CUDA-only).
  The production V100 runtime passes compiled FP16 perception, conditioning,
  MeanFlow primal/JVP, backward, optimizer step and deployment sampling in
  `178.74 s`. The graph remains `33,898,916` parameters with `29,018,980` in
  the decoder; E011 adds no parameter and removes one pre-proposal curve/C-space
  evaluation.
- Current training: the exact E011 run uses V100S GPUs 0 and 3, global batch
  `1792`, two `448`-sample microbatches per rank and FP16. At step `1820/4600`
  it sustains approximately `1393 samples/s` with both GPUs compute-saturated
  and about `27.9 GiB` allocated per device. This is approximately
  `696.5 samples/s/GPU`, slightly above the prior four-V100 E010/E011 topology's
  `692--705 samples/s/GPU`; the lower aggregate rate is entirely the two-card
  allocation, not duplicated model or data work.
- Throughput audit: packed depth stays resident on GPU, transfer is prefetched,
  condition K/V is projected once per microbatch, DDP uses `no_sync` for the
  first accumulation, AdamW/EMA are fused/foreach, and perception, conditioning,
  primal and JVP graphs are already compiled. On an idle RTX 4090, the eager
  FP32 MeanFlow JVP took about `150--155 ms` for the audit batch while the
  production compiled JVP took `23.8--25.3 ms`. BF16 JVP (`166--181 ms` eager),
  explicit attention and `max-autotune-no-cudagraphs` did not improve this path;
  the latter spent minutes enumerating infeasible Triton kernels. The metric
  depth projector costs only about `5 ms` in the same batch, so caching it would
  add a second data contract for a small upper bound. These variants are
  rejected: no code branch or custom kernel is retained. A material aggregate
  increase therefore requires more identical GPUs; it is not available from a
  mathematically equivalent local rewrite identified by this audit.
- Completed source-truth offline evaluation: ADE/FDE are `0.03713/0.09489 m`,
  full-path collision is `9.664%`, and first `0.5/1.0 m` collision is
  `0.000/0.346%`. Forward-direct collision is `1.516%`, but forward-detour,
  rear-goal and expert-moves-away strata remain `22.210/62.857/67.052%`.
  Approximately `69.97%` of colliding trajectories have no four-frame raw-depth
  evidence at their collision points. E011 therefore improves E010's
  `12.187%` full and `0.495%` first-metre collision, but it does not establish
  closed-loop safety.
- Vectorized online diagnostic: the fixed Home scene, official episodes `0--9`,
  seed `1234` and `num-envs=10` complete at `1/10` SR and `0.096125` mean SPL.
  Episode 6 is the sole success (`0.961246` SPL); the other nine fail. Scene
  startup plus execution takes about `16.2 min`, while the post-ready ten-episode
  workload takes about `8.1 min` (`1.23 episode/min`). There is no model, HTTP,
  Isaac or evaluator exception. This B10 result is a throughput diagnostic and
  must not be compared numerically with E008/E009/E010's B1 acceptance scores;
  nevertheless it rejects any claim that the improved offline prefix metric by
  itself predicts robust closed-loop navigation.
- Frozen-map replay closes the controller ambiguity. Across `11,043` E011 B10
  replans, `99.62%` of complete plans and `97.64%` of first-metre prefixes enter
  the benchmark robot-centre non-navigable raster; even among plans whose
  origin is free, `82.02%` of first-metre prefixes collide. Actual positions
  are free for only `19.65%` of sampled steps and the episode-mean stalled-step
  fraction is `73.77%`, while desired MPC speed remains `0.421 m/s`. Adjacent
  first-metre replans differ by only `3.59 mm`. E008 under the same frozen-map
  renderer had `69.56%` actual-position free fraction and `35.43%` stalled
  steps. E011 therefore generates stable unsafe intent; MPC is not the primary
  cause. The separately captured E011 B1 episode 0 also has `100%` full-plan
  collision and only `3.27%` actual-position free samples, so vector batching
  is not the source of the failure.
- Causal correction from the exact published bundles supersedes the earlier
  repository-commit audit. E008 and E010 have byte-identical `models/policy.py`;
  their decoder computation is also identical, with differences limited to
  comments, formatting and the contract string. The `6/10 -> 3/10` regression
  therefore cannot be attributed to goal-reference retrieval. E008 trained in
  BF16 with global batch `1024` for `8000` optimizer updates; E010 trained in
  FP16 with global batch `1792` for `4600` updates. The example budgets are
  close, but E010 made `42.5%` fewer parameter updates at the same learning
  rate. This optimizer-topology confound must be removed before another
  decoder mechanism is accepted.

## E012 — controlled E011 optimization audit

- Purpose: isolate E011's robot-origin global retrieval from the large-batch
  training change. The model, dataset, loss, deterministic source and one-call
  runtime are byte-for-byte E011; only the training contract returns to the
  evidence-backed E008 values.
- Contract: one RTX 4090, BF16/FP32 precision, global batch `1024`, three packed
  microbatches bounded by `342`, `40960` samples per epoch, `200` epochs and
  exactly `8000` successful optimizer updates. Learning rate remains `2e-4`;
  it is not rescaled because the original batch is restored.
- Active run: commit `fd76659` is running on the only free RTX 4090 (GPU2) in
  tmux `curvenav-e012`. After one-time graph compilation, steps `20--140`
  sustain `976--988 samples/s`; the device holds about `22.5 GiB` and reaches
  full compute utilization. The exact log is
  `/DataDisk2/hsb/curvenav-training-e011/train-e012.log`.
- Early paired evidence: archived E008 logs and E012 use the same RTX 4090,
  BF16, seed, data order, batch, schedule and update count. At steps
  `240/400/480/560`, E008 total loss is
  `0.6801/0.4811/0.4298/0.4061`, whereas E012 is
  `0.7534/0.5139/0.4876/0.4886`. The gap is principally MeanFlow imitation,
  not a missing clearance penalty. Repeating robot-origin geometry for all
  seven control tokens removes their distinct metric retrieval anchors and is
  the only remaining causal graph difference. This is preliminary convergence
  evidence; the run still completes before the architecture decision.
- The exact E012 checkpoints at steps `800/1600/2400` are retained. Their total
  losses are `0.41474/0.29187/0.17221`; the step-2400 split is MeanFlow
  `0.16962` plus visible-clearance `0.00259`. The active RTX 4090 run sustains
  about `986--1004 samples/s`. These snapshots will be evaluated with the same
  source-truth evaluator after the final checkpoint, so a lower training loss
  alone cannot accept robot-origin retrieval.
- Decision gate: compare the final EMA checkpoint with E011 using the same
  source-truth offline strata, then run the fixed ten episodes with
  `num-envs=1`. If E012 remains below E008, reject robot-origin retrieval and
  restore the goal-reference retrieval. If it recovers E008, the prior
  regression was optimization rather than architecture. No candidate set,
  safety projection, critic or extra loss is introduced during this audit.
- Completed source-truth comparison: steps `800/2400/8000` reach ADE
  `0.05554/0.03790/0.03124 m`, full-path collision
  `12.484/9.202/9.812%`, and first-metre collision
  `0.726/0.396/0.247%`. The final checkpoint restores E008's imitation
  accuracy (`0.03105 m`) while improving its full-path (`11.642%`),
  first-metre (`0.396%`), forward-detour (`26.572%`) and forward-direct
  (`2.152%`) collision rates to `9.812/0.247/23.456/1.174%`. It does not
  dominate E011: E011 remains slightly better on full-path and detour
  collision (`9.664/22.210%`), while E012 is better on first-metre and direct
  collision. Therefore the old `6/10 -> 1/10` evidence cannot be assigned to
  robot-origin retrieval without a matched online run; the training topology
  was a material confound.
- Checkpoint-selection lesson: step `2400` has the lowest full-path and detour
  collision, but the final checkpoint has substantially lower ADE and the
  lowest execution-prefix collision. Training loss or full-tail collision
  alone is not a valid checkpoint selector for receding-horizon navigation.
  At the final checkpoint, `68.24%` of colliding trajectories and all `15`
  first-metre collisions have no matching obstacle evidence in any of the four
  raw depth frames. More optimization of the same observation cannot directly
  correct those invisible cases.

## E013 — metric retrieval anchors with terminal-only goal intent

- Root correction: E012 repeats the robot-origin relative geometry for all
  seven control tokens. If memory token `k` is at `p_k`, its positional bias is
  `beta(p_k-0)` for every control index; token identity cannot restore the
  missing exact metric relation. E013 restores E008's distinct Greville
  reference anchors `R_i`, so the bias is `beta(p_k-R_i)` and each future
  segment retrieves the scene at its own physical location.
- PointGoal correction: E008/E010 encoded candidate control `C_i` relative to
  the corresponding straight-reference control `R_i`. That makes a safe
  lateral detour appear as an index-wise deviation from a straight template.
  E013 uses only the common terminal local goal `G=R_7`, encoding
  `[C_i/H,(G-C_i)/H,||G-C_i||/H]` for every control. The straight reference
  locates observation queries but supplies no intermediate output target.
- Scope: one decoder call, one instantaneous clean proposal, one proposal
  C-space query and one final B-spline remain unchanged. No parameter, block,
  candidate, critic, loss, projection, solver step, inference branch or data
  field is added. Relative to the accepted E008 graph, the sole learned
  semantics change is index-wise straight-template intent to terminal-only
  intent.
- Verification before CUDA: commit `41cb214` keeps exactly `33,898,916`
  parameters (`29,018,980` decoder), fourteen Flow coordinates and seven curve
  tokens. Focused config/checkpoint/policy/loader/precision regression is
  `65 passed, 1 skipped`; the skip is the production CUDA compile test reserved
  for GPU2 after E012. A direct counterfactual test changes all intermediate
  reference controls while fixing `G` and proves the terminal-goal feature is
  bitwise unchanged. Commit `0a5c0ca` records that contract.
- The complete CPU suite on the exact E013 release is `117 passed, 2 skipped`
  in `26.76 s` with CUDA hidden and eight pinned CPU cores. The two skips are
  exclusively the CUDA compile and CUDA prefetch tests; the former remains in
  the automatic GPU2 gate between E012 evaluation and E013 training.
- Paper/source basis: SanD explicitly separates generated B-splines from ESDF
  selection, while NavDP separates generation from critic selection. CurveNav
  deliberately has neither runtime selector, so trajectory-aligned observed
  C-space must enter its single generator. LoGoPlanner and DiffusionAnything
  independently support metric/trajectory-aligned geometry queries; none
  justify treating a straight goal ray as the desired intermediate path.
- Official-source audit: SanD's current condition encoder first self-attends
  depth/motion tokens and then injects one trajectory-endpoint token by cross
  attention. NavDP repeats its goal token in the generator, but explicitly
  zeros goal memory in the RGB-D critic that ranks safety. Both support a
  target-independent scene representation followed by goal intent; neither
  supports supervising intermediate controls against a straight goal ray.
- MeanFlow audit: the official iMF implementation returns average `u` and an
  auxiliary marginal velocity `v` from one network call, directly supervises
  both, and uses predicted `v` as the stopped JVP direction. E013 follows that
  contract; its only specialization is exposing `v` after the first half of
  the shared decoder so the resulting clean proposal locates the second
  half's geometry query. It does not use expert `e-x` as the JVP direction or
  perform a second inference call.
- Mixed-precision audit: Accelerate skips the underlying optimizer call when
  FP16 GradScaler detects overflow. The old loop correctly held LR and EMA but
  still advanced the reported step. The unique loop now recomputes the same
  sample batch and Flow source after the scaler is reduced, and counts the
  step only when parameters actually update. BF16 execution is unchanged.
- Acceptance: first finish and evaluate E012. E013 then trains from zero with
  the same BF16, global-1024, 8000-update contract. It must preserve E008's
  fixed `6/10` result and reduce source-truth forward-detour collision; average
  ADE alone cannot accept the model.
- Production training: the real RTX 4090 compiled forward, MeanFlow JVP,
  backward and deployment-sampling gate passes. The remote virtual environment
  had lost its declared `accelerate` dependency after E012; restoring the
  pinned `1.14.0` package repaired the sole import failure without changing
  code or adding a launcher path. E013 now runs on GPU2 with BF16, global batch
  `1024`, about `22.5 GiB` allocated and `975--996 samples/s` after compilation.
  At the matched step `800`, E013 total/MeanFlow/visible-clearance losses are
  `0.33864/0.32538/0.01326`, versus E012's
  `0.41474/0.39513/0.01961`. This is evidence of improved optimization only;
  the retained checkpoint still requires the same source-truth and online
  acceptance tests.

## E014 — target-independent metric horizon retrieval

- Root diagnosis: E013 restored distinct retrieval anchors, but placed them on
  the PointGoal ray. That lets the target choose where the first obstacle
  lookup occurs; E012's repeated robot-origin query has the opposite defect and
  cannot distinguish future metric locations. Neither is a clean safety
  contract.
- Unique architectural change: replace the first-stage query anchors with the
  fixed forward metric slots `A_i=(xi_i H,0)`, where `xi_i` are the seven
  non-origin cubic-B-spline Greville abscissae and `H=3.6 m`. These slots are
  scene-retrieval coordinates only. The clipped PointGoal terminal `G` remains
  a separate common intent `[A_i/H,(G-A_i)/H,||G-A_i||/H]`; it cannot move the
  safety query or impose a straight intermediate route.
- The rest of the contract is unchanged: four calibrated frames, observed
  metric C-space/BEV, one deterministic fixed-source improved-MeanFlow call,
  one clean proposal query, one final B-spline and the existing deployment-path
  clearance objective. No candidate set, critic, extra loss, hard projection,
  inference loop or data branch is introduced.
- Mathematical gate: changing PointGoal while holding depth and state fixed
  must leave `metric_reference`, the first-stage reference path and its scene
  relative geometry bitwise unchanged, while changing only terminal intent and
  the final trajectory. Each anchor must be distinct and decode to the identity
  forward segment from the robot origin to `H`; the anchors are never output
  initialization.
- Paper basis: SanD and NavDP separate target-conditioned generation from
  target-independent visual geometry before their ESDF/critic selection;
  LoGoPlanner uses metric task geometry rather than a straight output template.
  E014 adopts only that factorization while preserving CurveNav's single-step
  constraint. The fixed slots provide each future control a physical scene
  location without allowing PointGoal to bias the safety representation.
- Acceptance: run the focused mathematical/gradient suite, then train from
  zero with the E012 contract on the fastest available homogeneous GPUs. Do not
  select a checkpoint by ADE alone. Require source-truth full/first-metre and
  forward-detour strata plus the fixed ten-episode online trace with per-plan
  frozen-C-space first-hit labels. If E014 does not improve the executable
  prefix and visible first-hit rate, reject the anchor hypothesis instead of
  adding another penalty or inference stage.

## E012 — resident online B10 (2026-09-03)

- Checkpoint: `outputs/train_policy-e012/checkpoint.pt`; fixed Home scene
  `MVUCSQAKTKJ5EAABAAAAABA8_usd`, seed `1234`, `num-envs=10`, GPU2.
- Result: `0/10` success and mean SPL `0.000000`; all ten episodes timed out.
  The run is a throughput/architecture diagnostic, not a fixed-protocol score.
- Runtime: one resident scene startup took `941.703 s`; after that, the ten
  episodes completed without a model, HTTP, Isaac, or Acados crash. The
  trajectory trace contains 829 plans per episode, mean policy latency
  `200.34 ms`, mean MPC latency `262.27 ms`, and mean total plan latency
  `462.61 ms`.
- Interpretation: E012's offline source-truth collision reduction does not
  transfer to closed-loop execution on this scene. The trace must be read with
  the per-plan frozen-C-space hit/OOB labels before attributing the failure to
  the model or controller; no checkpoint is accepted from this run.
- Artifacts: `/DataDisk2/hsb/eval-server-audit/runs/curvenav-e012-online10-20260903/pointgoal-v2/resident/20260902_202230/`.
