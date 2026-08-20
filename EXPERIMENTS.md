# CurveNav 实验记录

最后更新：2026-08-20

本文件只记录实验假设、运行事实、失败原因和 keep/discard 决策。当前模型结构与数学协议只看
`ARCHITECTURE.md`。运行代码、checkpoint schema 和 wire protocol 不承载历史兼容逻辑。

## 目标与判定

当前里程碑：在完全相同的 100 个 PointGoal episodes、相机、控制器、seed、成功阈值和超时下，CurveNav
的 SR 与 SPL 达到 NavDP。

每轮只允许一个主要变量。保留条件：

1. 数据/坐标/相机审计通过；
2. 单元测试和固定批过拟合通过；
3. 相同训练预算下 held-out 指标没有数值退化；
4. matched quick-100 的 SR/SPL 改善，且碰撞或严重 stuck 不显著增加；
5. 没有通过修改阈值、episode 集合或接口语义获得虚假提升。

错误协议产生的数字只用于定位根因，不进入当前 baseline，也不要求运行代码兼容或标记它们。

## 当前范围

当前任务只维护本地 CurveNav 的数据合同、模型架构、训练实现、离线门禁与实验记录。不读取、调度或修改
其他 Codex 任务和测评服务器；外部闭环结果只在用户明确提供后作为新的实验输入。

## 实验表

| ID | 单一变化 | 结果 | 决策 |
| --- | --- | --- | --- |
| E000 | v1 SanD 数据训练 | 旧首轮 SR 43%、SPL 0.3981 | discard 架构语义，不复用权重 |
| E001 | v1 HSSD1000 / mixed | SR 5% / 7%；旧 HSSD FOV 约 89°，目标语义与部署不一致 | discard 数据配方，不再运行旧配置 |
| E002 | 修复 v1 部署历史/候选接口 | 10 条 smoke 为 8/10；样本过少 | 只证明接口可运行，不当模型成绩 |
| E003 | 检查 benchmark 深度传输 | 7 m、8 m 均被压成 6.5535 m，NaN 变 0 | 删除 uint16 路径；所有旧评测数字作废 |
| E004 | v2-A：最终 task goal + 自由局部终点 | 61 项测试通过；12,188,546 参数；v1 checkpoint 被 contract 拒绝 | 进入过拟合门禁 |
| E005 | v2-A 过拟合尝试 1 | 训练到 1000 step，loss `0.26894 -> 2.71e-5`；末尾诊断采样缺 autocast，未保存 | 修复统一推理 autocast 后重跑；失败日志保留 |
| E006 | v2-A 过拟合尝试 2 | fixed loss `0.26894 -> 2.91e-5`，ratio `1.08e-4`；fit RMSE `0.000736m` | keep；门禁通过 |
| E007 | v2-A SanD-only 完整训练尝试 1 | GPU 0/1 在首次 DDP collective 自旋；7 分钟无日志、各 542 MiB、低功耗 100% SM | 中止并保留日志；不是模型或数据失败 |
| E008 | v2-A SanD-only 完整训练尝试 2 | 15,000 steps；loss 0.1429→0.00419；约 7.2k–8.5k samples/s | 完成，待 quick-100 |
| E009 | HSSD v2 pilot 200 | 200/200 有效，12,210 samples，0 重复/泄漏，camera/depth/schema 审计通过 | keep 为下一轮候选数据，不混入 E008 |
| E010 | v2-A step 3,000 held-out | pairwise ADE 0.451m；oracle ADE 0.222m；oracle curvature p95 8.78m⁻¹ | 多样性恢复，但尚未收敛 |
| E011 | free-endpoint selector 语义修复 | step 7,500：selector arc 2.85m、curvature p95 4.06m⁻¹；旧选择为 1.36m/11.63m⁻¹ | keep；删除“最短路径优先” |
| E012 | v2-A step 15,000 held-out | oracle ADE 0.204m；pairwise 0.439m；selector curvature p95 3.70m⁻¹；batch-1 p50 68.0ms | 进入 format-10 smoke 与 quick-100 |
| E013 | HSSD v2 pilot 动力学审计 | episode 最大曲率 p50/p95/max=0.931/3.618/5.919m⁻¹；最紧半径 0.169m 可在约 0.084m/s 跟踪 | 不增加曲率硬门禁；闭环监控降速、饱和和跟踪误差 |
| E014 | 9998 v2-A 部署合约 | SHA 匹配；format-10、step 15,000、EMA strict load、final-goal/free-endpoint 合约通过 | 原子替换活动 format-9 链；等待同协议 smoke/quick-100 |
| E015 | pilot 200 同状态多拓扑审计 | 180/200 有多条唯一几何，120/200 Hausdorff≥0.5m；但当前 A* 无 corridor/homotopy ID | 不能把几何分离冒充左右绕；仅在 selector gap 成立后实现 scene-graph sidecar |
| E016 | SanD 训练/部署历史与尺度复核 | 23,025 个原始步的中位间距 0.14982m；runtime 为 0.15m；held-out 仅 6.05% 横向超 2.5m、2.34% 终点前向超 5.6m | 历史间距一致；scale-only 不截断，当前 checkpoint 有效 |
| E017 | benchmark→server→runtime 坐标链复核 | 仿真先算 `R_world→body(goal−robot)`；HTTP 原样传完整 xy；xyzw yaw 与训练 `R(-yaw)` 同号 | 无 world/body、符号或局部截断错误；继续等待闭环 |
| E018 | HSSD v2 正式加载链 | 删除逐样本 raw depth 路径；200 episode/12,410 帧 cache 用时 15.47s、全量 0.870GiB（train 常驻 bank 0.688GiB）；batch256/8 workers/CUDA 预取稳态约 33.2k samples/s；65 tests passed | keep；唯一入口为 SHA 校验的 packed bank，解除 policy-candidate sidecar 阻塞 |
| E019 | HSSD v2 policy hard-negative sidecar | 12,210 states、97,680 policy candidates、229,688 Pareto pairs、95.32% state coverage、0 失败；pairwise ADE p50 0.460m，nearest-expert ADE p50 0.230m；68 tests passed | keep 为 selector/critic 证据；topology=null、closed-loop=not_run，不提前改当前 selector |
| E020 | 当前 DepthSafetySelector 离线 gap | 原始 360×640 depth 重放：28.59% 选择被另一 policy 严格支配；碰撞/余量机会损失占全部状态 16.67%/17.00%；跨 scene gap 22.72%–31.62%；70 tests passed | learned critic 有明确离线空间；仍等 matched quick-100 oracle 决定是否作为下一唯一变量 |
| E021 | quick-100 三服务器迁移 | 9998 的 8×2080 被外部作业占用；3090/4090 共 8 张 24GiB GPU 空闲，但两个 `hsb` 账户都暂无线性 `renderD*` 访问权；9998 最小活动 payload 约 2.55GiB，源码/资产/三权重 SHA 已冻结 | 不让完整环境迁移阻塞闭环：并行验证目标机 Isaac 权限与“9998 仿真 + 3090/4090 统一远程推理”；首条 1-episode smoke 成功即进入 fixed quick-100 |
| E022 | 9998 evaluator/server 拆分 smoke | GPU4 server + GPU6 evaluator 的三模型均无 OOM。`hard_0/0`：SanD stuck、SR/SPL=0、mean/P95=152.73/170.32ms；NavDP 成功、SPL=0.9383、碰撞0、290.63/311.47ms；v2-A 成功、SPL=0.8338、碰撞0、110.32/122.24ms，活动 format-10/EMA 生效 | keep 资源调度、协议和部署证据，不把单 episode 当正式成绩；9998 保留为已验证后备，正式 quick-100 优先迁移到空闲 4×3090 |
| E023 | 3090 Isaac 硬验证 | 用现有 `isaaclab12.sh -p` 启动：约 78.6s `app ready`，本地 USD stage、PhysicsScene、cube 和一帧 update 成功，80.9s 正常关闭并释放 GPU；首次运行生成约149MB shader cache | keep；3090 具备 headless RTX/Isaac 仿真能力，进入最小 benchmark/权重同步与真实 PointGoal smoke |
| E024 | v2-A `hard_0` 10-episode 诊断 | episode 0–9、seed1234：SR=40%（4/10）、SPL=0.3546、碰撞率=20%、stuck=6/10；端到端9m46s，evaluator9m07s | 链路可用但不足以归因；10条与假设SR≈0.6只差2次成功，先看逐episode候选/selector证据，不把小样本波动误判为数据或架构结论 |
| E025 | 独立 stage-2 critic 数据/梯度门禁 | 修复 validation 的全局 `sample_index` 外键；12,210 states、229,688 pairs 全量加载；固定8状态 loss `0.971575→0.000353`，ratio `3.64e-4`；相关测试10项通过 | keep；冻结 policy 的 critic 链可学习，进入 scene-heldout 两卡训练 |
| E026 | Pareto score 直接 argmax | 40 epoch/1,520 steps，最佳 policy-pair accuracy 93.36%、strict-dominated 6.19%；但 collision/margin 机会损失 26.58%/24.62%，劣于同 validation 的 depth selector 18.89%/18.07% | discard 直接 argmax 合同；Pareto score 不能独自表达安全优先，不部署该 checkpoint 语义 |
| E027 | 预测安全→Pareto score 严格字典序 | 最佳 epoch29：policy-pair accuracy 93.36%；同一2,551-state validation 上 strict-dominated 29.40%→13.72%，collision opportunity 18.89%→17.60%，margin opportunity 18.07%→17.01%；稳态约5k–9k samples/s、每卡约1.7GiB；全量76 tests passed | keep 为 train-only stage-2 原型；三项均优于现有 selector，但安全提升仅1.29/1.06个百分点，闭环前不替换部署链 |
| E028 | v3：采样未来 PointGoal 与 target 终点对齐 | 旧 v2-A 监督审计：goal 距离 p50=17.277m、target 弧长均值=3.297m、方向差 p50=52.806°、66.27% 超过15°；v3 恢复 SanD 官方同一 `end_idx` 生成 `end_relative_pose`/监督轨迹的核心合同，保持可变窗口和自由预测终点；format-11；76 tests passed；固定批 loss `0.009608→3.84e-6`、ratio `3.99e-4`、fit RMSE `0.000736m`；两卡15,000-step最终 loss `1.5268e-4`，旧同预算为 `4.1867e-3` | keep；数据、梯度、完整训练门禁通过，进入同标签 held-out 对比 |
| E029 | 旧 format-10 在 v3 held-out 标签上的冻结基线 | 1,024 固定样本/8候选/EMA：single ADE 0.4523m、oracle ADE 0.1670m、selector ADE 0.2202m、pairwise ADE 0.2492m、selector 曲率p95 6.3047m⁻¹；batch-1 p50 73.89ms | keep 为唯一公平基线；只显式离线加载一次，不给生产 format-11 loader 增加兼容分支 |
| E030 | format-11 v3 对 format-10 的同标签 held-out 对比 | 相同1,024样本/seed/8候选/EMA：single ADE 0.0403m（-91.1%）、oracle ADE 0.0303m（-81.9%）、selector ADE 0.0407m（-81.5%）、selector弧长误差0.0308m（-92.6%）、曲率p95 2.5071m⁻¹（旧6.3047，expert2.2329）；但pairwise ADE 仅0.0178m（旧0.2492） | offline keep v3；监督错配是当前主要根因。候选已准且平滑，但多样性塌缩，matched闭环前不加critic；若闭环仍因候选覆盖失败，下一变量是同状态真实多拓扑/recovery数据，不是调selector |

## 已确认的根因

### R1：条件目标必须与当前监督任务构成可学习的一一对应

v1 的部署曾把最终任务目标截到 5.6 m，而训练把专家局部前缀末点当 PointGoal，两端输入不是同一个变量。
v2-A 虽删除了截断，却把整条 run 的最终目标和随机局部前缀配对。20,000 个窗口审计显示，这使 goal 距离
中位数达到 17.277 m、goal—target 方向差中位数达到 52.806°，66.27% 超过 15°；在有限局部深度中，模型
经常无法知道该随机前缀对应最终目标的哪一种绕行，监督本身是冲突的。

处理：v3 将每个随机窗口定义为一个独立 PointGoal 任务：同一个 sampled future point 同时作为条件 goal 和
专家 target 终点。窗口长度继续随机，预测终点继续自由，不恢复固定 2.1 m 或终点投影。部署保留完整
PointGoal，并通过闭环重规划消化训练窗口之外的长距离；其距离外推风险必须由 matched 闭环单独检验。

### R2：所有候选被迫共享可能不可达的终点

v1 的 Flow 源、训练插值、速度 mask 和采样投影都固定末点。8 候选平均两两 ADE 只有厘米级，selector 没有
真正的左绕/右绕选择。

处理：删除末点投影，Flow 只固定机器人原点；第 1–11 个控制点全部参与随机源、监督和 Euler 积分。

### R3：旧自建数据与评测相机不一致

旧 HSSD 深度约 89° FOV，实际 benchmark 为 640×360、水平 FOV 约 67.757°、相机高度 0.30 m。resize
不能恢复不同视锥下缺失或新增的观测。

处理：路径几何可复用，所有深度按最终相机重新渲染并保存无损 float32 米制物理量。

### R4：旧 benchmark 深度编码有损

旧客户端把米制深度乘 10000 后 clip 到 uint16。回环结果为：6.0→6.0、6.55→6.55、7.0→6.5535、
8.0→6.5535、NaN→0。

处理：当前 benchmark 只允许 float32 米制 raw depth 与 NaN invalid；删除旧编码及所有 fallback，不增加
协议版本分支。

### R5：overfit 末尾诊断没有走训练同一 AMP 边界

packed depth 在 GPU 上是 FP16，训练前向位于 autocast 中；末尾 `policy.sample()` 曾在 autocast 外运行，
导致 FP16 输入与 FP32 convolution 权重不匹配。

处理：诊断采样与正式评测/部署使用相同的推理 autocast 上下文。没有在 encoder 内增加隐式 dtype 转换。

### R6：当前宿主的 NCCL P2P collective 自旋

GPU 0/1 的最小两 rank `all_reduce` 在 NCCL `P2P/CUMEM` 和 `P2P/IPC` 路径都无法完成；关闭 P2P 后，
相同测试通过 `SHM/direct/direct` 在 6.5 秒内完成，结果为两个 rank 都得到 3.0。训练启动脚本因此固定
`NCCL_P2P_DISABLE=1`。这是一条经最小复现确定的当前宿主通信路径，不保留失败路径开关。

### R7：固定终点时代的 selector 与自由终点不相容

旧 selector 在安全候选中最小化绝对路径长度；终点放开后，它在 step 6,000 held-out 系统性选到平均仅
1.36 m 的短轨迹，而 target 为 2.44 m，且曲率 p95 达 11.63 m⁻¹。处理：安全优先不变；安全集合内首先
最大化终点对最终 task goal 的实际欧氏距离缩减，只在同进展时比较长度和 bend。候选分值改为严格排序值，
离线评测直接复用同一个 `select_indices`，不再从近似 cost 反推选择结果。

### R8：曲率阈值不是当前控制器的物理可执行性边界

quick-100 当前控制合同为 `0<=v<=0.5m/s`、`|omega|<=0.5rad/s`，允许左右轮反向和 `v=0` 原地旋转；
源码没有正的最小稳定速度、加速度、jerk、slip 或闭环跟踪误差约束。因而 `kappa=1m^-1` 只表示
`v=0.5m/s` 时达到角速度上限，不是最小转弯半径。pilot 200 中 95 条 episode 需要在局部降到 0.5m/s
以下，但没有 episode 超过 `kappa=10m^-1`。处理：不删除高曲率专家；保持现有曲率质量审计，并在统一
闭环中记录 MPC 降速、控制饱和与跟踪误差，之后才有证据定义动力学门禁。

### R9：sidecar 首次存储实现未对齐冻结 schema，压缩 NPZ 校验被重复解压放大

首个 32-state 几何 smoke 本身通过，但候选 producer revision、uint64 seed、JSONL state index、空 corridor、
nullable closed-loop 和 nominal angular-rate 字段不完整，因此在正式全量提交前停止，旧 smoke 不进入训练。
修正后加入唯一 reader/validator、确定性 candidate ID、三组 ragged offset 和源数据/packed-depth 双 SHA 绑定。
随后发现 validator 在 candidate 循环中反复访问压缩 NPZ 会重复解压；改为每 shard 一次解压到内存后，同样的
全量严格校验从数分钟未完成降到 4.18 秒。修复后的 smoke 与全量前 32 条逐数组一致，不保留旧兼容 reader。

### R10：统一 8.5GiB 门槛造成 quick-100 队头阻塞

9998 的唯一队列先等待 GPU 0–3 上的 NavDP；即使 GPU 4–6 各有约 6.6GiB 空闲，后续 SanD/v2-A 也无法
独立进入 smoke。该门槛没有按 evaluator 与三个模型的实测峰值拆分，连续等待没有产生一条 episode，因此
不能继续作为调度依据。处理：冻结协议和 episode 身份，但把资源判定改为逐组件 smoke 的实测峰值加余量；
优先利用空闲 3090/4090 作为统一推理端。显存不足的方案只运行一次并记录，不重复轮询同一失败条件。

### R11：自建 Isaac smoke 引入远端默认资产，误把测试夹具等待当作仿真失败

3090 的 `SimulationApp` 已实际创建，但首个诊断脚本随后调用 `World().add_default_ground_plane()`，进程持续向
外部 443 建连并停在 `construct_world`，这不是 Vulkan/render 权限失败。直接调用 Isaac `python.sh` 又缺少
IsaacLab extension path，同样不是正式入口。处理：删除这两个诊断变量，只保留该机现有
`isaaclab12.sh -p` + `AppLauncher` + 本地 USD 路径；一次有效启动完成后立即进入真实 benchmark，不继续优化
自建 smoke。生产链不增加网络资产 fallback。

### R12：CUDA 前瞻预取与 `gather_for_metrics` 的尾批语义冲突

critic 首次两卡启动在 condition cache 完整性门禁被拦截：最后 256 个 train sample ID 未写入。根因不是数据
缺失，而是前瞻预取器提前耗尽底层 DataLoader，使 Accelerate 的尾批去重在错误的 batch 上裁剪。缓存表按
全局 sample ID 幂等写入，本来不需要尾批去重。处理：condition cache 固定使用普通 distributed gather，允许
padding ID 重复覆盖；直接 validation loader 才使用 `gather_for_metrics`。修复后 9,659/2,551 条 cache 均完整，
训练缓存约 2.17s/0.56s；不保留失败路径开关。

## 当前本地状态

- v3 SanD goal-aligned 数据合同、format-11 checkpoint、76项测试、固定批过拟合、15,000-step训练和 held-out
  均已通过；checkpoint 为 `outputs/train_v3_sand_goal_aligned/checkpoint.pt`。E030 证明监督错配是主要根因，
  同时暴露 8 候选多样性不足。
- v2-A SanD-only 已训练完成；checkpoint：`outputs/train_v2a_sand_official/checkpoint.pt`，SHA256
  `0ed61f134d8161c9bc6edba9e1ef712337a9438c494d8414617e8283a57b7412`。
- HSSD v2 pilot 已完成最终审计，输出 `outputs/hssd_curvenav_v2_pilot_200`；正式 policy hard-negative sidecar
  已覆盖全部 12,210 states，输出 `outputs/hssd_curvenav_v2_pilot_200_policy_hardneg_v1`，约 71.08MiB，不复制
  depth，源 bundle SHA 保持不变。完整报告位于 `outputs/audits/hssd_v2_policy_sidecar_v1_20260820`；当前 selector
  的离线 gap 报告位于 `outputs/audits/hssd_v2_selector_gap_20260820`。设计审计位于
  `outputs/audits/hssd_v2_multitopology_design_20260819`；现有 planner 不具备可信 topology 标签，因此 E019
  只含真实 policy/expert/hold 候选，尚不能提供左/右等 topology 专家监督。
- standalone critic 固定批过拟合已通过；两卡 scene-heldout 链只训练 critic，policy/condition encoder 冻结。
  直接 `argmax(Pareto score)` 已由 E026 判定失败；E027 的唯一候选合同为“预测 collision-safe、预测
  margin-safe、Pareto score”的严格字典序，不使用加权 scalar utility。最佳 checkpoint 为
  `outputs/train_critic_hssd_v2/checkpoint.pt`，SHA256
  `57b5bcd47b3827d1c28ed54142c1a74cba6bfb0a05068afc78598a6ce8196660`。

## 下一决策

1. E030 已显著通过离线门禁，下一步只做 matched quick-100；闭环主要看相同 episode 的 SR/SPL、collision、
   stuck 与 oracle gap。达到 SanD 前不改视觉骨干、不训练新 critic，也不混入 HSSD。
2. 若 v3 闭环仍失败，必须先看候选级 trace：当前 pairwise ADE 只有0.0178m，若 oracle 同样失败，说明单一路径
   公开数据没有教会真实多拓扑/recovery覆盖；先补数据。只有 oracle 明显成功而 selector 失败，才重建 stage-2。
3. 后续数据主线必须同时满足：最终相机/坐标/控制合同，同状态多条真实可行拓扑，policy hard negatives，按 scene
   隔离 held-out，以及闭环失败状态回灌；不能只增加同一 `dataset_avoid` 窗口的重复次数。
4. 模型主线仍按两阶段门禁：阶段一先学习条件 Flow 的多候选 B-spline 分布；只有 stage-1 达标后，阶段二才用
   collision/margin/progress/clearance 的候选排序，再决定是否联合微调。critic 是完整架构的一部分，不是部署时
   可选启发式开关。
5. 根据 matched episode 与 oracle selector 结果决定：
   - 明显提升则保留自由终点语义；
   - oracle selector 高、当前 selector 低，则下一轮训练 trajectory critic；
   - oracle 也低，则先生成同状态多拓扑专家，不靠扩大采样数掩盖覆盖不足；
   - 时序窄障碍失败占主导时，才进入逐帧 shared CNN + temporal Transformer + SE(2) motion 的 v2-B。

learned critic 的 E019 sidecar 是唯一训练输入：train/validation scene 已隔离，离线不可判定的 15.30%
policy 终点不产生伪 geodesic 标签；训练只消费明确 Pareto pair。critic 必须先在固定 held-out pair ranking、
校准和吞吐门禁上优于当前 selector，再替换部署 selector，不能与生成器、视觉时序或数据扩容同时改。

当前单变量 critic 结构是：冻结 v2-A policy/condition encoder；候选 12 个归一化控制点进入 2 层
trajectory Transformer 并 cross-attend 同一 condition memory，输出一个 pairwise score，同时用独立辅助头预测
collision、margin violation、geodesic progress 和 clearance。candidate kind、producer slot 和 sample ID 不作为
输入，避免 expert/policy 身份捷径。对状态 `b` 的明确 Pareto pairs `P_b`，主损失固定为按状态等权的

```text
L_pair = mean_b [ mean_(w,l in P_b) softplus(-(s_w-s_l)) ]
```

辅助头只在对应标签有效时计算 BCE/Huber，用于几何表征和校准，不在数据中制造加权 scalar utility。第一轮
只训练 critic，不反传 policy encoder。E026 已证明 Pareto score 不能直接承担安全优先；唯一选择规则因此固定为
先预测 collision-safe，再预测 margin-safe，最后在同安全等级内最大化 score 的严格字典序。validation 只看
policy-policy pair accuracy、所选路径 strict-dominated fraction、碰撞/余量机会损失和额外延迟。只有这些门禁与
后续闭环证据同时成立，才替换当前 selector。

若进入 trajectory critic，数据链固定为：按 scene 缓存 clearance-constrained medial-axis corridor graph，
从同一 anchor/task goal 枚举 loopless K-shortest edge sequences，把 ordered edge sequence 作为 topology ID，
再保存到不复制 depth 的 `curvenav_critic_sidecar_v1`。当前逐状态重复五次规划外推约 31.2 小时，禁止作为
正式生成器；1000 episode 重渲染本身约 13.1 分钟/53.36GiB，sidecar 上界约 149MiB。先在 pilot 上用真实
模型候选作 hard negatives 验证 critic，再决定是否扩到 1000，不用旋转、加噪或扭曲单一路径伪造多模态。

## 2026 方法预注册

这些工作只用于预先约束下一轮选择，不在 E028 运行中改结构：

- [ForesightFlow](https://arxiv.org/abs/2606.04968)：把轨迹与成功 potential 放入同一 Flow，并解耦轨迹加权和
  potential 校准；仅当 oracle-selector gap 高时，作为 learned ranking 的首选依据。
- [FLUX](https://github.com/Zeying-Gong/FLUX)：Rectified Flow 导航与 static-to-dynamic curriculum；仅当动态/时序
  matched failure 占主导时，作为两阶段训练依据。
- [DreamFlow](https://arxiv.org/abs/2603.02976)：预测观测范围外的扩展环境 latent；仅当长墙、遮挡和 local-minimum
  失败占主导时考虑，不能在当前纯局部数据上凭空增加模块。
- [SafeFlowMatcher](https://github.com/takahashi-seiryu/SafeFlowMatcher)：在最终执行路径上做 prediction-correction
  安全修正；若生成覆盖已足够但碰撞仍高，再评估是否优于当前 depth selector。

NavDP 源码对照确认其平滑轨迹不是来自固定 2.1 m：它预测 24 个局部速度增量，10-step diffusion 一次生成
32 条，再用共享 Transformer critic 按障碍碰撞与 clearance/progress 标签排序并返回 top-8。critic 同时看真实
专家轨迹和随机旋转后经 cubic spline 插值得到的负样本。CurveNav 不复制其三维 action 表示，但若 E028 的
oracle gap 高，下一轮的训练数据必须提供同状态正负候选及安全/进展标签，不能只新增一个无监督评分头。
