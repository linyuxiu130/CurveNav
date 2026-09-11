# PointGoal 测评加速方案

本文记录当前唯一测评链路的吞吐优化。优化只作用于 launcher、进程绑定、
模型 adapter 和不参与指标的可视化输出；官方场景、起终点数组、前视相机、
控制器、成功阈值、超时、SPL 公式和原始 float32/NPZ 协议保持不变。

## 已落地

闭环模型调试统一使用 evaluator 的紧凑轨迹输出：每个完成回合在场景
`metric.csv` 旁写入 `trajectory_traces/episode-NNN.npz`。它只记录数值型的仿真步
状态/动作和 policy 更新时的局部轨迹/MPC 解，不恢复已删除的图片或视频写入，不改变
raw float32/NPZ 服务协议，也不新增 launcher 或模型分支。文件按回合完成时落盘，后续
场景异常不会丢掉已经完成的 trace。当前已加入聚焦的 writer/reader contract 与编译
检查。局部轨迹是可变长数值序列，统一写为点维填充数组和逐计划真实长度；
不再错误假定各次 policy 输出的点数相同。状态/动作以固定 `0.5 s` cadence 采样；
每一次 policy/MPC replan 仍完整记录。
这避免了为诊断在每个 `0.1 s` 仿真步执行多次 GPU→CPU 同步，而仍保留判别停车、
振荡、跟踪偏差和规划漂移所需的时间分辨率。SPL 路径长度也始终在 GPU 累积，只在
episode 终止时取标量；指标定义完全不变。下一次单回合 smoke 必须记录实际写入体积
和 wall overhead，之后才能进入正式评测。

CurveNav 的 `/navigator_reset` 现在把 evaluator 请求中的实际 `3x3` 相机内参传给
唯一 runtime，而不是只传 batch size。该接口与训练相机重采样合同一致，不新增标定
fallback 或第二个 server。唯一来源清理实现为 `0b913ac`，4090 生产 checkout 跟随
本分支当前 HEAD；本地正式 profile 为 B16，顶层完整 `pytest` 为 `28 passed`，
CurveNav B16 单场景 dry-run 也确认
`PYTHONPATH` 只包含显式 model-config bundle 的 `source_root/src` 和 benchmark shim。

2026-08-27 的 hot-loop 修改已在锁定 Isaac Python 上完成 `compileall`、CLI help/dry-run
与 12 项 contract tests。真实 wall-time A/B 尚未记录：正在执行的单回合诊断在修改前
已加载旧 evaluator，不能把它误报为新链性能。下一次新 evaluator 的同场景 smoke 将只
比较稳定执行阶段，并单列一次性 scene-ready 时间。

1. **物理 GPU 绑定。** 每个 worker 的 policy server 仍使用
   `CUDA_VISIBLE_DEVICES=<physical-id>`，保证服务端只占用分配到的卡。Isaac
   Sim 5 evaluator 不再设置 `CUDA_VISIBLE_DEVICES`；launcher 同时传入
   `LOCAL_RANK=<physical-id>` 和 evaluator 的 `--device cuda:<physical-id>`。
   这样 Kit 自己枚举物理卡，避免容器/可见卡重映射导致所有 worker 落到 GPU0。
2. **受控向量化。** `--num-envs N` 将 N 个 Isaac 环境写入评测配置。episode
   分配器只发放固定数组中的目标 ID；单个环境 reset 后原子领取下一个 ID，inactive
   slot 被屏蔽，跨 reset 的旧 planner action 由 generation 拒绝。CSV 可以按完成时间
   乱序写出，但 launcher 最终排序并校验 ID 集合无重复、无缺口、无越界。通用默认
   仍为 1；主机可用 `NAVBENCH_NUM_ENVS` 选择已经验收的吞吐配置。两种模式使用相同
   场景、题目、模型、控制器、终止和指标，但 simulator 调度不同，结果必须记录
   `num_envs`，同一比较中不得混用。
3. **多卡并发。** `--gpus 0,1,2,3` 为场景 worker 分配不同物理 GPU；每个 worker
   拥有独立端口、OptiX cache (`~/.cache/navbench/optix/gpu_<id>`)、server/eval
   日志和结果目录。`--gpu-poll-seconds` 在显存不足时等待，不抢占他人进程。
4. **启动稳定性。** `--launch-stagger` 错开 Kit 启动；tmux 只负责持久化 shell，
   launcher 负责进程组清理。每次 run.json 记录 GPU、num-envs、seed、checkpoint
   和 suite/runtime SHA，便于 resume 前做身份核对。
5. **CurveNav 单一来源。** adapter 从显式 `--model-config` 的 bundle 根推导
   `source_root/src`；benchmark 只保留协议 shim
   `baselines/curvenav/curvenav_server.py`，旧的复制模型、训练、数据生成、配置和测试
   已全部删除。server 传递 `trajectory.path` 并提供统一 `/shutdown` 生命周期端点，
   不再存在可被误加载的第二套 CurveNav 实现。
6. **统一 server 生命周期。** 六个 adapter 都由同一个 `ServerSpec` 注册表启动，
   evaluator 结束时访问的 `/shutdown` 在每个内置 server 上都存在。launcher 仍在
   scene 边界回收 server：各模型的 reset 状态语义并不完全相同，跨 scene 复用
   server 会改变隐藏状态，不能作为等价加速。
7. **指标专用渲染链。** 固定 X-NavDP runtime 不再创建仅用于视频的 bird-eye
   camera，也不再逐 step 把前视/鸟瞰图同步回 CPU、缩放并编码 MP4。policy 仍读取
   原来的前视 RGB/depth；环境 step、MPC、终止和 metric.csv 计算未改变。补丁固定在
   `config/xnavdp-metrics-only.patch`，由唯一 runtime 准备脚本原子应用。
8. **当前 Acados API。** 同一固定补丁将 X-NavDP 的已删除
   `ocp.code_gen_opts` 调用改为锁定 acados 0.5.1 的
   `ocp.acados_lib_path`/`ocp.code_export_directory`。这是唯一实现，不保留旧 API
   分支；launcher 自动将固定 Acados `lib` 注入 evaluator，环境脚本安装固定 SHA
   的 `t_renderer`。MPC 模型、约束、求解参数和生成目录不变。
9. **无空转的场景收尾。** launcher 等待 evaluator 时直接阻塞在子进程退出事件，
   不再先检查一次再固定睡眠 1 秒；取消检查仍保持最长 1 秒响应。相同 50 ms 子进程、
   同机各 5 次 A/B：旧链平均 `1.000193 s`，新链平均 `0.113571 s`，该控制面组件
   加速 `8.807×`、耗时下降 `88.645%`。固定协议、仿真和模型均未变化；这不是
   episode 端到端加速比例。
10. **NavDP 无分配历史与单次 letterbox。** NavDP 的 8 帧 RGB 历史在 reset 时一次
    分配，step 只原位移位，不再每次构造 deque、分配并复制完整历史。官方
    `B=1, 8×224×224×3 uint8` 同机 7 组中位数从 `1.359330 ms` 降到
    `0.339827 ms`，组件加速 `4.000×`、耗时下降 `75.000%`。官方 `640×360`
    RGB/depth letterbox 删除输出完全相同的第二次 resize，RGB 从 `0.229431 ms`
    降到 `0.128703 ms`（`1.783×`、下降 `43.903%`），depth 从 `0.149998 ms`
    降到 `0.081976 ms`（`1.830×`、下降 `45.349%`）。随机输入逐像素与连续
    12-step/reset 历史均与旧链完全相同；协议、模型、MPC和 episode 顺序未变化。
11. **SanD 单一推理与轻量 reset。** 删除临时 PNG/default trajectory fallback、
    丢弃结果的 Matplotlib 轨迹投影、重复 CUDA synchronize/empty-cache、GC 和 mapper
    清理，只保留 depth cache、warm-start 与 vector ESDF ranking。真实 checkpoint、
    GPU0、同进程各 7 次 episode reset 的中位数从 `384.287 ms` 降到 `3.250 ms`，
    加速 `118.227×`、耗时下降 `99.154%`。被丢弃的 16×41 候选 RGB 投影单次为
    `11.414 ms`，删除后该组件耗时 `0`、下降 `100%`；由于输出被完全移除，其加速
    倍数不定义。生产 CUDA 库路径下 5 次真实推理 P50 `179.972 ms`，reset P50
    `3.813 ms`，显存始终 `1116 MiB`、增长 `0 MiB`。响应仍为 finite float32 NPZ
    trajectory；同 seed/checkpoint/input 的旧、新成功路径均为 `[1,40,3]`，SHA256
    均为 `6e33ff47932ffc3f499b525d2202770e17d79b33e5911bd71fc382b138675467`。
    协议、selector 排名、模型和 MPC 未改变。
12. **X-NavDP 两端 raw runtime。** runtime 准备脚本现在同时安装并锁定 raw client、
    raw policy server 和无可视化 policy agent；不再出现 raw evaluator 对接上游
    PIL/uint16 server 的混合链。活动 server 只读取 uint8 RGB 与 float32-meter depth，
    只返回 float32 NPZ trajectory。旧混合链不符合协议，不能作为性能基线，因此本项
    明确为“未计算比例”；后续只对当前有效 raw 链做同条件 A/B。
13. **异构主机加权分片。** 同一个 launcher 增加确定性的 smooth weighted
    round-robin scene 分配。两端传入完全相同的 `--shard-weights`，再分别选择
    `--shard-index`；等权配置与原来的交错等分完全一致。主机内仍由动态 GPU queue
    消化本 shard，跨主机不需要共享数据库或第二套调度器。多 root 汇总现在要求
    每个 shard 的 suite/runtime/model/checkpoint/seed/num-envs/权重身份一致、索引齐全、
    scene 覆盖完整且无重叠。该优化只改变整 scene 被哪台主机执行，不改变 scene 内
    episode、相机、模型、MPC、阈值或指标。

    40 个等成本 scene、主机容量 `2:1` 的调度器 A/B（9 组中位数，快/慢主机每
    scene 等效 sleep 5/10 ms）中，旧等分 `20+20` 为 `0.200872 s`，新加权
    `27+13` 为 `0.135549 s`，加速 `1.482×`、耗时下降 `32.52%`；容量模型利用率
    从 `66.67%` 提高到 `98.77%`。这是 CPU 调度器合成验证，GPU 利用率不适用，
    不能替代双机 Isaac 端到端 A/B；完整 GPU 比例在两端同模型、同 40 scene
    实跑前标记为“未计算比例”。固定协议未改变。
14. **固定 simulator seed。** 吞吐 A/B 暴露出原 runtime 只把 1234 传给 policy
    server，而 `DingoEvalPointNavCfg.seed` 仍为 `None`；Isaac 日志也明确打印
    `Environment seed : None`。固定 runtime patch 现在在构造
    `ManagerBasedRLEnv` 前唯一设置 `env_config.seed = 1234`，活动
    `env_wrapper.py` SHA256 为
    `2d66cb7453245b289c8f653c78063ac01ff8840bf18fdb70bd3f95c96ae06f80`。
    没有 seed 开关、无 seed fallback 或旧 runtime 分支。这是补齐原本声明的固定
    seed 合同，不改变场景、episode、控制器、模型或指标。
15. **只计算被正式 evaluator 消费的传感器与受控 Isaac 线程池。** PointGoal
    policy 的 YAML 只读取 `raw_rgb`、`raw_depth`、`goal_pose` 和 robot pose；固定
    runtime 因此不再额外计算未消费的 224×224 RGB/depth 历史，也不再创建未被
    reward、termination 或官方 metric 读取的 contact sensor。前视相机、raw
    RGB/depth、物理碰撞响应、policy、MPC、成功判定和 SPL 均保持不变。launcher
    同时向 Isaac 的既有 distributed evaluator 传入真实 `WORLD_SIZE=本机 worker 数`：
    旧链在 7 worker 时因缺失该值而给每个进程分配 64 个 Kit worker，新链分配
    `64 // 7 = 9` 个；七进程线程池预算由 448 降为 63，资源项减少 `85.94%`
    （`7.11×`）。该比例是线程池预算，不是 episode wall-time；相同 fixed episode
    的七卡端到端 A/B 尚未完成，wall-time、GPU 利用率和对应加速比例标记为
    “未计算比例”。没有加入第二条 launcher、fallback 或旧 runtime 分支。
16. **复用场景已有的 Goal view。** IsaacLab 的 `InteractiveScene` 已经为
    `scene["goal"]` 创建唯一 `XFormPrim`。固定 runtime 现在直接复用这个 view：
    reset 用一次带 `indices=env_ids` 的批量 `set_world_poses` 取代逐环境构造和设置，
    observation/reward 也不再每 step 新建同路径 view。一次错误的中间实现让仍启用的
    ObservationManager 直接读取 reset-time `_goal_pos_w`；实测证明 manager 会在
    reset event 之前探测 observation shape，因此该读法以 `AttributeError` 退出。
    当前固定 eval 已关闭该 manager term，只在 reset 后由 evaluator/reward 读取缓存；
    错误读法已删除，没有 `hasattr` fallback 或兼容分支。活动
    event/observation/reward SHA256 分别为
    `77a63ceb4085735b44d6586e25bfd85a37a90ceb3c0b2a6540f744602d4f6046`、
    `c1fed0f2e2d08c822cedcc679673de32de3013f21e0f2540d85e0dd47021698e`、
    `a17d81cf3c660950f69f5276a3c58c82605aa072c1af1279dd86bf1109864174`。

    同一 Home scene ep0、seed 1234、GPU0、CPU8、旧单环境基线、performance governor
    的端到端观测从 `1318.81 s` 变为 `1208.38 s`，原始比值为 `1.091×`、耗时少
    `8.37%`；GPU SM 平均值分别为 `23.553%` 和 `19.877%`，峰值均为 `90%`。
    但旧/新分别为 timeout failure（854 次动作更新）和 success（355 次动作更新，
    SPL `0.850473`），异步 planning thread 造成的工作量不同，故正式等价性能比例
    仍标记为“未计算”，不能引用 `1.091×` 作为论文结论。固定场景、起终点、相机、
    控制器、MPC、阈值、SPL 与 raw float32/NPZ 协议均未改变。
17. **批量 world-to-body 变换。** Goal observation 原来对每个环境执行一次 Python
    循环和通用 `torch.inverse(3×3)`。机器人姿态矩阵是正交旋转矩阵，其逆严格等于
    转置；当前唯一实现因此改为一次 `transpose + torch.bmm`，不增加分支。GPU0、
    Torch 2.7、同一随机正交矩阵/位移、5 组各 1000 次的中位数中，B=1 从
    `0.218371 ms` 降到 `0.022737 ms`（`9.604×`、下降 `89.588%`，最大绝对误差
    `1.19e-7`）；B=10 从 `1.946870 ms` 降到 `0.022276 ms`（`87.397×`、下降
    `98.856%`，最大绝对误差 `4.77e-7`）。这些是 observation 算子比例，不是
    episode wall-time；端到端比例仍遵守上一项的等工作量判定。
18. **固定协议的 10-episode 稳态基线。** NavDP、同一 Home scene、ep0--9、
    seed 1234、GPU0、CPU8、旧单环境基线、performance governor 完整运行
    `2273.24 s`，SR `50%`、mean SPL `0.499015`，10 个 episode ID 连续且无重复。
    从 launcher 开始到 evaluator 创建首个输出目录约 `1065 s`，作为“USD/Isaac
    场景构造 + 首次 policy reset”的冷启动边界；余下 `1208.24 s` 折算稳态
    `120.824 s/episode`。整段 GPU0 SM 平均 `52.687%`、峰值 `92%`；冷启动窗口
    平均 `15.198%`，稳态窗口平均 `85.664%`，峰值仍为 `92%`。结果位于
    `/mnt/data1/huangshibo/H/general-navigation-benchmark/runs/validation/`
    `perf-navdp-steady10-20260826/pointgoal-v2/navdp/20260826_140423`，连续 dmon 为同一
    validation root 下的 `gpu0-dmon.log`。

    与旧的“把 `1318.81 s` 单回合冷启动重复外推到每个 episode”相比，本次实际
    10 回合平均为 `227.324 s/episode`，观测比值 `5.801×`、平均耗时低
    `82.763%`。这主要是正确摊销同一 scene 的一次冷启动，不应表述为单个代码改动
    带来的 `5.801×` 加速；模型 outcome/episode 距离也不相同。它是当前全量 ETA
    使用的有效吞吐基线。
19. **reset-time goal pose 缓存。** 固定 evaluator 不再让 ObservationManager
    每 step 查询静态 Goal USD pose；`goal_pose` term 在固定 eval 配置中关闭，reset
    事件在批量设置 Goal 后查询一次 `[B,3]` world pose，evaluator 与 arrival reward
    共用该 tensor，并用与上游一致的 `Rᵀ(goal−robot)` 生成机器人系 x前/y左目标。
    训练 observation 函数仍保留原场景 view，不修改训练模型。Isaac math 模块只在
    AppLauncher 创建后导入；一次放在模块顶层的错误中间实现因 `omni.log` 尚未注册而
    在 `11.65 s` 内退出，已完全删除，没有 import fallback。

    同一 Home ep0 timeout、seed1234、GPU0、CPU8、旧单环境基线、相同 governor 和
    仿真 step workload 的 A/B：旧 `1318.81 s`、新 `1317.14 s`；端到端加速
    `1.001268×`、耗时下降 `0.126629%`，节省 `1.67 s`。异步 policy action 更新数
    分别为 854/838，因此该 wall 比例仍有调度噪声，但 success、SPL、distance 与
    timeout step workload 相同。裁齐 wall 窗口的 GPU SM 平均为
    `26.140%/26.405%`，峰值 `90%/91%`；max RSS 下降 `0.0129%`。新 run 位于
    `/mnt/data1/huangshibo/H/general-navigation-benchmark/runs/validation/`
    `perf-goal-cache-v3-20260826/pointgoal-v2/navdp/20260826_145226`。缓存链的
    `evaluate_pointgoal.py`/`env_wrapper.py` SHA256 分别为
    `375ffb13e9a618814b7e76a059418ef3f6e7e2446d70b9d871c5ac66ee2403e6`、
    `c6a6e13415bc05c3b7467efb6454f0fe292d157b46006a72496b2a7b2dab952a`。

20. **严格的 vector episode 生命周期。** 旧向量路径按当前 active env 数累加 ID，
    在不同步 reset 时可能重复或错配 episode。当前唯一 runtime 使用单调 sample
    allocator、每槽 generation 和 target-prefix metric gate；Goal 也通过一次创建的
    replicated `XFormPrim("/World/envs/env_.*/Goal")` 批量更新，不再把单 prim view
    当作多环境 view。3090、首个官方 Home scene、seed 1234、NavDP、GPU0、CPU8、
    `num-envs=4` 的跨 reset 门禁最终得到且只得到 ID `0--5`，无重复/缺口/越界；完成
    顺序 `2,4,5,0,1,3`，SR `0.5`、mean SPL `0.4992697719`。wall time
    `1727.01 s`，其中约 `1273 s` 在首条 metric 前的首次场景构造/cooking；该门禁
    的目的为正确性，尚无同工作量单环境 A/B，因此正式端到端加速比例标记为
    **未计算比例**。结果在
    `/DataDisk/hsb/eval-server-audit/runs/vectorfixed-prefix-v3/pointgoal-v2/navdp/20260826_172125`。
21. **稳定的官方输入缓存。** launcher 不再为每个时间戳 run 复制出新的资产路径。
    它按 suite SHA 与规范 scene-root 路径建立一次只读语义的
    `~/.cache/navbench/official-inputs/<suite>-<scene-root>/data`：小型官方 episode/robot
    文件复制，较大的官方场景与 ESDF 只做符号链接；每个 run 再链接到这一稳定路径。
    这允许 Kit/PhysX/OptiX 以相同资产 key 复用可再生 cache，同时 run.json 记录解析后
    的 `input_root`。建立过程用同文件系统 rename 原子发布；未完成 cache 会拒绝运行，
    不提供第二路径 fallback。相同 NavDP、scene、ep0--5、seed、GPU0、CPU8、B4
    重复测量中，首次 wall `1727.01 s`，再次运行 `1747.41 s`；热缓存比值
    `0.988x`，反而慢 `1.18%`。simulation-start 也从 `1243.33 s` 变为
    `1276.06 s`（`0.974x`，慢 `2.63%`）。两次均为同一 3/6 outcome，第二次采样
    GPU SM 平均 `26.655%`、峰值 `98%`。结论是稳定路径只保留为一致性与 cache-key
    卫生改进，端到端加速收益为 **0**；真正的主因是每进程的 RTX/物理场景初始化。
    第二次结果在
    `/DataDisk/hsb/eval-server-audit/runs/vectorfixed-cache-repeat/pointgoal-v2/navdp/20260826_175550`。
22. **3090 的 B10 主机门禁。** 当前 lifecycle runtime 在 GPU0、CPU8、seed1234、
    首个官方 Home scene 上用 `num-envs=10` 完成 ep0--9，最终 ID 集合严格为
    `0--9`，无重复、缺口、越界或 traceback。wall `2018.97 s`，SR `0.7`、mean SPL
    `0.6729579853`；GPU SM 403 个 5 秒样本平均 `34.978%`、峰值 `100%`。与紧邻的
    B4/6 条复跑按“wall/完成 episode”比较，`291.235 -> 201.897 s/episode`，吞吐
    `1.442x`、平均耗时下降 `30.68%`，GPU SM 平均相对提高 `31.22%`（绝对
    `+8.323` 个百分点）。两轮 episode 数和 outcome 不同，因此这是主机容量诊断，
    不是严格 matched A/B；不能把 `1.442x` 当作模型分数结论。B10 与 B4 的
    simulation-start 分别为 `1277.53/1276.06 s`，B10 只慢 `0.115%`，说明额外槽位
    几乎不增加固定启动成本。B10 结果在
    `/DataDisk/hsb/eval-server-audit/runs/vectorfixed-b10-gate/pointgoal-v2/navdp/20260826_182714`。
    该结果先验收了 B10，但随后第 23 项的 B16 容量门禁取得更高吞吐。
23. **3090 最终 B16 host profile。** 在完全相同 GPU0、CPU8、NavDP checkpoint、
    scene、seed 和 lifecycle runtime 上，B16 完成官方 ep0--15，ID 集合严格连续，
    无重复/缺口/越界或 traceback。wall `2342.90 s`，SR `0.625`、mean SPL
    `0.5924020445`；GPU SM 473 个 5 秒样本平均 `42.693%`、峰值 `100%`，显存峰值
    `13328 MiB`。相对 B10，单位 episode 从 `201.897` 降到 `146.432 s`，吞吐
    `1.379x`、耗时下降 `27.47%`；GPU SM 平均相对提高 `22.06%`（绝对 `+7.715`
    个百分点）。相对 B4 容量样本，单位吞吐 `1.989x`、耗时下降 `49.72%`。这些
    容量样本的 episode 数/outcome 不同，因此比例用于选 host batch，不用于模型分数
    比较。B16 结果在
    `/DataDisk/hsb/eval-server-audit/runs/vectorfixed-b16-gate/pointgoal-v2/navdp/20260826_190604`。
    4x3090 活动 profile 最终锁定 `4 GPU x 16 env/GPU`；每个 worker 一次跑完一个
    scene 的全部 100 条，四卡动态领取 40 个 scene。不继续扩大 batch，以保留四个
    并发 Isaac/模型进程的显存与宿主内存余量。
24. **场景常驻的 checkpoint 序列。** 唯一 launcher 的 `--checkpoint` 可以重复；
    worker 以 scene 为生命周期创建一次 `navbench.scene_evaluator`，然后逐 checkpoint
    启动同一个 adapter 产生的模型服务。每轮开始把 `_next_sample_idx` 归零并执行完整
    `env.reset()`，重新创建 planner generation/active mask，调用 policy 的 batch reset，
    清空动作队列并重新累计 metric；轮末只关闭该 checkpoint 的 server，最后一个
    checkpoint 完成后才关闭 Isaac scene。每个 checkpoint 使用独立 SHA 命名的 run
    root、server log、scene metadata、CSV 和 summary，resident evaluator log 通过绝对
    路径记录在各 scene metadata 中。该实现删除了 launcher 对上游一次性 evaluator
    脚本的依赖；固定 runtime patch 只保留环境、episode allocator、传感器和 MPC 的
    唯一实现。两 checkpoint 控制通道、CLI dry-run、compile 与 10 项 contract tests
    已通过。3090 的旧单环境 smoke 在同一 evaluator PID 中完成 step 800 和
    step 2400：首次 scene ready 约需 23 分钟，两轮执行分别为 `171.16 s` 和
    `165.12 s`，第二轮前没有 scene load；两份 episode 0 均为 timeout，起始距离均为
    `5.2486815 m`。相同 step 800 的旧一次性 evaluator 冷启动对照也是
    `success=0, SPL=0, distance=5.2486815, episode_idx=0`，逐字段一致；从 session
    创建到 metric 落盘，冷启动单 checkpoint 为约 `1564 s`，常驻两 checkpoint 为
    约 `1721 s`。按同一冷启动成本外推两个 checkpoint，端到端从约 `3128 s` 降至
    `1721 s`（约 `1.82×`，墙钟降低 `45.0%`）；增加第二个 checkpoint 的边际成本
    只有约 `169 s`。独立结果位于
    `/DataDisk/hsb/eval-server-audit/runs/curvenav-persistent-smoke-20260827`。
    初次 smoke 还发现 Isaac `simulation_app.close()` 会在发出自定义 closed 事件前直接
    结束进程；最终链路删除该冗余事件，以进程退出码作为唯一关闭完成信号，并恢复上游
    `10 ms` planning poll 与原始距离公式。最终代码的双 checkpoint smoke 位于
    `/DataDisk/hsb/eval-server-audit/runs/curvenav-persistent-smoke-final-20260827`：同一 scene
    evaluator 在一次初始化后先后完成两轮，执行阶段分别约 `182.91 s`、`166.18 s`，
    两份 episode 0 仍逐字段等于冷启动对照；Isaac 以退出码 0 完整关闭，两组
    `episodes.csv` 和 `summary.csv` 均正常生成。上述加速比例是同场景多 checkpoint
    的真实吞吐诊断，不改变固定协议的模型成绩定义。

    随后用 CurveNav `9a9ca17` 的五个真实训练里程碑完成同一 Home scene、seed1234、
    `num_envs=10`、10 episode/checkpoint 的架构反馈。GPU0 的两 checkpoint session
    只出现一次 `[load]`，scene ready 约 `1242 s`，step800/2400 分别执行
    `466.54/422.12 s`；若各自冷启动，估计为约 `3372.65 s`，实际常驻约
    `2130.65 s`，即 `1.583×`、墙钟降低 `36.83%`。GPU1 的三 checkpoint session
    同样只出现一次 `[load]`，scene ready 约 `1285 s`，step4800/6400/8000 分别执行
    `434.91/450.49/581.98 s`；逐 checkpoint 冷启动估计约 `5322.37 s`，实际常驻约
    `2752.37 s`，即 `1.934×`、墙钟降低 `48.29%`。这里的估计只复用各 session
    实测的同一次冷启动成本；不是跨机器或固定协议加速比例。

    五份结果都完整生成 10 个 episode：step800 为 SR `0.1`、mean SPL
    `0.08686945`，step2400/4800/6400/8000 均为 SR/SPL `0`。这组 B10 只用于模型
    架构反馈，不作为固定协议成绩。两份绝对 session root 为
    `/DataDisk/hsb/eval-server-audit/runs/curvenav-direct-9a9ca17-b10-persistent-20260827/pointgoal-v2/curvenav/20260827_083453`
    和
    `/DataDisk/hsb/eval-server-audit/runs/curvenav-direct-9a9ca17-late-b10-persistent-20260827/pointgoal-v2/curvenav/20260827_090109`。
25. **模型无关的长期常驻场景队列。** `--checkpoint-queue DIR` 继续使用同一个
    `python -m navbench` launcher、`ServerSpec`、HTTP/NPZ 协议和
    `navbench.scene_evaluator`。每个选定 scene 独占一张 GPU，完成一次 Isaac
    初始化后保持 evaluator PID；初始 `--checkpoint` 运行结束后，launcher 每两秒
    检查 `DIR/*.ready` model bundle，并广播给所有常驻 scene。bundle 固定包含
    `artifact.json` 和 `checkpoint.pt`；`artifact.json` 的唯一 schema 为
    `{"schema":"navbench-model-artifact-v1","model":"<ADAPTER>"}`。CurveNav
    额外必须包含 `configs/base.yaml` 和 `src/curvenav/`。launcher 将 adapter 名称、
    权重、配置和全部 CurveNav 推理 Python 源码联合哈希，因此同一权重经不同 adapter
    执行不会被错误去重，架构代码、配置与权重也不会错配。

    resident worker 只拥有场景和 evaluator，不绑定模型。每个 target 到来时，调度层
    根据其 model adapter 在同一端口启动对应 server，重置并执行固定 episode，完成后
    关闭 server、生成 `models/<MODEL>/<SHA>/` 独立结果和 summary，但不关闭 scene。
    因此 CurveNav、NavDP、SanD 和 X-NavDP 可顺序复用同一个已加载场景。一个仿真状态
    不能同时表示四条不同策略时间线；真正同时测评仍需四个 evaluator，而快速公平对比
    使用单场景顺序切换，只承担一次 Isaac 冷启动。长期模式要求
    `scene 数 <= GPU 数`；四张 3090 最多固定四个快速回归 scene，完整 40-scene
    正式测评仍使用动态 scene queue。该模式不与 `--resume-root` 混用，SIGINT/SIGTERM
    通过现有 launcher 统一关闭 server 和 evaluator，不增加第二套入口或 fallback。

    单场景快速回归服务的唯一命令形状为：

    ```bash
    python -m navbench --model curvenav --gpus 0 \
      --scenes home/MVUCSQAKTKJ5EAABAAAAABA8_usd \
      --episodes-per-scene 10 --num-envs 10 --cpu-threads-per-worker 8 \
      --checkpoint <INITIAL_CHECKPOINT> --checkpoint-queue <INCOMING_DIR> \
      --model-config <CONFIG> --output-root <RUN_ROOT>
    ```

    新版本的交付顺序为“将完整 model bundle 同步到
    `<INCOMING_DIR>/<NAME>.partial/`，校验完成后在 3090 同一文件系统原子改名为
    `<INCOMING_DIR>/<NAME>.ready/`”。`.ready` 是唯一提交标志，launcher 不读取
    `.partial`；`.ready` 名称不可复用。launcher 首次联合哈希后记录已消费路径，
    后续轮询不再每两秒重复读取数百 MB 权重。NavDP 同一场景三次历史日志的
    `simulation start` 为 `1296.44/1264.89/1276.06 s`，证明约 21 分钟成本来自
    公共 Isaac 场景启动而非 CurveNav 模型；常驻队列正好消除后续权重的重复成本。

    2026-08-28 在 3090 GPU1 验证了第一版单模型长期服务
    `curvenav-resident-fast-home`，固定 Scene-N1 Home、`num-envs=10`。进程只输出
    一次 `[load]`，约 `1226 s` 后输出 `[resident] 1 scenes ready`；首个完整
    release 随后在约 `6.8 min` 内完成 10 个 episode，SR `5/10`、mean SPL
    `0.4769219367`，然后 evaluator PID 保持存活并继续监听
    `/DataDisk/hsb/eval-server-audit/queues/curvenav-fast-home`。结果位于
    `/DataDisk/hsb/eval-server-audit/runs/curvenav-resident-fast-home-20260828/pointgoal-v2/curvenav/20260828_102857/checkpoints/00-972ec7de1d7f/`。
    随后的 NavDP/SanD/X-NavDP 并行启动暴露出“launcher 绑定 model”会为同一 USD
    重复创建三个 Isaac 进程，因此该绑定已从 worker 中删除，以上 model bundle
    contract 是当前唯一实现。

## 3090 实测瓶颈

2026-08-25 在 `ubuntu-3090`、GPU0、Scene-N1 Home 首场景、seed 1234 上完成了
10 个 episode 的吞吐诊断：`--num-envs 10`，CPU threads 8。Isaac 和 server
均正常，GPU1--3 空闲；约 1219 个动作更新后 10 个 episode 全部写出指标。日志
中出现 Acados solver status 2 警告，但没有 policy/HTTP/Isaac traceback。主要
耗时在每一步批量 MPC/碰撞约束求解，而不是模型 checkpoint 加载；因此继续
增加 `num-envs` 不一定线性提速。

本次历史诊断写在独立 run root。它发生在第 20 项生命周期修复之前，因此只用于
吞吐观测，不能作为当前结果正确性的证据。当前加速 workload 允许向量执行，但必须
通过严格 ID gate，并在比较中统一 `num_envs`；四卡同时继续按 scene 动态分工。

### Policy/MPC 分解（2026-08-25）

在 GPU0 的独立 HTTP 探针中，CurveNav server 的严格 smoke 输出始终为
`float32 [B,64,3]`。`B=1` 时稳态 policy 请求 P50 约 `110.5 ms`；批量
`B=2/4/8/10` 时总请求 P50 约 `124.6/136.6/167.0/185.5 ms`，折算每环境
约 `62.3/34.2/20.9/18.5 ms`。因此批量推理有效，但完整仿真的主耗时仍是
上游逐环境 Acados MPC solve；status 2 警告来自该 solver，不能通过关闭 MPC
或改变控制器来“加速”。当前最有效的通用方案是四卡按 scene shard 并行。

本轮相对 B=1 基线的性能比例为：

| 配置 | P50 / env | 相对基线加速 | 单环境耗时下降 |
|---|---:|---:|---:|
| B=1 基线 | 110.5 ms | 1.00× | 0% |
| B=4 | 34.2 ms | 3.23× | 69.1% |
| B=10 | 18.5 ms | 5.97× | 83.3% |

### 可视化热路径（2026-08-25）

旧 evaluator 每个仿真 step 额外渲染 bird-eye camera，并将两路 `640×360` RGB
同步回 CPU、缩放为 `384×384` 后编码 MP4。相同尺寸的纯 CPU 复现为 60 step
耗时 `0.431974 s`，即 `7.200 ms/step`；该数字尚未包含 Isaac 的第二相机渲染和
GPU 同步，因此只是可删除开销的下界。新 runtime 完全移除这条可视化热路径。
在完成同 scene、同 episode 的端到端 A/B 前，加速倍数标记为“未计算比例”。

### 端到端 vector-env 诊断比例（同一 Scene-N1）

为避免把 policy-only 延迟误报成仿真吞吐，另外做了同一
`MVUCSQAKTKJ5EAABAAAAABA8_usd`、同一 CurveNav checkpoint/runtime、同一 seed
的 wall-time 对照。旧单环境对照在 episode 0 写完后停止，保留该行和完整
日志；`num-envs=10` 已完成 10 个 episode。两者都只是单场景吞吐诊断，不能作为
固定协议分数。

| 配置 | 完成内容 | wall time | 等效每 episode | 相对基线加速 | 耗时下降 | GPU 利用率采样 | 协议 |
|---|---:|---:|---:|---:|---:|---|---|
| 旧单环境基线 | ep0/10 | 264.99 s | 264.99 s | 1.00× | 0% | GPU0 约 63–71%，GPU1–3 0% | 诊断，未完成 10 回合 |
| `num-envs=10`（优化） | 10/10 | 740.36 s | 74.04 s | **3.58×** | **72.1%** | 本次未保留连续利用率采样；GPU0 为唯一 worker | 诊断，vector-env，非固定协议 |

比例按 `264.99 / 74.04 = 3.58×`、`1 - 74.04 / 264.99 = 72.1%` 计算。该结果
证明受控 vector-env 对这个场景有效，但不能推出四卡或完整 40-scene 的端到端
比例；后者必须等官方资产完整且有相同 episode 数的串行/分片基线后再测。
对照 run root 为
`/DataDisk/hsb/eval-server-audit/runs/curvenav-619fa608-numenv1-baseline-20260825`
（episode-0 metric SHA256 `e93941a5c63497240d15dacb39c7b177ec232c0088df1d1aa56e4743415bbb36`，
tmux log SHA256 `416598bf0eba317623bbfcffa99ab74feedb5e4d11b44fca733f44c758d0e9f7`），优化 run root 为
`/DataDisk/hsb/eval-server-audit/runs/curvenav-619fa608-numenv10-20260825`，其
10 行 metric SHA256 为
`895abb47fe741dd99e6f6c47b4aa7b534e69c8c99f2f20d305e63e73082603be`。

### 2080Ti scene 并发诊断（2026-08-26）

NavDP、同一 Home/Commercial 各 ep0、seed 1234、CPU8、旧单环境执行的单卡串行
与双卡并行链路均完整退出并生成 2 行指标。单卡 wall time `2470.98 s`，双卡
`1705.32 s`；原始观测值为 `1.449×`、耗时下降 `30.99%`。但这不是有效的等价
性能比例：单卡为 Home 成功/Commercial 失败，双卡恰好相反。无 seed runtime 和
上游异步 planning thread 使动作替换时机受运行负载影响，因此本轮端到端正式比例
标记为“未计算比例”，不能用 `1.449×` 作为论文或优化结论。

单卡 Commercial 的连续 20 分钟采样中 GPU0 SM 平均 `27.863%`；双卡的连续
25 分钟采样中 GPU0/GPU1 分别为 `22.613%/8.430%`。采样窗口不同，也只用于定位
GPU 未饱和，不能直接相减。诊断结果分别位于
`/mnt/data1/huangshibo/H/general-navigation-benchmark/runs/validation/distributed-ab-20260826/baseline-1gpu/pointgoal-v2/navdp/20260826_095909`
和
`/mnt/data1/huangshibo/H/general-navigation-benchmark/runs/validation/distributed-ab-20260826/optimized-2gpu/pointgoal-v2/navdp/20260826_104044`。
修复 seed 后必须用相同 outcome/step count 或固定仿真工作量重新 A/B，才能发布新的
端到端比例。固定协议没有为加速而改变；本轮只修复了其原本缺失的 seed 实现。

### 一天目标差距（2026-08-26）

旧的两 scene/两 episode 单卡 `2470.98 s` 被错误线性外推为 `205.92 h`
（7 GPU、40×100），相当于把每 scene 只发生一次的冷启动重复计算 100 次。当前
10-episode 固定协议实测把一 scene 的成本分解为约 `1065 s` 冷启动和
`120.824 s/episode` 稳态；同速场景假设下，100 episode/scene 约
`13147.4 s = 3.652 h`。40 scene 动态分给 7 张卡，平均下界约 `20.869 h`；因
40 不能整除 7，等成本最慢 worker 承担 6 scene，保守 ETA 约 `21.912 h`，比一天
短 `8.70%`。相对旧 `205.92 h` 外推，计划估计缩短 `9.397×`、下降 `89.359%`；
这是修正冷启动摊销后的容量估计变化，不是代码端到端加速倍数。

该 ETA 只基于一个 Home scene 的 5 success/5 timeout 样本；不同 scene 的 USD
复杂度和 success/timeout 比例仍会形成长尾。因此可以说“7 张当前 2080Ti 在本样本
下进入一天窗口”，不能承诺所有 40 scene 必然低于 24 小时。正式开跑前应再取一个
Commercial scene 做同样 10 回合校准；运行时保持动态 scene queue，让短 worker
自动接手下一 scene。上游 X-NavDP 使用单环境配置，因此旧结果只用于
“逐项复现上游执行拓扑”的独立模式；当前“同官方场景/同 100 次”的加速 workload
统一使用验收后的 vector batch，并在结果中明确记录 `num_envs`。

旧错误线程配置实测为 247 threads；运行中 RSS 约 45 GiB、GPU0 约 9.8 GiB。
当前宿主 governor 已一次性切到并验证为 `performance`，固定 evaluator 的 Kit
线程池也按真实 worker 数分配。最新同 outcome/step workload A/B 已列在第 19 项；
其提升很小但有效。一天 ETA 的主要改变量来自同 scene 100 episode 正确摊销一次
冷启动和 7 卡 scene queue，而不是放宽协议或伪报某个微优化倍数。

以后每份性能报告必须同时给出：固定基线、优化配置、绝对耗时、加速倍数
`baseline / optimized`、耗时下降比例
`1 - optimized / baseline`、GPU 利用率，以及是否改变固定协议。没有可比
基线时必须标记“未计算比例”，不能只报告单一吞吐数字。

3090 的活动 scene root 已通过完整静态门禁：20 Home + 20 Commercial、每场景
100 个官方 episode，共 4,000 条，PLY/episode/provenance 校验均通过。四卡运行使用
同一个动态 scene queue；每卡一次加载一个完整 scene，并在该进程内跑完 100 条，
避免把昂贵的 Isaac scene start 重复到每个 episode：

合并后的 server lifecycle smoke 已通过：`/navigator_reset` 返回实际
`candidates=16`，`/pointgoal_step` 返回 finite `float32 [1,64,3]`，
`/shutdown` 返回 HTTP 200。证据为
`/DataDisk/hsb/eval-server-audit/runs/curvenav-accel-smoke-20250825/smoke.json`
（SHA256 `98fcdf5d64d47d956036b88b368f0f4b07b5af20f75c028dfc4fa28612804c9a`）。

```bash
python -m navbench --model curvenav --gpus 0,1,2,3 \
  --num-envs 16 --cpu-threads-per-worker 8 \
  --launch-stagger 10 --checkpoint <CHECKPOINT> --model-config <CONFIG> \
  --output-root <RUN_ROOT>
```

## 推荐命令形状

固定协议（可比较分数）。先用各主机“完成 episode 数 / wall time”测量容量，化为
最小整数比；两端使用完全相同的 `--shard-count 2 --shard-weights <FAST,SLOW>`，
分别设置 `--shard-index 0` 和 `1`。若实测总吞吐相同就使用 `1,1`，不要按显卡型号
猜权重。每台机器内部继续用动态 scene worker 跑满可用卡：

```bash
python -m navbench --model curvenav --gpus 0,1,2,3 \
  --cpu-threads-per-worker 8 --shard-index <0-or-1> --shard-count 2 \
  --shard-weights <FAST,SLOW> \
  --launch-stagger 10 \
  --checkpoint <CHECKPOINT> --model-config <CONFIG> \
  --output-root <RUN_ROOT>
```

两个 shard 都完成后，用同一个指标入口严格合并；root 顺序无关：

```bash
python -m navbench.metrics \
  --root <SHARD_0_RUN_ROOT> --root <SHARD_1_RUN_ROOT> \
  --output <MERGE_ROOT>/episodes.csv --summary <MERGE_ROOT>/summary.csv \
  --expected 4000
```

同场景/同次数加速执行（整组比较必须使用相同 `num-envs`）：

```bash
python -m navbench --model curvenav --gpus 0,1,2,3 \
  --num-envs 16 --cpu-threads-per-worker 8 --launch-stagger 10 \
  --checkpoint <CHECKPOINT> --model-config <CONFIG> \
  --output-root <RUN_ROOT>
```

## 继续优化的边界

- 先用 profiler 区分 server policy latency、Acados solve latency、Isaac step
  latency；只有在结果与固定协议完全等价时，才把优化提升为默认值。
- 可以复用已生成的 OptiX/Kit/shader cache，但不能清空他人 cache，也不能以
  `CUDA_VISIBLE_DEVICES` 重新映射 Isaac 5 的物理卡。
- 不关闭 MPC、不替换 controller、不放宽阈值、不跳过官方 PLY 校验；这些会
  改变 benchmark 语义，不能作为“加速”。
- checkpoint、场景资产、私有凭据和测评结果不进入 Git；只提交 launcher、adapter
  和本说明文档。

## 本轮审计与验证

当前修改基线为 `main@4d4035d`。
审计覆盖 `navbench/cli.py`、`adapters.py`、`client.py`、`protocol.py`、`metrics.py`
及所有内置 policy server。当前只有一个 CLI launcher、一个 adapter 注册表、一个
动态 scene queue、多卡 worker 分配器和一套 raw tensor/NPZ 协议；`--resume-root`、
scene shard、异构权重、resume 与诊断用 `--num-envs` 均由该 launcher 提供。

本轮补齐 IPlanner、VIPlanner、SanD server 的 `/shutdown` 端点，并从固定 runtime
删除仅用于视频的 bird-eye sensor 与 MP4 热路径。未引入新的 launcher、模型实现、
fallback 或旧协议兼容；未改场景、起终点、前视相机、MPC、controller、阈值、
timeout、SPL 和权重。向量结果只在通过 ID 生命周期门禁且所有对比模型使用相同
`num_envs` 时纳入同场景/同次数 workload 比较；它不等同于上游单环境
执行复现。

当前验证项目：全仓 Python compile、10 项 contract test、CLI `--help`、NavDP
完整 40 scene/4000 episode dry-run、`2:1` shard 的 `27+13` 覆盖/零重叠验证，
以及多 root 身份/完整性/重复拒绝测试。已有 raw float32 depth/NPZ HTTP smoke
继续由同一 contract test 覆盖。重建后的 X-NavDP runtime hash/dependency/raw bridge
contract 通过；dry-run 同时确认 seed 1234 与 `27 scene/2700 episode` shard 身份。
验证不启动 Isaac、quick100 或正式测评。对应日志写入运行机的
`runs/validation/acceleration-<timestamp>/`，不进入 Git。

## 4090 用户态常驻测评运行时（2026-08-30）

4090 宿主为 Ubuntu 18.04、NVIDIA 驱动 `535.261.03`，没有 sudo。Isaac Sim 5.0
虽然能到达 `app ready`，但会在 RTX shader 初始化阶段崩溃；因此正式链路固定为
该主机已经真实闭环验证的单一用户态组合：Python `3.10.20`、Isaac Sim
`4.2.0.2`、Isaac Lab v1.2.0（包版本 `0.24.13`）、Torch
`2.4.0+cu121`、Open3D `0.18.0`、Acados template `0.5.1` 和 X-NavDP
`878740a2011856d0e3782dd6ccd880fd2eccd70f`。这不是第二条 fallback：4090
只保留这一条可执行 evaluator；模型权重、官方 USD/PLY/episode、Dingo、相机、
controller、MPC、成功阈值、timeout 和 SPL 定义均未改变。结果应标记为
Isaac Sim 4.2 兼容复现，不能冒充 Isaac Sim 5 的二进制级复现。

正式运行时路径：

```text
evaluator python: /DataDisk2/hsb/navdp/envs/isaacsim42/bin/python
Acados:          /DataDisk2/hsb/eval-server-prep/runtime/acados-48e223e85f04
X-NavDP source:  /DataDisk2/hsb/general-navigation-benchmark-resident/.runtime/x-navdp-878740a20118/baselines/x-navdp
server python:   /DataDisk2/hsb/eval-server-prep/bin/ubuntu22-python
scene root:      /DataDisk2/hsb/eval-server-audit/assets/scene-n1/n1_eval_scenes
queue:           /DataDisk2/hsb/eval-server-audit/queues/four-model-home-b100-20260830
```

X-NavDP 的 evaluator 移植集中在唯一
`config/xnavdp-rootless-runtime.patch`：只导入 Dingo wheeled 所需模块，统一旧版
`omni.isaac.*` API，使用 `env.scene["goal"]` 的唯一 Goal view，按全局顺序分配
episode，并使用 Acados 0.5 的正式 `ocp.acados_lib_path/code_export_directory`
字段。被替代的 Isaac 5 constraints、旧 metrics-only patch、G1/Go2 evaluator
导入和 bird-eye/video 热路径已删除。

Isaac Sim 4.2 会接管启动前已有的 stdin/文件描述符，因此 resident 控制面不继承
任何控制 FD。evaluator 完成场景构建后才绑定唯一 Unix domain socket，再发出
`ready`；launcher 收到 `ready` 后连接，`run/close` 只走该 socket，Kit 的 stdin
固定为 `/dev/null`。这保证模型切换时不会让常驻场景误读 EOF。

真实闭环进一步定位到一个独立的设备契约错误：Dingo 差速控制器从 CPU 上的 MPC
命令生成 CPU action，而 GPU Isaac 环境使用 CUDA joint indices。旧链路因此在首次
`env.step` 报出 CPU tensor/CUDA index 不一致，并在 Kit 关闭阶段表现为 evaluator
正常退出。唯一 evaluator 现在在 controller/Isaac 边界将 action 放到 policy
observation 所在设备；控制器、MPC 数值和动作限制均未改变。失败现场保存在
`/DataDisk2/hsb/eval-server-audit/failed/action-device-mismatch-20260830/`。

四卡并行时采用一张卡一个模型，而不是在单卡串行切换四个 policy server。
Isaac Lab 1.2 会正确设置每个进程的 `active_gpu/physics_gpu`，但 SimulationApp 的
默认 `multi_gpu=True` 仍会让每个 renderer 在其余三张卡创建上下文并参与调度。
唯一 evaluator 因此显式设置 `multi_gpu=False`；每个进程只保留所属 GPU 上的场景、
传感器、物理和模型，四个模型仍使用相同的官方 100 个 episode。被停止的预热现场
保存在 `/DataDisk2/hsb/eval-server-audit/failed/multigpu-render-context-20260830/`。
Isaac Lab 1.2 把同一个物理 ordinal 同时用于 CUDA physics 与 Vulkan
`active_gpu`；不能用 `CUDA_VISIBLE_DEVICES` 把 CUDA 重映射成逻辑 0，否则相机图找不到
renderer。进程因此继续接收物理 `cuda:N`，而多个模型的首次 MDL/LLVM 编译按顺序
预热；场景 ready 后才并行执行，避免冷启动 JIT 争用。
`--cpu-threads-per-worker` 同时约束 BLAS、PXR 和 Carb tasking；之前只约束 BLAS，
并行 Kit 初始化会让每个进程额外创建大线程池。SanD 在这种竞争下于场景构建末段
抛出 `std::system_error(Invalid argument)` 的现场保存在
`/DataDisk2/hsb/eval-server-audit/failed/isaac42-thread-oversubscription-20260830/`。
同一旧链路的 X-NavDP 在 LLVM/MDL 并行 JIT 阶段耗尽内存，现场保存在
`/DataDisk2/hsb/eval-server-audit/failed/isaac42-llvm-jit-oversubscription-20260830/`；
两者均在进入模型服务前失败，受限线程链路不改变任何测评数值定义。
此外，每张卡使用独立的 Kit `--portable-root`。Isaac Sim 4.2 默认把所有进程的
KVDB、MDL/LLVM JIT 和用户配置写入同一 portable root，并在并发场景构建时出现
锁竞争；按 GPU 隔离 cache 后只共享只读官方资产，不共享可变 Kit 状态。

Scene-N1 的 USD 静态闭包为 `5080` 个文件、`1,476,080,171` bytes；816 个
layer、4261 个 asset reference 的未解析项为 0。进一步的真实加载发现，1441 个
场景 MDL 使用相对 `.::OmniUe4*` 导入但资产包没有携带三个公共模块。launcher
现在一次性构建不可变的运行时符号链接覆盖层：官方资产文件只读复用，只在 cache
覆盖层补齐 `OmniUe4Base/Function/Translucent.mdl`，不修改源 USD、PLY、episode
或材质正文。

真实 Scene-N1 验证已通过 Dingo、PhysX、RGB、深度和 5 个 simulation step：
首次完整材质编译后场景创建耗时 `369.909 s`，深度张量
`[1,360,640,1]`，有限命中比例 `0.6074`，有限深度范围
`0.5261–99.8250 m`，RGB 范围 `0–235`，MDL shade node 创建失败为 0。日志：
`/DataDisk2/hsb/eval-server-prep/logs/rootless-scene-probe-20260830-r3.log`。
Acados 采用 glibc 2.18 可运行的固定 `t_renderer v0.0.34`，SHA-256
`390063f34a8e13620564b4a136012270168e1421dd7920a747048749e1d99718`；
MPC 首次生成 `2.782 s`、求解 `12.0 ms`，输出有限且控制量不超过
`0.5`。

四模型 B100 使用同一 Scene-N1、官方前 100 个 episode、`seed=1234` 和
`num-envs=10`。这是历史吞吐诊断，不与当前 B16 正式成绩混报。一个
resident evaluator 只加载一次场景，然后按 CurveNav、NavDP、SanD、X-NavDP
顺序热切换 policy server；每个模型仍从相同 episode 0 重新开始。每回合同时写
官方 `metric.csv` 和精简 NPZ 轨迹，完成后生成
`episodes.csv/summary.csv` 与跨模型
`trajectory_episodes.csv/trajectory_summary.csv`，覆盖目标进展、回退、路径
效率、曲折度、速度、规划/MPC 延迟、轨迹终点误差和离散曲率。

唯一启动命令：

```bash
cd /DataDisk2/hsb/general-navigation-benchmark-resident
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/DataDisk2/hsb/eval-server-prep/vulkan/nvidia_egl_icd.json
export NAVBENCH_CACHE_ROOT=/DataDisk2/hsb/eval-server-prep/cache/navbench-rootless
export NAVBENCH_EVAL_PYTHON=/DataDisk2/hsb/navdp/envs/isaacsim42/bin/python
export NAVBENCH_SERVER_PYTHON=/DataDisk2/hsb/eval-server-prep/bin/ubuntu22-python
export NAVBENCH_XNAVDP_ROOT=/DataDisk2/hsb/general-navigation-benchmark-resident/.runtime/x-navdp-878740a20118/baselines/x-navdp
export ACADOS_SOURCE_DIR=/DataDisk2/hsb/eval-server-prep/runtime/acados-48e223e85f04
export LD_LIBRARY_PATH=/DataDisk2/hsb/eval-server-prep/runtime/acados-48e223e85f04/lib
export NAVBENCH_EVAL_KIT_ARGS=--/rtx/verifyDriverVersion/enabled=false
/DataDisk2/hsb/navdp/envs/isaacsim42/bin/python -m navbench \
  --model curvenav --gpus 0 \
  --scene-root /DataDisk2/hsb/eval-server-audit/assets/scene-n1/n1_eval_scenes \
  --scenes home/MVUCSQAKTKJ5EAABAAAAABA8_usd \
  --episodes-per-scene 100 --num-envs 10 --cpu-threads-per-worker 8 \
  --checkpoint /DataDisk2/hsb/curvenav-f19ba8a/outputs/train_policy/checkpoint.pt \
  --model-config /DataDisk2/hsb/curvenav-f19ba8a/configs/base.yaml \
  --checkpoint-queue /DataDisk2/hsb/eval-server-audit/queues/four-model-home-b100-20260830 \
  --output-root /DataDisk2/hsb/eval-server-audit/runs/four-model-resident-b100-20260830
```

正式 4×100 在本文写入时尚未完成，因此不提前填写成绩。tmux、绝对 session
路径、每模型结果和最终轨迹分析必须在真实完成后回填本节。

2026-08-30 的旧单环境实测进一步收紧了当时 Isaac 4.2 的并发上限：
SanD 和 X-NavDP 在独立 GPU、端口和 portable root 上并行稳定；第三个 NavDP
evaluator 在场景 ready 前以 `std::system_error(Invalid argument)`/SIGABRT 退出。
当时全机只有 `2395` 个线程，可用内存 `135 GiB`，不是宿主线程或内存
上限耗尽；前两个 evaluator 继续产生完整 metric/trace。因此当前运行事实是
该 Isaac 4.2/驱动组合在本宿主上最多同时维持两个 Scene-N1 evaluator；
不通过降低 CPU8、改变模型或放宽协议来规避。NavDP 已排队到 SanD
`100/100` 并完整退出后，复用释放的 GPU0/18880 执行同一固定协议。
失败现场保存在
`/DataDisk2/hsb/eval-server-audit/failed/third-concurrent-isaac-abort-20260830/`。

## 2080 用户态常驻验证（2026-09-02）

9998（`61.52.209.211:9998`）已同步常驻分支 `6223acb` 及当前运行时契约，使用
Isaac Sim `5.0.0.0`、IsaacLab `0.46.2`、Torch `2.7.0+cu126` 和
X-NavDP `878740a2011856d0e3782dd6ccd880fd2eccd70f`。运行时校验和 20 项
contract tests 均通过。场景 overlay 会排除资产包中可能过期的
`OmniUe4Base/Function/Translucent.mdl`，统一链接当前 Isaac 运行时模块；原始官方
资产保持只读不变。

缓存分为两层：`~/.cache/navbench/official-inputs/<suite-key>` 保存原子发布的官方
输入 overlay，`~/.cache/navbench/kit/gpu_<id>` 保存 Kit/MDL/OptiX/shader cache。
常驻服务启动后场景只加载一次，后续 checkpoint 从 `--checkpoint-queue` 热切换，不
重新复制资产或重建场景。第一次 2080 冷启动实测约 5 分 46 秒完成 app/shader 初始化，
随后仍需等待该场景的 USD/材质实体创建；因此“5 分钟内”是热缓存启动目标，不是
首次冷启动保证。当前验证 session 与日志：

```text
session: curvenav-2080-resident-smoke-20260902-v5
log: /mnt/data1/huangshibo/eval-server-audit/runs/2080-resident-smoke-20260902-v5.log
cache: /home/huangshibo/.cache/navbench
```

同一 GPU0 的第二次热启动实测 app ready 为 `32.6 s`（Kit/RTX shader cache 命中），但
完整 USD/材质实体仍需继续解析；该阶段必须由 resident 进程保持一次，不能通过重复
启动获得吞吐收益。

### 主机吞吐 profile

性能参数集中在 `config/profile-2080.env` 和 `config/profile-4090.env`，路径、权重和
运行时仍由 `config/local.env` 提供。加载 profile 只设置同一个 launcher 的环境变量：

```bash
set -a
source config/local.env
source config/profile-2080.env   # 或 config/profile-4090.env
set +a
```

2080 多场景正式 profile 每个 evaluator 使用 `num-envs=16`；4090 正式 profile
同样使用 `num-envs=16`。单场景 profile 的 `num-envs=24` 只用于独立吞吐诊断。
多场景正式模式采用 4 个
resident evaluator；单场景模式只
保留一个 resident evaluator，并把策略 batch 分片到独立 GPU。

单场景 2080 可使用 `config/profile-2080-single.env`：仿真固定在 GPU0，策略
服务放到空闲 GPU，避免渲染与模型推理争用同一张卡。单场景时
`--server-gpus` 是策略分片 GPU 池；多场景时它与 `--gpus` 按 worker 一一对应。
不设置时服务仍与仿真同卡。
全套 2080 profile 根据完整场景轮换实测把并发场景数限制为 4，并将
前三个策略服务映射到剩余 GPU5--7。这样同时利用 8 张卡但不创建会耗尽
`251 GiB` 宿主内存的 5 个 Isaac 场景进程。

### 公开方案审计与当前候选

优化采用 Isaac Lab 官方建议的测量边界：冷启动与稳态分开，稳态再拆成
environment step、policy inference 和端到端 episode wall time；不同 episode
数量或 outcome 的结果不互相冒充严格 A/B。官方基准说明见
<https://isaac-sim.github.io/IsaacLab/develop/source/testing/benchmarks.html>。

原固定 runtime 的普通 `CameraCfg` 会为每个向量环境维护独立渲染产品。当前实现
改用同版本已有的 `TiledCameraCfg`，把所有前视相机合并为一次批量渲染；官方说明其
目的正是降低多相机的 render 与 host-device 开销：
<https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/sensors/camera.html>。
相机仍是同一 D455 640×360、RGB、`distance_to_image_plane`、内参、位姿和 0.05 s
更新周期，不改变模型输入定义。B8/B10/B16/B24 均已完成 shape/dtype、冷启动、
稳态 wall、显存和结果完整性门禁。

真实 NavDP 权重、B8、640×360 raw RGB/深度协议的策略池筛选结果：单卡
`0.709349 s/batch`，双卡 `0.392726 s/batch`，四卡 `0.260161 s/batch`；四卡相对
单卡为 `2.727×`，吞吐从 `11.278` 增至 `30.750 samples/s`。B8 继续拆到 7 卡后，
每卡只有 1--2 个样本，HTTP/预处理/小 batch 开销占主导，实测反而退化到
`1.368801 s/batch`、`5.845 samples/s`；五卡也只有 `0.268677 s/batch`、
`29.776 samples/s`，略慢于四卡。因此单场景 profile 选择 GPU1--4 四个策略
shard；这只是 policy 边界结果，最终仍以闭环 episode/min 决定。

NavDP 的策略分片还保持一个全局候选噪声序列：所有 shard 按全局 batch 大小生成
同一序列，只消费各自连续区间，避免每张卡重复使用头部候选。实测单卡 B8 与四卡
B2×4 即使候选噪声完全一致，扩散迭代和 critic 仍会放大不同 batch shape 的 CUDA
数值差异，三步中最终选中轨迹的最大绝对差为 `0.285/0.543/0.337 m`。因此该模式
说明 NavDP 的批量执行数值不同于上游单环境执行。当前项目正式成绩统一采用 B16，
不得和上游论文数值声称逐位等价。其余模型在各自随机性和批处理语义完成审计前，
launcher 会拒绝单场景多 policy shard。

同一 Home 场景、NavDP、seed 1234、CPU6、TiledCamera、四策略卡的闭环筛选如下。
冷启动均约 19 分钟，表中只统计 `ready` 到 `done` 的稳态；SR/SPL 仅为诊断，不能
跨不同 batch shape 比较模型效果。

| 配置 | 完成/墙钟 | episode/min | aggregate sim-s/wall-s | GPU0 mean/peak VRAM | policy mean util | 规划中位延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 Camera B8 K1 | 8/353.08 s | 1.360 | 1.33 | 约 9.5 GiB | 约 80%（1卡瞬时） | 824.8 ms |
| Tiled B8 K4 | 8/300.76 s | 1.596 | 1.56 | 73.6% / 9.43 GiB | 47.5% | 363.4 ms |
| Tiled B10 K4 | 10/346.08 s | 1.734 | 1.49 | 72.6% / 9.71 GiB | 39.3% | 514.3 ms |
| Tiled B16 K4 | 16/499.84 s | 1.921 | **2.50** | 81.0% / 10.08 GiB | 43.5% | 762.0 ms |
| Tiled B24 K4 | 24/672.16 s | **2.142** | 2.42 | 84.4% / 10.28 GiB | 44.8% | 1045.2 ms |

B16 是单位仿真工作的效率拐点；B24 在约 0.7 GiB 显存余量下取得最大实际
episode/min，因此 `profile-2080-single.env` 唯一固定为 B24/K4。相对旧 B8/K1，
墙钟 episode 吞吐提高 `57.5%`。B32 不测试：收益已趋缓且会把 11 GiB 显存余量
压到不可接受范围。GPU5--7 不继续切策略，因为 K7 微基准已退化；单 resident 的
正确最优是五张有效 GPU，而不是用无效小 batch 人为填满八张卡。

### 2080 全 40 场景最快配置

全套任务声明 GPU0--4，但以 `max-workers=4` 同时维持4个 resident evaluator；
策略服务映射为 `5,6,7,5,6`，实际活动 worker 使用对应前4项。B16 的 5
场景校准中，仿真卡稳态利用率为约 82--92%，策略卡为约 84--100%；最重仿真卡达到
10.77/11.26 GiB。该短校准最低 available 约27 GiB，但正式场景轮换时5 worker
达到238/251 GiB且四个额外 CUDA context 使复杂场景只剩约14 MiB，随后在16 MiB
分配处 OOM。因此5 worker不再属于安全配置；B24也不进入全套 profile。

同一组 5 个 Home 场景的整机校准结果如下。W4/B8 完成 40 回合的总墙钟约
48.8--51.2 分钟；W5/B16 完成 80 回合从 launcher 启动到汇总共 33.77 分钟，含
冷启动吞吐为 2.37 episode/min，无 OOM、Isaac、HTTP 或模型错误。由于两次执行的
episode 数和 outcome 不同，这只是整机容量/吞吐诊断，不是模型成绩 A/B。按五个
场景的 startup（997--1362 s）和稳态（497--680 s/16 episodes）外推，40 场景各
100 回合约 10.7 小时；Commercial 首次缓存和场景差异纳入后，运行预算取 11--12
小时。

上述 W5 数据保留为历史容量诊断，不能再外推正式总耗时。正式 profile 的 W4
吞吐和总耗时以本次完整续跑重新统计。

40 场景输入缓存采用两层唯一合同：`official-inputs/<suite-split-key>` 原子建立
Home+Commercial 的只读 symlink overlay；`kit/gpu_0` 到 `kit/gpu_4` 分别保存
Kit/MDL/OptiX/PhysX 运行缓存。40 个已实例化场景不能同时常驻：按约 38 GiB/场景
需要约 1.5 TiB 主存。正确做法是 4 个场景常驻并由公共队列流水换入；同一模型的
多个 checkpoint 在场景卸载前连续执行，从而只支付一次场景构建。首次完整运行会
自然填充真实命中的缓存，不做耗时且无收益证据的 40×5 重复预热。

40 场景、每场景 100 回合的唯一正式成绩命令如下；省略 `--scenes` 和
`--episodes-per-scene`，直接使用冻结 suite 的 40×100 合同：

```bash
cd /mnt/data1/huangshibo/H/general-navigation-benchmark-resident-04984bd
set -a
source config/local.env
source config/profile-2080.env
set +a
export OMNI_KIT_ACCEPT_EULA=YES
$NAVBENCH_EVAL_PYTHON -m navbench --model navdp \
  --checkpoint /mnt/data1/huangshibo/H/NavDP/checkpoints/navdp/navdp_pretrain.ckpt \
  --output-root /mnt/data1/huangshibo/eval-server-audit/runs/navdp-40x100-b16
```

固定多模型正式测评使用同一个 launcher，并通过不可变 artifact bundle 声明其余
模型。每个 worker 加载一个场景后，按声明顺序完成全部模型，再卸载并领取下一场景；
因此 SanD、NavDP、X-NavDP 的 40 场景总共只构建 40 次场景，而不是 120 次：

```bash
$NAVBENCH_EVAL_PYTHON -m navbench \
  --model sandplanner \
  --checkpoint /mnt/data1/huangshibo/H/NavDP/checkpoints/sandplanner/NoMax.pth \
  --artifact-bundle /mnt/data1/huangshibo/eval-server-audit/artifacts/navdp.ready \
  --artifact-bundle /mnt/data1/huangshibo/eval-server-audit/artifacts/x-navdp.ready \
  --output-root /mnt/data1/huangshibo/eval-server-audit/runs/sand-navdp-xnavdp-40x100-b16
```

每个 bundle 只含 `artifact.json` 和指向冻结权重的 `checkpoint.pt` 符号链接；它不
复制权重、不保存运行时状态，也不引入第二套 adapter 或 evaluator 链路。三组结果
分别写入同一 session 的 `models/sandplanner`、`models/navdp` 和 `models/x-navdp`。

正式启动前的 B16 服务预检发现并修正了 SanD adapter 的批语义错误：上游
`process_depth_arrays` 是“一个观测生成一批候选”，旧 wrapper 却把16个独立环境
当成候选 batch 后只读取第0个环境，返回 `(1,N,3)`。当前 wrapper 为每个环境维护
独立四帧缓存和 warm-start 状态，逐环境调用上游候选生成与 ESDF 选择，最后仅在
HTTP 边界补齐轨迹点数并组成 `(16,N,3)`；不复制模型、不共享 episode 状态、不把
一条轨迹广播给其他环境。真实 B16 预检结果：SanD `(16,54,3)`、NavDP
`(16,24,3)`、X-NavDP `(16,24,3)`，均为有限 `float32`；SanD 16个不同横向目标
产生16个不同终点。SanD 热步约 `15.49 s/B16`，这是其每个观测仍需独立生成并
评价候选集的真实计算成本，不以减少候选或共享轨迹换取虚假吞吐。

首轮固定三模型运行在5个 SanD 场景和部分 NavDP/X-NavDP 完成后暴露了策略卡
内存调度缺口：profile 将两个 worker 映射到同一张11GB卡，两个 X-NavDP server
重叠时第二个实例 OOM。根修不是捕获 OOM 重试，而是统一的加权策略 GPU 容量池：
每张2080 Ti策略卡有2个显存 slot，SanD/NavDP 各占1，X-NavDP 占2。因此两套
轻量 server 仍可并发，X-NavDP 在其策略卡上从启动到退出均独占；等待期间对应
Isaac 场景继续常驻。该调度不改变模型、episode、观测或指标，并由同一个 launcher
在首次启动和 `--resume-root` 续跑时共同执行。

该命令同时是当前项目正式成绩协议。批量形状引入的数值差异必须记录，因此结果只
与同一 B16 协议下的其他模型比较，不伪装成 X-NavDP 上游执行的逐值复现。

单场景吞吐诊断的唯一启动形式：

```bash
set -a
source config/local.env
source config/profile-2080-single.env
set +a
$NAVBENCH_EVAL_PYTHON -m navbench --model navdp \
  --scenes home/MVUCSQAKTKJ5EAABAAAAABA8_usd \
  --episodes-per-scene 100 \
  --checkpoint /mnt/data1/huangshibo/H/NavDP/checkpoints/navdp/navdp_pretrain.ckpt \
  --output-root /mnt/data1/huangshibo/eval-server-audit/runs/navdp-home-b100
```

未采用的公开选项也有明确原因：跳帧、降低分辨率、关闭 RGB/depth 或异步纹理流会
改变模型观测；合并/flatten 官方 USD 会改变资产 provenance；Fabric、stage-in-memory
在当前全局 USD 场景和相机链上尚无等价性证据。因此这些均不进入固定测评链路。

唯一常驻启动形式（把 `--checkpoint` 指向首个模型，后续 bundle 原子发布为
`<queue>/*.ready`）：

```bash
cd /mnt/data1/huangshibo/H/general-navigation-benchmark-resident-04984bd
export OMNI_KIT_ACCEPT_EULA=YES
export NAVBENCH_CACHE_ROOT=/home/huangshibo/.cache/navbench
export NAVBENCH_EVAL_PYTHON=/mnt/data1/huangshibo/H/general-navigation-benchmark/.runtime/envs/xnavdp-eval/bin/python
export NAVBENCH_SERVER_PYTHON=/mnt/data1/huangshibo/.venvs/navbench-server/bin/python
export NAVBENCH_XNAVDP_ROOT=/mnt/data1/huangshibo/H/general-navigation-benchmark/.runtime/x-navdp-878740a20118/baselines/x-navdp
export ACADOS_SOURCE_DIR=/mnt/data1/huangshibo/H/general-navigation-benchmark/.runtime/acados-48e223e85f04
export NAVBENCH_GPUS=0
$NAVBENCH_EVAL_PYTHON -m navbench --model navdp --gpus 0 \
  --scenes home/MVUCSQAKTKJ5EAABAAAAABA8_usd --episodes-per-scene 1 \
  --num-envs 1 --checkpoint-queue /mnt/data1/huangshibo/H/eval-queues/2080-resident \
  --output-root /mnt/data1/huangshibo/eval-server-audit/runs/2080-resident
```
