# CurveNav 仿真场景方案

## 结论

当前首批闭环采用 **HSSD static scene + Habitat-Sim**。这条链能在现有 V100S
直接完成 navmesh、专家路径、D455 几何一致的深度渲染和 SanD run 落盘，避免
等待 RTX/Isaac worker；场景和家具均来自开源资产，不使用自造栅格作为训练场景。

Scene-N1 的 InternScenes home/commercial official-train 保留为第二阶段近域扩展。
它与 NavDP/X-NavDP 的 Isaac/Usd 域更接近，但当前 V100S 无 RT Core，不能在本机
完成可信的 Isaac 深度采集。不能使用 `home_eval`、`commercial_eval`、
`cluttered_easy` 或 `cluttered_hard` 生成训练数据。

## 来源选择

| 优先级 | 数据源 | 规模/格式 | 用途 | 约束 |
|---|---|---|---|---|
| P0 | [HSSD](https://huggingface.co/datasets/hssd/hssd-hab) | 211 个带家具 Habitat 场景 | 当前 V100 端到端主链 | static-only；按场景依赖下载，建筑级切分 |
| P1 | [Scene-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1) | 99 个 home/commercial，原生 USD | NavDP 近域扩展/验证 | gated；只允许 official train；采集需要 RTX worker |
| P2 | [InteriorAgent](https://huggingface.co/datasets/spatialverse/InteriorAgent) | 25 个原生 USD/USDA，约 9.94 GB | 建筑与材质远域验证 | 非商业研究；采集需要 RTX worker |
| P3 | [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/) | 84 个布置变化、6 个基础布局，约 155 MB | 传感器冒烟测试 | 基础布局太少，不计入泛化场景数 |

HM3D 的 1,000 个真实扫描适合做远域验证，但扫描孔洞、非物理网格和 GLB→USD
转换成本使它不适合作为第一批闭环专家数据。iGibson 资产被限制在 iGibson 内
使用，因此不进入 Isaac 主链。ProcTHOR/MolmoSpaces 可在后续规模实验中单独
评估，当前不与原生 Scene-N1 链路混合。

## 下载与隔离

HSSD 下载器锁定仓库 revision，只拉取所选 scene instance、stage、semantic 和
精确引用的家具资产，不下载完整 12 GB 仓库：

```bash
python scripts/download_hssd_scenes.py data/scene_assets/hssd-hab 102344280
```

下载结果在资产根目录生成 `download_manifest.json`。正式扩批时先离线统计
navmesh 面积、连通率和布局类型，再按建筑划分训练/验证；同一建筑或布置变体
不能跨 split。

Scene-N1 需要先在页面接受数据协议并提供只读 Hugging Face token：

```bash
HF_TOKEN=... scripts/download_scene_n1_archives.sh data/scene_assets/scene_n1
```

脚本固定到 `2195d46aaab0ff48673b275fdfdc0731075b5ff2`，只下载
home/commercial 及其共享材质依赖，不下载 clutter benchmark。由于上游把多个
场景打在共享归档中，下载资产不等于授权训练；训练场景仍必须经过现有
`scene_split.json` 和 `discover_training_scenes()` 白名单。

归档解压和 Isaac 验收必须在拿到实际文件后进行，不能猜测归档内部目录。每个
场景进入采集队列前至少满足：USD 引用完整、单位/朝向正确、静态碰撞有效、
深度无大面积空洞、机器人可生成 navmesh/occupancy、official split 无泄漏。

## 执行机器

当前服务器是 8 张 Tesla V100S。Habitat-Sim 0.3.3 headless 已验证可在 V100S
通过 EGL 渲染 HSSD depth，因此当前机器能够承担完整 HSSD static-scene 链。
V100 仍不满足 Isaac Sim 的官方 RTX 要求；Scene-N1/InteriorAgent 的 Isaac
验收和采集必须调度到带 RT Core 的 worker。
