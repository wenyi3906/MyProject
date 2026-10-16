# CloudEdgeSim：Python 云边协同负载均衡仿真平台

用于比较 GA、ACO、SA 与随机、轮询、预测最早完成策略。支持静态已知任务集、动态到达、突发流量、计算/网络瓶颈和异构资源，提供可重放轨迹、批量实验、断点续跑、统计表及 PNG/SVG 图表。

## 快速开始（Windows PowerShell）

进入项目根目录后执行：

```powershell
Set-Location '<项目所在目录>\MyProject'
& '.venv\Scripts\python.exe' -m cloudedge --help
& '.venv\Scripts\python.exe' -m cloudedge suite --manifest configs/smoke_suite.yaml --out outputs/smoke
```

轻量套件包含静态和动态两种模式、六种策略、三个种子，共 **36 次运行**。再次执行同一命令会检查文件校验值并跳过完整且参数一致的结果。

主要结果：

- `outputs/smoke/analysis/report.md`：实验报告。
- `outputs/smoke/analysis/runs.csv`：每次运行的指标。
- `outputs/smoke/analysis/statistics.csv`：均值与 95% bootstrap 区间。
- `outputs/smoke/analysis/paired_differences.csv`：按相同任务轨迹种子配对的算法差值。
- `outputs/smoke/analysis/plots/`：各场景指标对比、共享轨迹的收敛对比。
- `outputs/smoke/runs/<场景>/seed_<种子>/<算法>/`：任务、决策、收敛、资源时序、配置和复现信息。

## 在新机器构建运行环境

Python 3.12 是已验证的运行版本。运行依赖和测试工具的精确版本保存在 `requirements.lock`。

> 不要把其他机器或其他目录中的 `.venv` 直接复制过来。Python 虚拟环境会记录创建它时所用解释器的绝对路径，移动后可能仍然尝试从旧机器的盘符或用户目录加载 Python。请在每台新机器上重新创建环境。

### 方案一：在项目的 `.venv` 目录中安装环境（推荐用于项目归档和打包）

该方案把 Python 虚拟环境及依赖集中放在项目根目录的 `.venv` 中，不会把依赖安装到全局 Python。新机器需要先安装 **Python 3.12**，并确保 Windows Python Launcher `py` 可用。

在 PowerShell 中进入项目根目录，然后执行：

```powershell
# 确认 Python 3.12 已安装
py -3.12 --version

# 如果存在从其他机器复制来的旧环境，先将它改名备份
if (Test-Path '.venv') { Rename-Item '.venv' '.venv.old' }

# 使用本机 Python 重新创建项目内虚拟环境
py -3.12 -m venv .venv
& '.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.venv\Scripts\python.exe' -m pip install -r requirements.lock
& '.venv\Scripts\python.exe' -m pip install --no-build-isolation --no-deps -e .
& '.venv\Scripts\python.exe' -m pytest -q
```

验证无误后可以删除备份的 `.venv.old`。以后无需激活环境，直接通过项目内解释器运行即可：

```powershell
& '.venv\Scripts\python.exe' -m cloudedge --help
```

归档项目时可以将 `.venv` 一并保留，方便在**同一台机器和同一路径**下继续使用；若压缩包需要迁移到另一台机器，建议排除 `.venv`，并按上述步骤根据 `requirements.lock` 重建，以避免绝对路径和平台差异导致环境失效。

### 方案二：使用本机 Conda 创建独立虚拟环境

该方案由 Conda 统一管理 Python 和依赖，项目目录中不会创建 `.venv`。新机器需要先安装 Miniconda、Anaconda 或其他兼容的 Conda 发行版。

```powershell
# 创建名为 cloudedge 的 Python 3.12 环境
conda create -n cloudedge python=3.12 pip -y

# 激活环境
conda activate cloudedge

# 确认当前 Python 来自 cloudedge 环境
python --version
Get-Command python

# 安装锁定依赖和当前项目
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .

# 运行测试
python -m pytest -q
```

以后使用项目前先执行：

```powershell
conda activate cloudedge
python -m cloudedge --help
```

如果希望使用其他环境名称，可将命令中的 `cloudedge` 替换为自定义名称。使用 Conda 方案时，不要再调用 `.venv\Scripts\python.exe`，否则实际使用的是项目内 `.venv`，而不是当前激活的 Conda 环境。

其他系统可在方案一中使用 `python3.12 -m venv .venv` 和 `.venv/bin/python`。Python API 位于 `src/cloudedge/`；安装后也可以使用 `cloudedge` 命令。

## 运行单个实验

```powershell
& '.venv\Scripts\python.exe' -m cloudedge run --config configs/smoke.yaml --algorithm ga --seed 0 --out outputs/single
```

`--algorithm` 可选 `random`、`round_robin`、`greedy`、`ga`、`aco`、`sa`。结果位于 `outputs/single/run/`。使用 `--no-resume` 强制重新运行。

分别生成轨迹与执行，便于共享实验输入：

```powershell
& '.venv\Scripts\python.exe' -m cloudedge generate --config configs/smoke.yaml --seed 0 --out outputs/shared_trace.jsonl
& '.venv\Scripts\python.exe' -m cloudedge run --config configs/smoke.yaml --algorithm sa --seed 0 --trace outputs/shared_trace.jsonl --out outputs/shared_sa
```

重放已保存的配置、算法种子和原始轨迹：

```powershell
& '.venv\Scripts\python.exe' -m cloudedge replay --run outputs/single/run --out outputs/replay
```

实际搜索耗时会随机器负载变化；调度结果、任务阶段时间、目标值应保持一致。重放使用当前安装的代码，原始代码指纹和依赖版本在 `metadata.json` 中。

## 批量比较与正式研究

```powershell
# 先用少量开发种子比较一个正式场景
& '.venv\Scripts\python.exe' -m cloudedge compare --config configs/base.yaml --seeds 0 1 --algorithms greedy ga sa --out outputs/pilot

# 主实验：12 个场景 × 30 个种子 × 6 个策略 = 2160 次运行
& '.venv\Scripts\python.exe' -m cloudedge suite --manifest configs/benchmark_suite.yaml --out outputs/benchmark --workers 1

# 窗口、评价预算、目标权重，各自单因素扫描
& '.venv\Scripts\python.exe' -m cloudedge suite --manifest configs/sensitivity_suite.yaml --out outputs/sensitivity

# 仅重新计算统计和图表，不重新仿真
& '.venv\Scripts\python.exe' -m cloudedge analyze --out outputs/benchmark
```

正式配置为 300 秒动态仿真、30 秒预热、种子 100～129；开发调参使用种子 0～9。`--seeds` 与 `--algorithms` 覆盖本次运行选择。

完整候选评估使用精确事件回放，运行成本随队列长度、批量大小和评价预算增长。先通过 `compare` 运行小规模试验，测量耗时后再启动正式套件。`--workers 4` 可并行运行独立实验；比较算法实际计算耗时时采用 `--workers 1` 并保持机器负载条件一致。

批次完成后才写入 `_SUCCESS.json`；中断后重跑同一命令即可恢复。配置、代码指纹、依赖或任务轨迹变化会触发重算。已有结果损坏也会重算。批量运行按完整运行恢复，不恢复单次仿真内部的搜索状态。

## 小规模最优解验证

```powershell
& '.venv\Scripts\python.exe' -m cloudedge exhaustive --config configs/tiny.yaml --seed 0 --out outputs/exhaustive
```

该算例有 6 个任务和 2 个 VM，枚举全部 64 种分配。`optimality_gaps.csv` 输出精确最优值及各算法的绝对差距。默认最多枚举 1,000,000 种分配，避免误运行大规模指数搜索。

## 配置与模型

配置支持相对路径 `extends` 和递归覆盖，例如：

```yaml
extends: base.yaml
name: my_experiment
simulation: {duration: 30, warmup: 3, window: 0.1}
workload: {devices: 100}
optimization: {dynamic_budget: 50, weights: [0.5, 0.3, 0.2]}
experiment: {seeds: [0, 1, 2]}
```

场景名仅使用英文字母、数字、下划线和连字符。数据单位为秒、字节、MI、MIPS、Mbps，报告中的 KB 按 1000 字节转换。负值、非有限值、重复种子和非法权重会在运行前报错。

| 配置 | 用途 |
|---|---|
| `base.yaml` | 默认动态场景，32 个边缘 VM、4 个云 VM、8 个接入点 |
| `static.yaml` | 预知整个有限任务集，一次性优化；释放时间均匀分布在 30 秒内 |
| `burst.yaml` | 第 120～150 秒到达率提升 3 倍 |
| `compute_bottleneck.yaml` | 云和边缘算力缩减至 25%，提高网络带宽 |
| `network_bottleneck.yaml` | WLAN 50 Mbps、WAN 100 Mbps |
| `heterogeneous.yaml` | 边缘 VM 算力 5000/10000/15000，总容量不变 |
| `smoke.yaml` / `smoke_static.yaml` | 快速验证场景，评价预算 50，三个开发种子 |
| `tiny.yaml` | 可穷举的最优解验证算例 |

模型假设、评价口径和边界处理详见 [docs/MODEL.md](docs/MODEL.md)。

## 软件结构

```text
src/cloudedge/
  models.py       实体、快照、评价与运行结果类型
  config.py       配置读取、默认值和校验
  workload.py     可复现静态/动态轨迹、分段泊松突发流量
  engine.py       SimPy 内核、FIFO 资源、超时、快照与统一评估
  algorithms.py   三个优化算法和三个基础策略
  simulation.py   静态/动态调度控制与指标计算
  experiments.py  批量执行、并行、校验与断点续跑
  analysis.py     bootstrap、配对统计和绘图
  cli.py          命令行与穷举验证
configs/          基准、轻量、突发、瓶颈、异构与敏感性配置
tests/            手算、资源竞争、取消、回放、预算、统计和端到端测试
```

后续算法实现同一个 `Scheduler.solve` 接口，在算法列表及配置校验中注册名称，便可复用 `Evaluator` 和实验流程。原报告的 GRASP-GA、三窗口流水线、CPU/带宽并行共享、移动性、故障、迁移和能耗留作下一阶段。
