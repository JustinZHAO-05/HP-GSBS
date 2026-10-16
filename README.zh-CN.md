<div align="center">

![HP-GSBS：多通道神经信号经过三条计算路线，得到一致的时间状态边界](docs/assets/hpgsbs-hero.png)

# HP-GSBS

**高性能 Greedy State Boundary Search**

更快地寻找神经活动的时间状态，保留可审查的分段决策。

[![CPU tests](https://github.com/JustinZHAO-05/HP-GSBS/actions/workflows/tests.yml/badge.svg)](https://github.com/JustinZHAO-05/HP-GSBS/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![MIT](https://img.shields.io/badge/License-MIT-22b8a8)](LICENSE)

[English](README.md) · [下载安装](#下载安装) · [性能实测](#性能实测) · [一致性](#与原始-gsbs-的一致性) · [Release 下载](https://github.com/JustinZHAO-05/HP-GSBS/releases)

</div>

## 用来做什么？

GSBS 将 **时间点 × 特征** 矩阵划分为连续状态，每个状态内具有相对稳定的空间活动模式。HP-GSBS 面向全通道 EEG、ROI 和大量独立记录的分段计算，在保持原始搜索语义的前提下减少重复计算。项目以 BBBD 教育视频 EEG 为开发场景，也可接收其他已正确预处理的多变量信号。

输入是你已有的矩阵和研究参数；输出是边界、状态、完整 t-distance 曲线、空间模式与边界强度。预处理、artifact 规则、K 选择与统计分析继续由原研究控制。源空间网络应使用原定义的多个 parcel/source features；单条平均信号 P=1 不满足空间模式相关的定义。

| 路线 | 核心实现 | 本机适用场景 |
|---|---|---|
| **CPU FastGSBS** · `cpu-fast` | NumPy/BLAS + Numba + 记录间多进程 | 通用运行、大量短记录 |
| **TensorGSBS** · `torch-cuda` | PyTorch CUDA 批量预计算 + CPU 顺序控制 | 无需 Triton 的 NVIDIA GPU 拟合 |
| **TurboGSBS** · `triton-cuda` | 实际执行的 Triton 区间评分 kernel + 共同控制器 | 多种规模低延迟，需按任务实测 |
| **官方对照** · `reference` | 直接调用未经修改的 `statesegmentation.GSBS` | 基准与逐轮验证 |

GPU 路线目前是**混合实现**：Q、相关矩阵与统计前缀在 GPU 预计算；有顺序依赖的搜索和微调在 CPU 执行。并未宣称完整控制流已经常驻 GPU，也不假定 Triton 总比 PyTorch 快。

## 性能实测

**五类负载均已补齐原版 FP64 实测基线。** 下表每个加速比使用同一输入、参数、环境和精度；不同负载的加速幅度分别报告，不将 Small 倍率外推。

测试机器：Ryzen 7 7840H，RTX 4050 Laptop 6 GB，WSL2 Ubuntu 22.04；PyTorch 2.11.0+cu128、Triton 3.6.0。合成负载为 **预热后三次运行中位数**；BBBD0 为 **单次实测，无重复**，CUDA 已同步，CPU 单进程/单 BLAS 线程。全部开启 statewise detection、`finetune=1`、strength-order、`blocksize=50`，精度 FP64，关闭 TF32。

| 工作负载 | T × P | Kmax | 官方 GSBS | CPU FastGSBS | TensorGSBS | TurboGSBS |
|---|---|---|---|---|---|---|
| Small · 合成 | 450 × 16 | 20 | 48.035 s | 0.169 s | 0.110 s | 0.098 s |
| Medium · 合成 | 900 × 32 | 30 | 349.946 s | 0.455 s | 0.276 s | 0.328 s |
| Large · 合成 | 1500 × 64 | 40 | 2213.493 s | 1.659 s | 0.673 s | 0.515 s |
| XL · 合成 | 2100 × 64 | 50 | 5011.254 s | 3.778 s | 2.314 s | 2.229 s |
| BBBD0 · 真实 EEG（单次） | 1336 × 64 | 131 | 6115.379 s | 4.489 s | 3.738 s | 3.421 s |

相对官方 GSBS 的加速倍数：

| 工作负载 | CPU | PyTorch | Triton |
|---|---|---|---|
| Small | 285.0× | 438.4× | 489.1× |
| Medium | 768.5× | 1265.7× | 1066.6× |
| Large | 1334.1× | 3287.4× | 4300.9× |
| XL | 1326.5× | 2165.6× | 2248.1× |
| BBBD0 | 1362.3× | 1636.2× | 1787.4× |

Small 保留 v0.1.0 的原始同环境测量。Medium、Large、XL 在同一 WSL 进程重新成对测量：一轮预热后，三轮依次运行原版/CPU/PyTorch/Triton，报告三次中位数。BBBD0 按用户要求每个后端仅运行一次，倍率来自单次实测，不能估计重复波动；原始日志中的 warmup 标签对应这一保留的单次运行。合成负载 dmin=1，真实 EEG dmin=5；计时关闭 trace，排除保存与一致性核对，未并发运行其他 GSBS 拟合。此前带 trace 的 Windows 全记录用时不参与倍率计算。

**32 条 Small 独立记录的吞吐：**CPU 8 workers 为 1.006 s、约 **1,908 fits/min**；PyTorch B=32 为 1.684 s、1,140 fits/min；Triton B=32 为 1.909 s、1,006 fits/min。大量短任务在本机由 CPU 多进程胜出。

**真实冷启动：**新进程、空 JIT 缓存、含导入和初始化，分别为 CPU 3.44 s / PyTorch 7.96 s / Triton 10.55 s。预热耗时不能预测第一次运行。测量来自正常桌面/WSL 环境，存在后台任务资源竞争。

→ [完整性能、显存与 profiler](docs/performance.md) · [原始单条测量](results/benchmarks/delivery_latency.json) · [CPU 吞吐记录](results/benchmarks/batch_cpu8.json)

## 与原始 GSBS 的一致性

加速来自数学复用，**没有降低 Kmax、降采样、关闭双边界或关闭微调**。保留官方 block 分组、同一旧状态内双边界、单/双候选 t-distance 决策、顺序微调和边界编号、`ddof=1` 空间标准化、`dmin`、可能到达 `kmax+1` 的索引行为。支持 `y`：搜索用 `x`，t-distance 用 `y`。

| 证据 | 结果 |
|---|---|
| 本次配对基线：3 个合成输入 + 1 个 EEG 输入，共 52 次保存的拟合 | 每个 K 的边界及最终 K 一致；相对 CPU 最大 t-distance 误差 **2.44×10⁻¹³**，patterns/strengths 通过；计时拟合不采集候选 trace。 |
| 原始 Windows 本机回归 | 103 通过，1 个 GPU 模块跳过 |
| 原始 WSL 含 CUDA/Triton 回归 | 134 通过 |
| 4 条完整 EEG，覆盖 full64/frontal32、两个 session、四个视频 | CPU 每个 K 的 `all_bounds` 与历史官方一致，最大 t 误差 **7.46×10⁻¹³** |
| 新跑完整官方 reference：1336×64、Kmax=131 | 三条 FP64 路线逐轮候选、选择及微调路径全部通过；最大 t 误差 **2.44×10⁻¹³** |
| 两条 GPU FP64 对照已验证 CPU 全轨迹 | 每个后端 **4/4** 通过 |
| 两条 GPU FP32 同组测试 | 边界均一致，但每个后端 **0/4** 通过严格数值容差 |

测试核对每轮候选得分、单/双 t 值、选择、微调前后、所有 K 边界、最终 K、patterns 与 strengths。初始容差 `atol=rtol=1e-8`；FP32 的 t 误差约 `4e-5`–`3.54e-4`，因此**默认科研精度为 FP64，FP32 仅实验使用**，不提供 TF32 拟合模式。

近平局按原运算顺序重算；病态近常数空间模式可整条回退官方，并记录真实 backend 和原因。P=1、非有限输入、空间常数行明确拒绝。上述是已测试范围内的证据，不是所有浮点输入上的普遍等价证明。合成信号的真实边界恢复率与算法 parity 分开统计。

公共仓库不分发真实 EEG 或逐记录拟合结果。公共 CI 可独立运行合成测试，缺少私有 fixture 与可选 fork 时相关测试跳过；测试用例数不是参与者数。**尚未验证源空间七网络，也未完成数千条生产压力测试。**

→ [一致性详表](docs/reference_parity.md) · [比较记录](results/benchmarks/saved_validation.json) · [已知限制](docs/known_limitations.md)

## 下载安装

需要 **Python 3.10+**，已有验证使用 3.10/3.11。CPU 基础安装不会拉入 CUDA/PyTorch/Triton。

```bash
git clone https://github.com/JustinZHAO-05/HP-GSBS.git
cd HP-GSBS
python -m venv .venv
```

Linux/WSL 使用 `source .venv/bin/activate`；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，或直接调用 `.venv\Scripts\python.exe`。

```bash
python -m pip install -e ".[test]"
python -m hpgsbs doctor
python examples/quickstart.py
```

示例生成 80×8 合成输入，拟合优化版与官方版，验证完整 trace，并保存 `examples/generated/input.npy` 和 `fit.npz`。不需要任何私有数据。重复执行请改用新的 `--output` 目录，避免覆盖结果。

也可从 GitHub **Code → Download ZIP** 下载源码，解压后安装，或在 [Releases](https://github.com/JustinZHAO-05/HP-GSBS/releases) 下载 wheel。当前未发布至 PyPI，不能假设 `pip install hpgsbs` 会安装本项目。

### Python API

```python
import numpy as np
from hpgsbs import GSBSConfig, fit, GSBSResult

x = np.load("examples/generated/input.npy")  # time × features
config = GSBSConfig(kmax=12, dmin=1, dtype="float64")
# statewise_detection=True, finetune=1, finetune_order=True, blocksize=50 为默认值
result = fit(x, config, backend="cpu-fast")
print(result.nstates, np.flatnonzero(result.deltas))
boundaries = result.get_bounds(6)
full_curve = result.tdists  # 继续交给原研究的 K 选择函数
result.save("my_fit.npz")  # 原子写入，默认拒绝覆盖
restored = GSBSResult.load("my_fit.npz")
```

支持原版 `all_bounds/bounds/deltas/states/tdists/nstates/state_patterns/strengths`，并记录 backend、精度、参数、用时与上游来源。`save(..., times=原时间轴)` 保存边界 timestamps；`to_statesegmentation_compatible()` 返回兼容对象。跨验证使用 `fit(x, config, backend="cpu-fast", y=y)`。

### GPU 可选依赖

按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 匹配驱动与 CUDA wheel。本次已测试的 Linux/WSL 组合为：

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install triton==3.6.0
python -m hpgsbs doctor
python examples/quickstart.py --backend torch-cuda --output examples/generated/torch
python examples/quickstart.py --backend triton-cuda --output examples/generated/triton
```

PyTorch 路线不强制需要 Triton。Windows CPU 可原生运行；Triton 推荐使用已测试的 WSL2/Linux，参见 [官方 Triton 安装](https://triton-lang.org/main/getting-started/installation.html)。CUDA/Triton 不可用时明确报错，可手动选择 `cpu-fast`；未实现 AMD/macOS GPU 后端。

| 依赖 | 安装要求 |
|---|---|
| NumPy、SciPy、Numba | `numpy>=1.24`、`scipy>=1.14`、`numba>=0.59` |
| 官方 reference | 固定 `statesegmentation==0.0.6` |
| 线程/环境工具 | `threadpoolctl>=3`、`psutil>=5`、`tqdm>=4` |
| GPU 可选 | CUDA 版 PyTorch；Triton 后端再安装 Linux Triton |
| 测试可选 | `pytest>=8`、`joblib>=1.3`，通过 `[test]` 安装 |

完整约束见 [pyproject.toml](pyproject.toml)，实测版本见 [env/](env/)。最低依赖约束不等于所有版本组合都经过验证。

## 批量、恢复与 CLI

`fit_many(recordings, config, backend, batch_size="auto")` 支持逐记录配置、精确 T/P 分桶、独立分段路径和 OOM 缩批。`mode="latency"` 自动 GPU B=1；`mode="throughput"` 根据可用显存估计，显式 batch size 优先。CPU 用 `n_jobs` 控制记录间并行，并限制每个进程的 BLAS 线程，避免过度并行。

磁盘任务使用 [examples/tasks.json](examples/tasks.json)，相对路径以运行目录为准。每条任务完成立即保存；重跑验证后跳过，失败单独记录。Windows 多进程 API 调用应放在脚本的 `if __name__ == "__main__":` 内。

```bash
python -m hpgsbs fit examples/generated/input.npy --kmax 12 --output another_fit.npz
python -m hpgsbs fit-batch examples/tasks.json --backend cpu-fast --workers 4 --output examples/generated/batch
python -m hpgsbs validate examples/generated/input.npy --kmax 12 --backend cpu-fast
python -m hpgsbs bench examples/generated/input.npy --kmax 12 --backend cpu-fast --repeats 3
python -m hpgsbs compare examples/generated/fit.npz another_fit.npz
```

单条命令支持 `--config/--dtype/--y`，`fit` 还支持 `--times`。批量支持 `--batch-size auto` 或指定整数。

## 加速算法

原始 GSBS 为大量候选反复构造状态均值及标准化矩阵。HP-GSBS 用区间统计复用这些计算。

设 Z 是每个时间点按空间特征做 `ddof=1` 标准化，A(t) 是空间去均值后 X 的前缀和，B(t) 是 Z 的前缀和。非退化区间 `[a,b)` 对全局匹配目标的贡献为：

$$
F(a,b)=\frac{[B(b)-B(a)]^\top[A(b)-A(a)]}{T\sqrt{P-1}\,\|A(b)-A(a)\|_2}.
$$

1. **预计算 Q：**`Q[a,b]=F(a,b)`；候选从 O(TP) 的重构变为 O(1) 查表。
2. **单/双边界评分：**分别组合两个/三个连续区间贡献，仍按官方 block 搜索候选。最终单/双选择继续比较 t-distance。
3. **快速 t-distance：**对 `R=ZZᵀ/(P−1)` 应用官方 `dmin` 上三角掩码，构建 count/sum/squared-sum 二维前缀。每个状态及相邻状态的样本矩由矩形查询获得，统计汇总从 O(T²) 降至 O(K)；从边界向量提取区间另需 O(T)。保留无偏方差和 Welch 定义。
4. **顺序控制保持不变：**finetuning 按强度依次调整，保留前次修改对下一次的影响。病态或近平局回到原运算顺序。
5. **硬件实现：**CPU 使用 Numba 查询；PyTorch 使用批量矩阵运算；Triton 融合区间评分。单/双候选 Triton kernel 已独立测试，但生产逐轮控制仍在 CPU。

Full Q 是 O(T²) 存储，T=2000、FP64 约 32 MB（30.5 MiB），还需相关矩阵、统计前缀和临时空间。CPU `q_strategy="lazy"` 仅节省 Q，t-distance 仍为二次存储。GPU 使用分块/批量计算，不构造 T×T×P 大张量。

→ [完整数学说明](docs/mathematics.md) · [官方源码审计](docs/upstream_algorithm_audit.md) · [架构](docs/architecture.md) · [BBBD 集成](docs/bbbd_integration.md)

## 测试、开源与引用

```bash
python -m pytest tests -q
python benchmarks/single_fit.py --backend cpu-fast --T 450 --P 16 --kmax 20 --output small_cpu.json
python benchmarks/single_fit.py --backend reference --T 450 --P 16 --kmax 20 --output small_reference.json
python benchmarks/batch_throughput.py --backend cpu-fast --workers 4 --batch-sizes 8,32 --output batch_cpu.json
```

公共版本包含代码、合成示例、测试、性能与验证汇总；不包含研究矩阵、参与者文件映射、逐记录结果或虚拟环境。依赖私有数据的脚本不属于快速开始必需步骤。贡献与问题复现见 [CONTRIBUTING.md](CONTRIBUTING.md)。

基于 [statesegmentation](https://github.com/lgeerligs/statesegmentation)，使用时请引用原始算法论文及本软件：

- Geerligs et al. (2021), [10.1016/j.neuroimage.2021.118085](https://doi.org/10.1016/j.neuroimage.2021.118085)。
- Geerligs et al. (2022), [10.7554/eLife.77430](https://doi.org/10.7554/eLife.77430)。
- [软件引用元数据](CITATION.cff)。

代码采用 [MIT License](LICENSE)，上游署名保留于 [NOTICE](NOTICE)。宣传图由 AI 生成，仅为概念艺术，不是科学测量图；[生成说明](docs/assets/README.md)。


补测复现：`python benchmarks/complete_reference.py --workloads Medium Large XL --output paired_results.json`。完整三重复可能持续数小时；加入 BBBD0 需要未公开的原始冻结输入，并添加 `--single-pass BBBD0` 复现本次单次协议。每次拟合后的结果会即时写入对应 `_trials.json`，不会根据部分结果自动填表。
