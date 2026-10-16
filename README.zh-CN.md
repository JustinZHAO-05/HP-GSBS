<div align="center">

![HP-GSBS：神经信号经 CPU、PyTorch、Triton 三条计算路线得到一致状态边界](docs/assets/hpgsbs-hero.png)

# HP-GSBS

**High-Performance Greedy State Boundary Search · v0.2.0**

更快计算多变量神经记录中的时间状态，并保留可审计的科学决策。

[![CPU tests](https://github.com/JustinZHAO-05/HP-GSBS/actions/workflows/tests.yml/badge.svg)](https://github.com/JustinZHAO-05/HP-GSBS/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-22b8a8.svg)](LICENSE)
[![Precision](https://img.shields.io/badge/Scientific_default-FP64-153c60)](#科学一致性)

[快速开始](#快速开始) · [性能实测](#性能实测) · [迁移说明](docs/v02_migration.md) · [下载](https://github.com/JustinZHAO-05/HP-GSBS/releases) · [English](README.md)

</div>


## 项目用途

GSBS 将 **时间 × 特征** 矩阵分为连续、空间活动模式相对稳定的状态。HP-GSBS 保留原始 GSBS 科学定义，面向 EEG、其他多变量神经时间序列及多个独立 recordings。输入应是已经完成研究规定预处理的矩阵，算法参数来自研究设计。

默认科学精度是 FP64。保留单/双边界插入、blockwise search、t-distance 决策、顺序 finetuning、boundary birth label 和特殊 K 索引。预处理、伪迹策略、K 选择及下游统计仍由原研究决定。原始方法及上游实现归属于原作者，见[作者与引用](#作者与引用)。

| 路线 | 实现 | 用途 |
|---|---|---|
| **CPU FastGSBS** · `cpu-fast` | NumPy/BLAS 区间统计及 Numba streaming 搜索 | CPU-only 拟合、独立多进程 |
| **TensorGSBS** · `torch-cuda` | PyTorch CUDA 批量预计算及共享 CPU streaming/refinement controller | NVIDIA PyTorch 路线 |
| **TurboGSBS** · `triton-cuda` | Triton 区间评分 kernel 及同一 CPU controller | 另一 GPU 路线，需按实际负载测量 |
| **官方参考** · `reference` | 未修改的 statesegmentation.GSBS 0.0.6 | 独立科学对照 |

GPU 路线是 **GPU 预计算 + CPU 顺序搜索的混合架构**，并非完整 GPU-resident GSBS。三个优化后端都受益于新的 CPU streaming/refinement controller。Triton 不保证总是最快；各记录保留独立搜索路径。

## v0.2.0 新功能

- **默认稳定优化：**streaming candidate reducer 不再物化完整双切候选 score matrix；window-local refinement 与 active-block cache 减少重复工作。保留 birth label、微调顺序、sentinel/tie/NaN/stateopt 语义及 t-distance 决策。
- **有界执行：**候选、原表达式回退、pair evaluation 和 workspace 预算超限时明确拒绝；CPU 取消抛出 `SearchCancelled`，不将未完成的近似结果标为成功。
- **可审计性：**严格的输出/搜索/逐移动/numeric 比较等级、扩展数值边界测试、版本化 manifest 执行签名；GPU 分桶隔离模式及显存比例，CPU/GPU 恢复都要求 success。
- **显式实验功能：**raw-prefix bounded chunks、Feature-Moments TD、selective Triton resident candidate reduction、FP32。它们有更窄的验证域，不自动替代默认路线。
- **研究原型：**incremental TD transaction/rollback/commit、GPU no-Q prefix、GPU fixed-partition moments。不能将这些组件描述为稳定完整拟合的公开 API 后端。

见 [CHANGELOG](CHANGELOG.md)、[Release Notes](docs/release_notes_v0.2.0.md) 和 [API 范围](docs/v02_api_scope.md)。

## 性能实测

![v0.1 到 v0.2 的 CPU、PyTorch、Triton 完整拟合耗时、实际重复范围与各负载加速比](docs/assets/performance-v0.2.0.png)

[SVG](docs/assets/performance-v0.2.0.svg) · [未舍入汇总](docs/assets/performance-v0.2.0.json) · [绘图脚本](scripts/plot_v02_performance.py) · [原始 receipts](results/v02/merge_9baff92/revisions/trials.json)

Ryzen 7 7840H、RTX 4050 Laptop 6 GB、WSL2 Ubuntu 22.04、FP64、TF32 off。实际 v0.1.0、修复前 v0.2 (`9309874`) 与测量 v0.2 (`9baff92`) 三份源码交错运行：**每个负载/后端/快照两次预热、三次正式测量**；BLAS/Numba 单线程、CUDA 同步，同负载使用同一输入和科学参数。输出验证与保存不计入 fit wall。

下表对比 v0.1.0 与 `9baff92`，倍率直接由未舍入中位数计算。范围为三次实测 min–max，不是置信区间。BBBD0 是同一真实记录反复拟合，不是三个参与者。后续 C22–C24 调度修复及正式版本号变更**没有重新测量此性能矩阵**。

<!-- V02_TABLE_START -->
| Workload | Backend | v0.1.0 (s) | v0.2 snapshot (s) | v0.2 range (s) | Speedup |
|---|---|---:|---:|---:|---:|
| Small | CPU | 0.1235 | 0.1060 | 0.0918–0.1110 | 1.16× |
| Small | PyTorch | 0.0905 | 0.0745 | 0.0713–0.0767 | 1.21× |
| Small | Triton | 0.0850 | 0.0577 | 0.0550–0.0594 | 1.47× |
| Medium | CPU | 0.4231 | 0.3594 | 0.3218–0.3648 | 1.18× |
| Medium | PyTorch | 0.2027 | 0.1294 | 0.1127–0.2284 | 1.57× |
| Medium | Triton | 0.2205 | 0.1417 | 0.1278–0.1572 | 1.56× |
| Large | CPU | 1.2682 | 0.9342 | 0.9200–0.9438 | 1.36× |
| Large | PyTorch | 0.4950 | 0.2209 | 0.2123–0.2647 | 2.24× |
| Large | Triton | 0.5057 | 0.2572 | 0.2038–0.3187 | 1.97× |
| XL | CPU | 3.9172 | 2.1516 | 2.1451–2.3305 | 1.82× |
| XL | PyTorch | 2.1961 | 0.3596 | 0.3377–0.3660 | 6.11× |
| XL | Triton | 2.1311 | 0.5592 | 0.4573–0.5790 | 3.81× |
| BBBD0 | CPU | 3.0056 | 1.5711 | 1.4586–1.5922 | 1.91× |
| BBBD0 | PyTorch | 2.3270 | 0.7896 | 0.7769–0.9666 | 2.95× |
| BBBD0 | Triton | 2.3370 | 0.8193 | 0.7551–0.8731 | 2.85× |

<!-- V02_TABLE_END -->

负载：Small 450×16/K20、Medium 900×32/K30、Large 1500×64/K40、XL 2100×64/K50、BBBD0 1336×64/K131。全部保留 statewise、finetune=1、强度排序及 blocksize=50；合成 dmin=1，BBBD0 dmin=5。FP64 Full-Q/Prefix2D 科学配置相同。图上方为对数秒轴，下方为从零起的倍率及 1× 基线。

完整三快照数据均保留在汇总 JSON。另一组历史集中 CPU 复验中，修复前/修复后为 1.4245/1.4496 s 中位数；这一小幅开销不隐藏，也不与上表拼接。冷启动、批量吞吐、实验性低内存均有独立计时范围。受限 50,000×64、K16、单切/无微调实验中位 1.4439 s、峰值 RSS 469.5 MiB；它不是完整长 N 官方双切轨迹认证。见[性能及边界](docs/performance.md)。

### 历史官方 GSBS 基线

保留 [v0.1 历史性能报告](docs/performance_v0.1.md)、[原 PNG](docs/assets/performance-comparison.png)、[SVG](docs/assets/performance-comparison.svg) 和[原脚本](scripts/plot_performance.py)。该组相对官方 GSBS 的千倍加速使用另一测量合同：合成三次中位数/范围，BBBD0 每后端仅一次。不要将其与本轮 v0.1→v0.2 倍率混用。

## 科学一致性

正式 v0.2.0（`b86b409`）：**Windows 319 passed / 3 skipped；WSL 384 passed / 1 skipped**。公开 wheel/sdist 的 CPU-only 安装验证通过；公开 wheel 的 CPU/PyTorch/Triton 共 12 个完整冻结 EEG 对照通过。[发布后验收](docs/release_verification_v0.2.0.md)分别记录本次验证与历史性能测量。

| 证据 | 范围 |
|---|---|
| `7da2611` 调度快照，Windows | **315 passed / 3 skipped** |
| 同快照，WSL 实际 CUDA/Triton | **380 passed / 1 skipped** |
| 四条完整冻结真实 EEG | Full64/frontal32，默认 FP64 完整输出与经过核验的官方路径一致 |
| dev96 新鲜官方对照 | 原 v0.2 验收中的 8 项完整搜索/微调逐移动比较 |
| C22–C24 回归 | 版本/默认值身份、旧签名与非成功状态拒绝、混合 GPU 策略及输入顺序 |

测试用例数不是参与者数。[验收报告](docs/v02_merge_readiness.md)与[一致性详表](docs/reference_parity.md)区分原科学快照、调度补充和发布检查。**并非所有 full EEG 都重新生成了完整官方逐移动轨迹**。源空间七网络尚无对应冻结输入。GPU FP32 即使边界相同，TD 仍存在严格 FP64 数值差异，故为实验性。严格 trace 验证缺证据就失败。

近平局用原表达式重评；病态输入可回退官方并记录原因，资源/数值保护也可能明确拒绝。这是已验证域的证据，不是所有浮点输入的等价证明。软件等价不代表 BBBD 行为或记忆假说得到验证。不分发私有 EEG 和逐记录拟合输出。

## 快速开始

### 安装

需要 **Python 3.10+**；记录中的 CI 覆盖 3.10/3.11。[v0.2.0 Release](https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.2.0) 提供正式 CPU-only wheel：

```bash
python -m pip install https://github.com/JustinZHAO-05/HP-GSBS/releases/download/v0.2.0/hpgsbs-0.2.0-py3-none-any.whl
python -m hpgsbs doctor
```

也可下载 [Python sdist](https://github.com/JustinZHAO-05/HP-GSBS/releases/download/v0.2.0/hpgsbs-0.2.0.tar.gz)，运行 `python -m pip install hpgsbs-0.2.0.tar.gz`。GitHub 自动 source ZIP/tar.gz 是仓库快照，不等于 Python sdist。**尚未发布 PyPI**，不能假设 `pip install hpgsbs` 安装本项目。

需要示例和测试时，clone 仓库、`git checkout v0.2.0`，创建并激活虚拟环境，再运行：

```bash
python -m pip install -e ".[test]"
python examples/quickstart.py --output examples/generated
python -m pytest tests -q
```

CPU 依赖 NumPy、SciPy、Numba、threadpoolctl、psutil、tqdm 和固定 `statesegmentation==0.0.6`，不会自动安装 Torch/Triton。[pyproject.toml](pyproject.toml) 列出约束和可选依赖。NVIDIA CUDA 请用 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 匹配驱动；实测 WSL/Linux 环境为：

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install triton==3.6.0
python -m hpgsbs doctor
python examples/quickstart.py --backend torch-cuda --output examples/generated/torch
python examples/quickstart.py --backend triton-cuda --output examples/generated/triton
```

PyTorch CUDA 路线不强制需要 Triton 后端。Triton 在此支持 Linux/WSL，非原生 Windows；见[官方安装说明](https://triton-lang.org/main/getting-started/installation.html)。未实现 AMD/Apple GPU 后端。缺少 CUDA 依赖时明确报错，可显式改用 `cpu-fast`。

### 拟合、检查、验证和保存

```python
import numpy as np
from hpgsbs import GSBSConfig, GSBSResult, fit, fit_many
from hpgsbs.validation import compare
from hpgsbs.core.budgets import ResourceBudgetExceeded, SearchCancelled

x = np.random.default_rng(42).normal(size=(40, 8))  # 合成演示
config = GSBSConfig(
    kmax=8, dmin=1,  # 仅演示：EEG 必须使用研究设计规定的 Kmax/dmin
    statewise_detection=True, finetune=1, finetune_order=True,
    blocksize=50, dtype="float64", q_strategy="full",
    candidate_backend="streaming", td_backend="prefix2d", trace=True,
)
try:
    result = fit(x, config, backend="cpu-fast")
    reference = fit(x, config, backend="reference")
    report = compare(reference, result, level="finetune-move-trace")
    if not report["passed"]:
        raise RuntimeError(report)
    bounds_at_6 = result.get_bounds(6)
    full_tdistance_curve = result.tdists
    result.save("my_fit.npz")  # 拒绝覆盖
    restored = GSBSResult.load("my_fit.npz")
    batch = fit_many([x, x.copy()], config, backend="cpu-fast")
except (ResourceBudgetExceeded, SearchCancelled) as exc:
    print("未生成成功拟合：", exc)
except FileExistsError:
    print("请选择新输出文件名；原结果已保留。")
```

可用时保留同一配置，改用 `backend="torch-cuda"` 或 `"triton-cuda"`。`fit(..., y=y)` 在 X 上搜索、用 held-out Y 计算 TD；`save(..., times=times)` 保存原时间映射。比较等级包括 `output-only`、`full-search-trace`、`finetune-move-trace`、`numeric-stress`；严格轨迹等级要求 trace。保存/加载保留兼容 GSBS 方法及所有 K 的输出。

低内存与 resident 明确 opt-in：

```python
from dataclasses import replace
low_memory = replace(config, q_strategy="streaming", td_backend="feature-moments")
# 仅 CPU；普通高 K EEG 往往更慢；近退化统计可能明确拒绝。
resident = replace(config, candidate_backend="triton-resident")
# 仅 CUDA FP64 + full Q + prefix2d；选择性候选计算，并非完整 GPU 常驻。
```

Kmax/dmin 和预处理来自研究设计，示例不修改 BBBD 设置。[迁移文档](docs/v02_migration.md)说明兼容性和实验模式限制。

## 批量、恢复与 CLI

`fit_many` 支持独立记录和逐记录配置。精确 shape/compute-policy 分桶还隔离 `mode` 与 `memory_fraction`；自动 latency 为 B=1，throughput 使用各桶自己的比例规划，显式 batch_size 优先；OOM 缩半。CPU `n_jobs` 控制记录间进程，Windows 多进程调用放入 `if __name__ == "__main__":`。

任务参照 [examples/tasks.json](examples/tasks.json)，路径相对运行目录。每条结果原子写入且不覆盖。恢复要求**版本化执行身份相同且 status=success**：签名版本、软件版本、backend、规范化完整配置、输入 path/size/mtime、可选 Y/time-map 身份。失败、取消、缺失状态或旧签名不会被静默跳过。

v0.1 结果仍可 `GSBSResult.load()`，但不代表已经用 v0.2 执行。身份不匹配时请选择新输出目录。**0.2.0.dev0→0.2.0 会改变执行签名**：开发版结果不自动作为正式版同版本执行缓存复用。这是来源追溯设计，不是计算错误。签名不是内容或源码树 hash，不能检测保留 stat 的输入变化或同一开发版本中的任意源码编辑。

```bash
python -m hpgsbs fit examples/generated/input.npy --kmax 12 --output another_fit.npz
python -m hpgsbs fit-batch examples/tasks.json --backend cpu-fast --workers 4 --output examples/generated/batch-v020
python -m hpgsbs validate examples/generated/input.npy --kmax 12 --backend cpu-fast
python -m hpgsbs compare examples/generated/fit.npz another_fit.npz
```

## 加速算法

空间样本 z-score (`ddof=1`) 与时间前缀和复用区间充分统计。Full Q 提供 O(1) 区间查询；streaming reducer 遍历合法单/双候选而不保存完整 score matrix；window refinement 仅计算实际窗口；active-block cache 仅在同一同步 fit 内有效。Prefix2D TD 保留无偏方差、dmin 与 Welch 定义。默认存储仍为 O(T²)，候选 streaming 并不将 Full Q 变为线性内存。

[数学](docs/mathematics.md) · [架构](docs/architecture.md) · [技术报告](docs/technical_report.md) · [BBBD 集成](docs/bbbd_integration.md) · [已知限制](docs/known_limitations.md) · [贡献指南](CONTRIBUTING.md)

## 作者与引用

**[Yanzhe Zhao](https://github.com/JustinZHAO-05) — Main contributor and maintainer**（Tianjin University，主要贡献者与维护者）。

> Zhao, Y. (2026). *HP-GSBS: High-Performance Greedy State Boundary Search* (Version 0.2.0) [Computer software]. Zenodo. [doi:10.5281/zenodo.23422148](https://doi.org/10.5281/zenodo.23422148)

使用 [CITATION.cff](CITATION.cff)，并引用原始 GSBS 论文及[上游实现](https://github.com/lgeerligs/statesegmentation)：

- Geerligs et al. (2021), [Detecting neural state transitions underlying event segmentation](https://doi.org/10.1016/j.neuroimage.2021.118085)。
- Geerligs et al. (2022), [A partially nested cortical hierarchy of neural states underlies event segmentation in the human brain](https://doi.org/10.7554/eLife.77430)。

[v0.1.0](https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.1.0) 及固定源码 `922a69c` 保留。采用 [MIT License](LICENSE)，[NOTICE](NOTICE) 保留上游归属。[主视觉](docs/assets/README.md)为 AI 概念艺术，不是科学测量。软件验收与发布不构成下游研究结论。
