# 一致性验收

> 本页保留 v0.1.0 的证据与范围；新增路径及逐次微调验收见
> [v0.2 实测报告](v02_results_CN.md)。

- Windows：103 passed / 0 failed / 1 skipped。
- WSL2（含真实 CUDA/Triton）：134 passed / 0 failed / 0 skipped。
- 计数是测试用例数，不是独立生物学记录数。真实记录只有 4 条，覆盖 4 视频、2 session、full64 与 frontal32。源空间网络尚未提供合法冻结输入，不计入完成。

数学验证涵盖全部小型合法区间、ddof=1、单/双评分、实际 stateopt 筛选、不同 dmin、Welch、小样本、零方差、平局和索引。参数矩阵包含 statewise True/False、finetune 0/1/2/-1、顺序 True/False、blocksize 0/7/50、y 与低 P/64P。跨后端回归记录每轮候选、分数、两类 t-distance、单/双选择、微调前后编号边界。

## 真实记录

| id | T×P | CPU vs 历史 all_bounds | 最大 t 误差 |
|---|---|---|---|
| 0 | 1336×64 | True | 2.43e-13 |
| 1 | 1857×32 | True | 3.02e-13 |
| 2 | 1754×64 | True | 7.46e-13 |
| 3 | 1374×64 | True | 1.35e-13 |

每条历史缓存对应完整原 Kmax，不减采样、不关 statewise/finetune。另有四条 96 点、Kmax=12 的开发片段官方 trace，明确与完整记录分开。新跑完整官方 reference 已保存，并完成逐轮候选/微调路径比较，详见 saved_validation.json。

| id | 后端 | 精度 | 所有边界一致 | 严格数值通过 | 最大 t 误差 | 首次路径分歧 |
|---|---|---|---|---|---|---|
| 0 | torch-cuda | float64 | True | True | 2.43e-13 | None |
| 0 | torch-cuda | float32 | True | False | 4.68e-05 | None |
| 0 | triton-cuda | float64 | True | True | 2.43e-13 | None |
| 0 | triton-cuda | float32 | True | False | 4.68e-05 | None |
| 1 | torch-cuda | float64 | True | True | 3.13e-13 | None |
| 1 | torch-cuda | float32 | True | False | 0.00011 | None |
| 1 | triton-cuda | float64 | True | True | 3.13e-13 | None |
| 1 | triton-cuda | float32 | True | False | 0.00011 | None |
| 2 | torch-cuda | float64 | True | True | 2.77e-13 | None |
| 2 | torch-cuda | float32 | True | False | 0.000354 | None |
| 2 | triton-cuda | float64 | True | True | 2.77e-13 | None |
| 2 | triton-cuda | float32 | True | False | 0.000354 | None |
| 3 | torch-cuda | float64 | True | True | 1.85e-13 | None |
| 3 | torch-cuda | float32 | True | False | 4.02e-05 | None |
| 3 | triton-cuda | float64 | True | True | 1.85e-13 | None |
| 3 | triton-cuda | float32 | True | False | 4.02e-05 | None |
| 0 | cpu-fast | float64 | True | True | 2.43e-13 | None |
| 0 | torch-cuda | float64 | True | True | 9.59e-14 | None |
| 0 | triton-cuda | float64 | True | True | 9.59e-14 | None |

GPU 表主要以通过历史缓存对照的 CPU 全轨迹作基准；baseline 字段为 fresh official trace 的行是直接官方对照。默认 `atol=rtol=1e-8`，实际 FP64 误差远小于阈值。FP32 的 4 条边界虽一致，但 t-distance 误差为约 4e-5–3.54e-4，首次数值分歧可在 K=2 出现，因此没有标记为科学等价。记录差异定位见 saved_validation.json 的 first_numerical_divergence。

## 负面结果与保护

1. 空间上近常数、偏置 1e10 加 1e-5 噪声的输入，原优化路径首轮曾出现 single=6，而官方 single=1；官方候选本身出现 NaN。现在按相对机器精度检测病态行，整条回退未改动的官方计算，并记录 requested_backend、actual backend=reference、fallback_reason。不是加 epsilon 或填补信号。
2. 精确相消的区间均值会导致官方候选 NaN；局部原评分回退保留原行为。P=1、全零/空间常数行、NaN/Inf 输入明确拒绝。不能把含未定义量的结果解释为正常可用科学测量。
3. 合成恢复与 reference parity 分开：synthetic_recovery.json 的 4 个信号全部 parity 通过，但真值边界 recall 随 seed/噪声变化（低噪声也有 2/3 案例）。算法一致不等于真值完美恢复。
4. 平局/近平局使用原运算顺序重新评分，t 比较近平局使用原 SciPy Welch；这不是任意病态矩阵上的误差上界证明。

恢复功能不仅做存在性检查：包括真实四任务首跑成功/重跑跳过、受控终止子进程后恢复、不覆盖异配置文件、失效输入隔离，以及注入 CUDA OOM 后缩批重试。没有通过真实耗尽显存来测试 OOM。

原始证据：[Windows 测试](../results/benchmarks/tests_windows.txt)、[WSL 测试](../results/benchmarks/tests_wsl.txt)、[真实完整比较](../results/benchmarks/saved_validation.json)、[历史缓存比较](../results/benchmarks/historical_parity.json)、[合成恢复](../results/benchmarks/synthetic_recovery.json)。
