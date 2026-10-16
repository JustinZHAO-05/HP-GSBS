# HP-GSBS 技术报告

> 本页是 v0.1.0 交付报告。v0.2 分支另见[算法审查](v02_algorithm_review_CN.md)
> 和[实测报告](v02_results_CN.md)，历史结果保留。

日期：2026-10-16。交付目录 `<workspace>`。源码、CLI、测试、基准与环境记录均已落盘；官方 GSBS 和 E 盘 BBBD 研究产物保持不变。

## A. 总体完成情况

|路线|当前实现|验收边界|
|---|---|---|
|CPU FastGSBS|完整拟合、Full/lazy Q、Numba 评分、区域 t-distance、顺序微调、y、多进程|4 条完整 EEG 历史边界路径一致；Windows/WSL 回归通过|
|TensorGSBS|PyTorch CUDA FP64/FP32、批量 Q/R/prefix、CPU 独立控制器|B=1 与 B>1 实跑，FP64 完整真实路径一致|
|TurboGSBS|Triton tiled Q 融合、单/双评分 kernels、统一混合拟合 API|真实 kernel 执行；本次 Small/Large/XL 和单次 EEG 的中位数或单次值较低，Medium 为 TensorGSBS 较低|
|批量系统|分桶、显存估计、OOM 缩批、逐任务原子输出、失败隔离、恢复|B<=32，CPU<=8 workers；真实四任务与中断恢复测试|

P0–P7 的实现与有限验证已经交付；覆盖范围及未完成项逐项记录，原文访问和源空间验证不能标成完成。新跑完整官方 reference 已保存，并完成逐轮候选/微调路径比较，详见 saved_validation.json。

## B. 数学与算法优化

使用区间匹配 F 与 Q，将每候选重复 O(TP) 标准化降为 O(1) 查表；二维相关矩阵前缀将每次 t-distance 的 O(T²) 统计降为 O(K) 区域聚合（从边界向量构造区间另需 O(T)）。搜索模式、t-distance 的单/双决策、特殊 K 索引、block 分组与依赖顺序保留。向量化每轮 boundary strengths，保持 SciPy Pearson 定义与顺序。

详细输入/输出/频次/复杂度见 upstream_algorithm_audit.md；推导及退化行为见 mathematics.md。已有 fork 的 memmap 可写性和进程结果丢失风险已审查并以小型测试验证基础失效机制，未直接采用其评分核心。

## C. 科学一致性

Windows：103 passed / 0 failed / 1 skipped；WSL：134 passed / 0 failed / 0 skipped。真实独立输入 4 条，不将测试函数数量当样本量。CPU 对历史官方 all_bounds 全一致，最大 t 误差 7.46e-13。GPU FP64 4/4 全路径一致，FP32 4/4 边界相同但 0/4 严格数值通过。完整逐轮诊断、首次数值分歧和病态回退见 reference_parity.md 与 saved_validation.json。

## D. 速度与显存

| 工作负载 | 官方 s | CPU s | PyTorch s | Triton s |
|---|---|---|---|---|
| Small | 48.0354 | 0.1685 | 0.1096 | 0.0982 |
| Medium | 349.9463 | 0.4553 | 0.2765 | 0.3281 |
| Large | 2213.4934 | 1.6591 | 0.6733 | 0.5147 |
| XL | 5011.2542 | 3.7777 | 2.3140 | 2.2291 |
| BBBD0 | 6115.3790 | 4.4892 | 3.7375 | 3.4214 |

这是相同 WSL 环境、相同输入/参数/FP64 的比较；合成负载报告预热后三次中位数，BBBD0 按用户要求报告每后端单次实测（无重复波动估计）；Small 的 K=20 属开发合成配置，BBBD0 为真实 K=131。完整 batch size、吞吐、cold-start、RSS、GPU allocated/reserved 和分阶段用时见 performance.md。四个大输入现已补齐官方与优化后端的同进程配对重测；Small 保留初始同环境结果，具体协议与倍率见 performance.md。

## E. 本机建议

- 默认科学精度 FP64。当前 FP32 结果不足以替代研究计算。
- 本次延迟表中 Medium 为 torch-cuda 较快，Small/Large/XL 及单次 BBBD0 为 triton-cuda 较快。GPU 差距存在运行波动，应按自己的输入实测。没有 WSL/CUDA 时使用 cpu-fast。
- 数百条 Small 类型任务本机 CPU 8 workers / 每 worker 单 BLAS 线程吞吐最高；GPU 的 B=4–32 优于串行逐条，实际 batch size 按空闲显存与其他应用留量选择。不要把该短记录结论外推到所有源空间负载。
- 病态空间近常数输入显式回退 reference；未定义 P=1/常数行明确拒绝。缺少 CUDA/Triton 时给出错误及 CPU 命令，不静默伪装 GPU。
- 七网络任务必须使用原定义的多维 parcel/source 特征，按 recording×network 独立拟合和保存，不混合统计量。

## F. BBBD 使用

公开版说明：本节原命令属于本地私有研究验证流程；未分发导出脚本、私有 manifest 或 EEG。公开用户从 `python examples/quickstart.py` 与 `examples/tasks.json` 开始，自己的 EEG 集成见 bbbd_integration.md。

从已有 frozen manifest 导出代表性任务：`python scripts/export_bbbd_tasks.py --output configs/bbbd_tasks.json`；运行 `python -m hpgsbs fit-batch configs/bbbd_tasks.json --backend cpu-fast --workers 4 --output results/bbbd_validation`。本次已实跑四条并验证重跑全部跳过，记录保存在 bbbd_cli_run.json / bbbd_cli_resume.json。

保留现有峰值 K 选择函数，使用完整 tdists/all_bounds；输出保存原时间映射与 boundary timestamps，不修改滤波、artifact、quiz、统计假设、LOSO 或网络定义。README 提供 Windows/WSL 安装和全部六个 CLI；known_limitations.md 列出尚未完成的覆盖与后续边界。

证据入口：[一致性报告](reference_parity.md)、[性能详表](performance.md)、[已知限制](known_limitations.md)、[运行环境](../results/benchmarks/doctor_wsl.json)、[使用说明](../README.md)。
