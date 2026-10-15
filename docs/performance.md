# 性能实测

日期：2026-10-16。本机 Ryzen 7 7840H（8C/16T）、RTX 4050 Laptop 6 GB / CC 8.9。以下同环境表均为 WSL2 Python 3.10 / PyTorch 2.11.0+cu128 / Triton 3.6.0，FP64，TF32 关闭。Windows CPU 优化前后数据另列，不能跨 OS 拼接成公平加速倍数。

测试依次执行；每格为 3 次预热后 wall-clock 中位数。首调用包含在进程内的初始化/JIT，但磁盘缓存可能存在。CUDA 在计时前后同步。保存结果不计入 wall；返回同范围结果。桌面应用与一个 Windows 官方 reference 长任务在后台，因此是实际使用环境的观测，不是独占、固定频率的硬件极限。大负载未在此环境重复跑官方时不填倍率，不外推 Small 的倍率。

| 工作负载 | 后端 | T×P | Kmax | 首调用 s | 预热 s | fits/min | 相对官方 | 相对 CPU | RSS MiB | GPU allocated MiB | GPU reserved MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Small | reference | 450×16 | 20 | 45.8050 | 48.0354 | 1.2 | 1.0× | 0.00× | 191.7 | 0.0 | 0.0 |
| Small | cpu-fast | 450×16 | 20 | 0.1806 | 0.1685 | 356.0 | 285.0× | 1.00× | 199.0 | 0.0 | 0.0 |
| Small | torch-cuda | 450×16 | 20 | 1.1798 | 0.1096 | 547.5 | 438.4× | 1.54× | 1103.1 | 22.7 | 42.0 |
| Small | triton-cuda | 450×16 | 20 | 2.0759 | 0.0982 | 610.9 | 489.1× | 1.72× | 1196.1 | 19.5 | 42.0 |
| Medium | cpu-fast | 900×32 | 30 | 0.6372 | 0.5098 | 117.7 | 未重测 | 1.00× | 1245.4 | 0.0 | 0.0 |
| Medium | torch-cuda | 900×32 | 30 | 0.2719 | 0.2446 | 245.3 | 未重测 | 2.08× | 1227.5 | 66.4 | 84.0 |
| Medium | triton-cuda | 900×32 | 30 | 0.5559 | 0.2999 | 200.1 | 未重测 | 1.70× | 1227.7 | 53.5 | 84.0 |
| Large | cpu-fast | 1500×64 | 40 | 1.5947 | 1.6042 | 37.4 | 未重测 | 1.00× | 1356.6 | 0.0 | 0.0 |
| Large | torch-cuda | 1500×64 | 40 | 0.8402 | 0.4816 | 124.6 | 未重测 | 3.33× | 1339.6 | 175.7 | 198.0 |
| Large | triton-cuda | 1500×64 | 40 | 0.6265 | 0.5007 | 119.8 | 未重测 | 3.20× | 1356.9 | 139.8 | 198.0 |
| XL | cpu-fast | 2100×64 | 50 | 4.2448 | 4.1448 | 14.5 | 未重测 | 1.00× | 1473.6 | 0.0 | 0.0 |
| XL | torch-cuda | 2100×64 | 50 | 2.4795 | 2.2567 | 26.6 | 未重测 | 1.84× | 1406.7 | 327.1 | 504.0 |
| XL | triton-cuda | 2100×64 | 50 | 2.8573 | 2.3134 | 25.9 | 未重测 | 1.79× | 1388.2 | 257.0 | 504.0 |
| BBBD0 | cpu-fast | 1336×64 | 131 | 3.9600 | 4.1066 | 14.6 | 未重测 | 1.00× | 1328.8 | 0.0 | 0.0 |
| BBBD0 | torch-cuda | 1336×64 | 131 | 3.3656 | 3.1249 | 19.2 | 未重测 | 1.31× | 1315.3 | 138.7 | 504.0 |
| BBBD0 | triton-cuda | 1336×64 | 131 | 3.6049 | 3.3391 | 18.0 | 未重测 | 1.23× | 1383.7 | 110.1 | 504.0 |

Small/Medium/Large/XL 是合成开发负载，Kmax=20/30/40/50；BBBD0 是原始完整 T=1336、P=64、Kmax=131、dmin=5，其余搜索参数相同。RSS 为每进程 5 ms 采样高水位，包含已加载库/缓存；GPU 数值按每次计时调用重置统计，不是净增内存。

## 完整 BBBD0 时间组成

| 后端 | 预计算 s | 候选 s | t-distance s | finetuning s | 传输 s |
|---|---|---|---|---|---|
| cpu-fast | 1.0720 | 1.1891 | 0.3865 | 1.4061 | 0.0000 |
| torch-cuda | 0.1601 | 1.1886 | 0.3908 | 1.3773 | 0.0468 |
| triton-cuda | 0.3157 | 1.1975 | 0.3938 | 1.3880 | 0.0427 |

传输包含在预计算中，不能再次相加；其他列是互斥控制器计时。候选与微调仍在 CPU。profiler 的设备 kernel、拷贝、launch/synchronization 证据保存在 `final_torch-cuda_profile.json/.txt` 与 `final_triton-cuda_profile.json/.txt`。每批预计算显式同步 4 次；4 个 D2H 拷贝也可能阻塞。不在每个 K 调 `.item()` / `.cpu()`。此设计降低逐轮同步，但保留 CPU 与传输瓶颈。

## 批量吞吐量

同一 Small 负载 T=450/P=16/Kmax=20，独立随机记录，3 次中位数；CPU worker 仅使用 1 个 BLAS/Numba 线程。包含建池/任务传递，排除文件保存。GPU 是批量预计算、独立顺序 controller；CPU 是进程并行。

| 后端 | workers | B | 整批 s | fits/min | allocated MiB | reserved MiB | 边界一致 |
|---|---|---|---|---|---|---|---|
| torch-cuda | 1 | 1 | 0.0925 | 648.9 | 22.7 | 42.0 | True |
| torch-cuda | 1 | 2 | 0.1519 | 789.8 | 35.6 | 62.0 | True |
| torch-cuda | 1 | 4 | 0.2052 | 1169.6 | 61.2 | 84.0 | True |
| torch-cuda | 1 | 8 | 0.4147 | 1157.6 | 112.6 | 156.0 | True |
| torch-cuda | 1 | 16 | 0.8481 | 1131.9 | 215.2 | 368.0 | True |
| torch-cuda | 1 | 32 | 1.6840 | 1140.2 | 424.2 | 768.0 | True |
| triton-cuda | 1 | 1 | 0.0764 | 784.8 | 19.5 | 42.0 | True |
| triton-cuda | 1 | 2 | 0.1143 | 1049.6 | 29.1 | 42.0 | True |
| triton-cuda | 1 | 4 | 0.2165 | 1108.8 | 48.4 | 82.0 | True |
| triton-cuda | 1 | 8 | 0.4597 | 1044.2 | 86.8 | 126.0 | True |
| triton-cuda | 1 | 16 | 0.9423 | 1018.8 | 163.8 | 286.0 | True |
| triton-cuda | 1 | 32 | 1.9086 | 1006.0 | 320.4 | 586.0 | True |
| cpu-fast | 1 | 1 | 0.1413 | 424.6 | 0.0 | 0.0 | True |
| cpu-fast | 1 | 8 | 0.9350 | 513.4 | 0.0 | 0.0 | True |
| cpu-fast | 1 | 32 | 3.7744 | 508.7 | 0.0 | 0.0 | True |
| cpu-fast | 2 | 8 | 0.6910 | 694.6 | 0.0 | 0.0 | True |
| cpu-fast | 2 | 32 | 2.5497 | 753.0 | 0.0 | 0.0 | True |
| cpu-fast | 4 | 8 | 0.5212 | 920.9 | 0.0 | 0.0 | True |
| cpu-fast | 4 | 32 | 1.5004 | 1279.7 | 0.0 | 0.0 | True |
| cpu-fast | 8 | 8 | 0.4689 | 1023.6 | 0.0 | 0.0 | True |
| cpu-fast | 8 | 32 | 1.0062 | 1908.2 | 0.0 | 0.0 | True |

## 真正冷启动

新 Python 子进程、空 NUMBA_CACHE_DIR/TRITON_CACHE_DIR、包含导入与首次设备/JIT 成本；Small 同输入。

| 后端 | 进程总 wall s | 导入及拟合 s |
|---|---|---|
| cpu-fast | 3.442 | 2.980 |
| torch-cuda | 7.962 | 6.743 |
| triton-cuda | 10.545 | 9.486 |

## 优化证据

Windows 完整 BBBD0 的 CPU strength 向量化：原预热中位数 7.2645 s，优化后 3.3689 s。阶段数据保留在对应 JSON；其后所有路径回归仍通过。

Triton Q 从每 interval 一个 program 改为每 program 16 对，前后见 q_before_tiling.json/q_after_tiling.json。T=1500/P=64 完整预计算约从 0.2261 s 降到 0.1158 s，仍慢于同组 torch 0.0746 s。候选 kernel 本身很快，但生产没有据此强行把逐轮局部评分搬 GPU；`final_kernels.json` 是 kernel 证据，不能当端到端加速。

2048² GEMM 的实际 FP64/FP32/TF32 能力见 gemm_capability.json（分别约 0.185 / 7.60 / 10.33 TFLOP/s，单次小型测试，不是硬件理论峰值）。TF32 仅用于这一能力测量，未用于 GSBS 科学结果。

完整官方 BBBD0 新跑：Windows、T=1336/P=64/Kmax=131/dmin=5，启用只读逐轮 trace，wall=4252.563 s；Windows 进程 lifetime peak working set=149.5 MiB。此单次带 trace 验证不是 WSL 预热基准，不用于表中跨后端加速倍数。[reference receipt](../results/fixtures/0_full_receipt.json) / [内存记录](../results/benchmarks/reference_full_peak_memory.json)。

原始证据：[单条基准](../results/benchmarks/delivery_latency.json)、[GPU 批量](../results/benchmarks/batch_torch.json)、[Triton 批量](../results/benchmarks/batch_triton.json)、[CPU 8 workers](../results/benchmarks/batch_cpu8.json)、[冷启动](../results/benchmarks/cold_start.json)。
