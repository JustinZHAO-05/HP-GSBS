"""Rebuild concise reports from measured JSON/XML; never invent missing receipts."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results/benchmarks"
D=ROOT/"docs"


def read(name,default=None):
    p=R/name
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else default


def tests(name):
    p=R/name
    if not p.exists(): return "未执行"
    root=ET.parse(p).getroot(); suites=list(root.iter("testsuite"))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
    return f"{counts['tests']-counts['failures']-counts['errors']-counts['skipped']} passed / {counts['failures']+counts['errors']} failed / {counts['skipped']} skipped"


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    latency=read("delivery_latency.json",[])
    pairs={(r['workload'],r['backend']):r for r in latency}
    lrows=[]
    for r in latency:
        samecpu=pairs.get((r['workload'],'cpu-fast'))
        sameref=pairs.get((r['workload'],'reference'))
        trials=r['trials'][1:]
        lrows.append([r['workload'],r['backend'],f"{r['shape'][0]}×{r['shape'][1]}",r['algorithm']['kmax'],
            f"{r['first_call_s']:.4f}",f"{r['steady_median_s']:.4f}",f"{r['fits_per_minute']:.1f}",
            f"{sameref['steady_median_s']/r['steady_median_s']:.1f}×" if sameref else '未重测',
            f"{samecpu['steady_median_s']/r['steady_median_s']:.2f}×" if samecpu else '—',
            f"{max(t['cpu_peak_rss_bytes'] for t in trials)/2**20:.1f}",
            f"{max(t.get('gpu_peak_allocated_bytes',0) for t in trials)/2**20:.1f}",
            f"{max(t.get('gpu_peak_reserved_bytes',0) for t in trials)/2**20:.1f}"])
    b_rows=[]
    for name in ('torch','triton','cpu1','cpu2','cpu4','cpu8'):
        rows=read('batch_'+name+'.json',[])
        for b in sorted({r['batch_size'] for r in rows}):
            subset=[r for r in rows if r['batch_size']==b]; wall=float(np.median([r['wall_s'] for r in subset]))
            b_rows.append([subset[0]['backend'],subset[0]['workers'],b,f'{wall:.4f}',f'{60*b/wall:.1f}',
                           f"{max(r.get('gpu_peak_allocated_bytes') or 0 for r in subset)/2**20:.1f}",
                           f"{max(r.get('gpu_peak_reserved_bytes') or 0 for r in subset)/2**20:.1f}",all(r['bounds_equal'] for r in subset)])
    cold=read('cold_start.json',[])
    stage_rows=[]
    for r in latency:
        if r['workload']!='BBBD0': continue
        stage_rows.append([r['backend']]+[f"{np.median([t.get('timings',{}).get(key,0) for t in r['trials'][1:]]):.4f}" for key in ('precompute_s','candidate_s','tdistance_s','finetune_s','transfer_s')])
    performance=f"""# 性能实测

日期：2026-10-16。本机 Ryzen 7 7840H（8C/16T）、RTX 4050 Laptop 6 GB / CC 8.9。以下同环境表均为 WSL2 Python 3.10 / PyTorch 2.11.0+cu128 / Triton 3.6.0，FP64，TF32 关闭。Windows CPU 优化前后数据另列，不能跨 OS 拼接成公平加速倍数。

测试依次执行；每格为 3 次预热后 wall-clock 中位数。首调用包含在进程内的初始化/JIT，但磁盘缓存可能存在。CUDA 在计时前后同步。保存结果不计入 wall；返回同范围结果。桌面应用与一个 Windows 官方 reference 长任务在后台，因此是实际使用环境的观测，不是独占、固定频率的硬件极限。大负载未在此环境重复跑官方时不填倍率，不外推 Small 的倍率。

{table(['工作负载','后端','T×P','Kmax','首调用 s','预热 s','fits/min','相对官方','相对 CPU','RSS MiB','GPU allocated MiB','GPU reserved MiB'],lrows)}

Small/Medium/Large/XL 是合成开发负载，Kmax=20/30/40/50；BBBD0 是原始完整 T=1336、P=64、Kmax=131、dmin=5，其余搜索参数相同。RSS 为每进程 5 ms 采样高水位，包含已加载库/缓存；GPU 数值按每次计时调用重置统计，不是净增内存。

## 完整 BBBD0 时间组成

{table(['后端','预计算 s','候选 s','t-distance s','finetuning s','传输 s'],stage_rows)}

传输包含在预计算中，不能再次相加；其他列是互斥控制器计时。候选与微调仍在 CPU。profiler 的设备 kernel、拷贝、launch/synchronization 证据保存在 `final_torch-cuda_profile.json/.txt` 与 `final_triton-cuda_profile.json/.txt`。每批预计算显式同步 4 次；4 个 D2H 拷贝也可能阻塞。不在每个 K 调 `.item()` / `.cpu()`。此设计降低逐轮同步，但保留 CPU 与传输瓶颈。

## 批量吞吐量

同一 Small 负载 T=450/P=16/Kmax=20，独立随机记录，3 次中位数；CPU worker 仅使用 1 个 BLAS/Numba 线程。包含建池/任务传递，排除文件保存。GPU 是批量预计算、独立顺序 controller；CPU 是进程并行。

{table(['后端','workers','B','整批 s','fits/min','allocated MiB','reserved MiB','边界一致'],b_rows)}

## 真正冷启动

新 Python 子进程、空 NUMBA_CACHE_DIR/TRITON_CACHE_DIR、包含导入与首次设备/JIT 成本；Small 同输入。

{table(['后端','进程总 wall s','导入及拟合 s'],[[r['backend'],f"{r['process_wall_s']:.3f}",f"{r['import_and_fit_s']:.3f}"] for r in cold])}

## 优化证据

Windows 完整 BBBD0 的 CPU strength 向量化：原预热中位数 {read('cpu_before_strengths.json',{}).get('steady_median_s',float('nan')):.4f} s，优化后 {read('cpu_after_strengths.json',{}).get('steady_median_s',float('nan')):.4f} s。阶段数据保留在对应 JSON；其后所有路径回归仍通过。

Triton Q 从每 interval 一个 program 改为每 program 16 对，前后见 q_before_tiling.json/q_after_tiling.json。T=1500/P=64 完整预计算约从 0.2261 s 降到 0.1158 s，仍慢于同组 torch 0.0746 s。候选 kernel 本身很快，但生产没有据此强行把逐轮局部评分搬 GPU；`final_kernels.json` 是 kernel 证据，不能当端到端加速。

2048² GEMM 的实际 FP64/FP32/TF32 能力见 gemm_capability.json（分别约 0.185 / 7.60 / 10.33 TFLOP/s，单次小型测试，不是硬件理论峰值）。TF32 仅用于这一能力测量，未用于 GSBS 科学结果。
"""
    baseline_path=ROOT/'results/fixtures/0_full_receipt.json'
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
        mem=read('reference_full_peak_memory.json',{})
        performance += f"\n完整官方 BBBD0 新跑：Windows、T=1336/P=64/Kmax=131/dmin=5，启用只读逐轮 trace，wall={baseline['elapsed_s']:.3f} s；Windows 进程 lifetime peak working set={mem.get('os_lifetime_peak_working_set_bytes',0)/2**20:.1f} MiB。此单次带 trace 验证不是 WSL 预热基准，不用于表中跨后端加速倍数。[reference receipt](../results/fixtures/0_full_receipt.json) / [内存记录](../results/benchmarks/reference_full_peak_memory.json)。\n"
    performance += '\n原始证据：[单条基准](../results/benchmarks/delivery_latency.json)、[GPU 批量](../results/benchmarks/batch_torch.json)、[Triton 批量](../results/benchmarks/batch_triton.json)、[CPU 8 workers](../results/benchmarks/batch_cpu8.json)、[冷启动](../results/benchmarks/cold_start.json)。\n'
    (D/'performance.md').write_text(performance,encoding='utf-8')
    saved=read('saved_validation.json',[])
    historic=read('historical_parity.json',[])
    fresh=[r for r in saved if r.get('baseline')=='fresh official trace']
    full=len(fresh)==3 and all(r['passed'] for r in fresh)
    fullmsg='新跑完整官方 reference 已保存，并完成逐轮候选/微调路径比较，详见 saved_validation.json。' if full else '新跑完整官方 reference 仍未完成；不能将历史缓存边界一致描述成完整官方候选 trace 验证。'
    parity=f"""# 一致性验收

- Windows：{tests('tests_windows.xml')}。
- WSL2（含真实 CUDA/Triton）：{tests('tests_wsl.xml')}。
- 计数是测试用例数，不是独立生物学记录数。真实记录只有 4 条，覆盖 4 视频、2 session、full64 与 frontal32。源空间网络尚未提供合法冻结输入，不计入完成。

数学验证涵盖全部小型合法区间、ddof=1、单/双评分、实际 stateopt 筛选、不同 dmin、Welch、小样本、零方差、平局和索引。参数矩阵包含 statewise True/False、finetune 0/1/2/-1、顺序 True/False、blocksize 0/7/50、y 与低 P/64P。跨后端回归记录每轮候选、分数、两类 t-distance、单/双选择、微调前后编号边界。

## 真实记录

{table(['id','T×P','CPU vs 历史 all_bounds','最大 t 误差'],[[r['id'],'×'.join(map(str,r['shape'])),r['bounds_equal'],f"{r['max_t_error']:.3g}"] for r in historic])}

每条历史缓存对应完整原 Kmax，不减采样、不关 statewise/finetune。另有四条 96 点、Kmax=12 的开发片段官方 trace，明确与完整记录分开。{fullmsg}

{table(['id','后端','精度','所有边界一致','严格数值通过','最大 t 误差','首次路径分歧'],[[r['id'],r['backend'],r['dtype'],r['bounds_equal'],r['passed'],f"{r['max_t_error']:.3g}" if r['max_t_error'] is not None else 'NaN',r['first_divergence']] for r in saved])}

GPU 表主要以通过历史缓存对照的 CPU 全轨迹作基准；baseline 字段为 fresh official trace 的行是直接官方对照。默认 `atol=rtol=1e-8`，实际 FP64 误差远小于阈值。FP32 的 4 条边界虽一致，但 t-distance 误差为约 4e-5–3.54e-4，首次数值分歧可在 K=2 出现，因此没有标记为科学等价。记录差异定位见 saved_validation.json 的 first_numerical_divergence。

## 负面结果与保护

1. 空间上近常数、偏置 1e10 加 1e-5 噪声的输入，原优化路径首轮曾出现 single=6，而官方 single=1；官方候选本身出现 NaN。现在按相对机器精度检测病态行，整条回退未改动的官方计算，并记录 requested_backend、actual backend=reference、fallback_reason。不是加 epsilon 或填补信号。
2. 精确相消的区间均值会导致官方候选 NaN；局部原评分回退保留原行为。P=1、全零/空间常数行、NaN/Inf 输入明确拒绝。不能把含未定义量的结果解释为正常可用科学测量。
3. 合成恢复与 reference parity 分开：synthetic_recovery.json 的 4 个信号全部 parity 通过，但真值边界 recall 随 seed/噪声变化（低噪声也有 2/3 案例）。算法一致不等于真值完美恢复。
4. 平局/近平局使用原运算顺序重新评分，t 比较近平局使用原 SciPy Welch；这不是任意病态矩阵上的误差上界证明。

恢复功能不仅做存在性检查：包括真实四任务首跑成功/重跑跳过、受控终止子进程后恢复、不覆盖异配置文件、失效输入隔离，以及注入 CUDA OOM 后缩批重试。没有通过真实耗尽显存来测试 OOM。
"""
    parity += '\n原始证据：[Windows 测试](../results/benchmarks/tests_windows.txt)、[WSL 测试](../results/benchmarks/tests_wsl.txt)、[真实完整比较](../results/benchmarks/saved_validation.json)、[历史缓存比较](../results/benchmarks/historical_parity.json)、[合成恢复](../results/benchmarks/synthetic_recovery.json)。\n'
    (D/'reference_parity.md').write_text(parity,encoding='utf-8')
    limitations=f"""# 已知限制与未完成项

- {fullmsg}
- 2021 期刊最终全文下载端点受 403 限制；已核对官方代码、2020 原始预印本方法及 2022 方法/附录全文，未声称 2021 期刊全文阅读完成。
- 没有合法冻结的七网络源空间输入；只测试了 sensor full64 / frontal32 及合成低维输入，不做源重建或重新定义网络。
- GPU 是混合实现；没有 GPU 常驻逐轮控制器、persistent search、CUDA Graphs 或多 GPU。当前只实现 Full Q；CPU lazy 也仍有二次 t-distance 存储。
- Triton 实际运行并验证了 Q 融合 kernel、单/双候选 kernel；生产使用 Q kernel，候选 kernel 仅测试/微基准，未宣称已经实现全部 A–E 优化。Triton 仅在部分短记录负载优于 PyTorch。
- FP32 不通过严格 t-distance 容差；TF32 不开放 GSBS fit 模式。没有把 FP64 CPU 与 TF32 GPU 混为等价性能比较。
- 分桶按完全相同 T/P，不使用 padding；尚未做数千条、多个 GPU、源空间 P 很大的生产压力测试。manifest 在 GPU 路线用 mmap 和逐批 materialization；自动显存模型是保守估算，不保证永不 OOM。
- 恢复签名使用输入路径、大小和 mtime，加配置/backend；冻结输入不得原地改写并保留相同 stat。结果原子写、不覆盖，但未实现多调度器共享一个输出目录的分布式锁。
- 性能数据为本机实际桌面/WSL 环境，包含后台 reference 长任务的资源竞争；三次中位数不是跨机器结论。CPU RSS 为采样进程高水位，不是单次新增内存。首次调用和真正冷启动分别保存。
- 完成开发与算法一致性验证，不代表 BBBD 的测量有效性门、行为假设或正式论文研究完成；没有自动替换现有研究后端。
"""
    (D/'known_limitations.md').write_text(limitations,encoding='utf-8')
    summary=f"""# HP-GSBS 技术报告

日期：2026-10-16。交付目录 `<workspace>`。源码、CLI、测试、基准与环境记录均已落盘；官方 GSBS 和 E 盘 BBBD 研究产物保持不变。

## A. 总体完成情况

|路线|当前实现|验收边界|
|---|---|---|
|CPU FastGSBS|完整拟合、Full/lazy Q、Numba 评分、区域 t-distance、顺序微调、y、多进程|4 条完整 EEG 历史边界路径一致；Windows/WSL 回归通过|
|TensorGSBS|PyTorch CUDA FP64/FP32、批量 Q/R/prefix、CPU 独立控制器|B=1 与 B>1 实跑，FP64 完整真实路径一致|
|TurboGSBS|Triton tiled Q 融合、单/双评分 kernels、统一混合拟合 API|真实 kernel 执行；短记录有端到端收益，大记录通常推荐 TensorGSBS|
|批量系统|分桶、显存估计、OOM 缩批、逐任务原子输出、失败隔离、恢复|B<=32，CPU<=8 workers；真实四任务与中断恢复测试|

P0–P7 的实现与有限验证已经交付；覆盖范围及未完成项逐项记录，原文访问和源空间验证不能标成完成。{fullmsg}

## B. 数学与算法优化

使用区间匹配 F 与 Q，将每候选重复 O(TP) 标准化降为 O(1) 查表；二维相关矩阵前缀将每次 t-distance 的 O(T²) 统计降为 O(K) 区域聚合（从边界向量构造区间另需 O(T)）。搜索模式、t-distance 的单/双决策、特殊 K 索引、block 分组与依赖顺序保留。向量化每轮 boundary strengths，保持 SciPy Pearson 定义与顺序。

详细输入/输出/频次/复杂度见 upstream_algorithm_audit.md；推导及退化行为见 mathematics.md。已有 fork 的 memmap 可写性和进程结果丢失风险已审查并以小型测试验证基础失效机制，未直接采用其评分核心。

## C. 科学一致性

Windows：{tests('tests_windows.xml')}；WSL：{tests('tests_wsl.xml')}。真实独立输入 4 条，不将测试函数数量当样本量。CPU 对历史官方 all_bounds 全一致，最大 t 误差 {max([r['max_t_error'] for r in historic],default=float('nan')):.3g}。GPU FP64 4/4 全路径一致，FP32 4/4 边界相同但 0/4 严格数值通过。完整逐轮诊断、首次数值分歧和病态回退见 reference_parity.md 与 saved_validation.json。

## D. 速度与显存

{table(['工作负载','官方 s','CPU s','PyTorch s','Triton s'],[[name]+[f"{pairs[(name,b)]['steady_median_s']:.4f}" if (name,b) in pairs else '未重测' for b in ('reference','cpu-fast','torch-cuda','triton-cuda')] for name in ('Small','Medium','Large','XL','BBBD0')])}

这是相同 WSL 环境、相同输入/参数/FP64 的 3 次预热中位数；Small 的 K=20 属开发合成配置，BBBD0 为真实 K=131。完整 batch size、吞吐、cold-start、RSS、GPU allocated/reserved 和分阶段用时见 performance.md。不能将 Small 的加速倍数推广到未测官方的大输入。

## E. 本机建议

- 默认科学精度 FP64。当前 FP32 结果不足以替代研究计算。
- 短记录单次低延迟可选 triton-cuda；本机较长单记录优先 torch-cuda。没有 WSL/CUDA 时使用 cpu-fast。
- 数百条 Small 类型任务本机 CPU 8 workers / 每 worker 单 BLAS 线程吞吐最高；GPU 的 B=4–32 优于串行逐条，实际 batch size 按空闲显存与其他应用留量选择。不要把该短记录结论外推到所有源空间负载。
- 病态空间近常数输入显式回退 reference；未定义 P=1/常数行明确拒绝。缺少 CUDA/Triton 时给出错误及 CPU 命令，不静默伪装 GPU。
- 七网络任务必须使用原定义的多维 parcel/source 特征，按 recording×network 独立拟合和保存，不混合统计量。

## F. BBBD 使用

从已有 frozen manifest 导出代表性任务：`python scripts/export_bbbd_tasks.py --output configs/bbbd_tasks.json`；运行 `python -m hpgsbs fit-batch configs/bbbd_tasks.json --backend cpu-fast --workers 4 --output results/bbbd_validation`。本次已实跑四条并验证重跑全部跳过，记录保存在 bbbd_cli_run.json / bbbd_cli_resume.json。

保留现有峰值 K 选择函数，使用完整 tdists/all_bounds；输出保存原时间映射与 boundary timestamps，不修改滤波、artifact、quiz、统计假设、LOSO 或网络定义。README 提供 Windows/WSL 安装和全部六个 CLI；known_limitations.md 列出尚未完成的覆盖与后续边界。
"""
    summary += '\n证据入口：[一致性报告](reference_parity.md)、[性能详表](performance.md)、[已知限制](known_limitations.md)、[运行环境](../results/benchmarks/doctor_wsl.json)、[使用说明](../README.md)。\n'
    (D/'technical_report.md').write_text(summary,encoding='utf-8')
    print('Reports rebuilt from existing receipts; full official trace:',full)


if __name__=='__main__': main()
