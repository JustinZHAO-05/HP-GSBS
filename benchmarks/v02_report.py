"""Render the Chinese review report from the committed measurement ledger."""
import json
from pathlib import Path
from statistics import median


def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    rows=read("results/v02/accepted/paired/trials.json")
    text=["# v0.2 科学一致性与性能报告\n",
    "日期：2026-10-16（北京时间）。本分支可独立审核；未合并 main，未改写 v0.1.0。\n",
    "算法、复杂度、失效条件和配置见 [逐项算法审查](v02_algorithm_review_CN.md)。\n",
    "## 配对完整拟合\n",
    "同一 WSL2 / Ryzen 7 7840H / RTX 4050 Laptop 6 GB，Python 3.10、NumPy 2.2.6、"
    "PyTorch 2.11.0+cu128、Triton 3.6.0。FP64，BLAS/Numba 各 1 线程，TF32 关闭，"
    "trace=False。每格 warmup 一次，随后三对交错运行，两个顺序轮换；本次没有并行运行"
    "其他 HP-GSBS 计时任务。桌面和系统频率仍可能波动，三次测量不代表普遍速度保证。\n",
    "before 为保留的 v0.1 `full_scores` + 原物化 reducer，禁用新增局部缓存；after 为"
    "streaming + window refinement + active-block cache。Q/TD、科学参数、输入均相同。"
    "保留同一控制器的对照避免跨 OS/跨硬件混算。最新 main 与 v0.1 的源码/测试相同，"
    "另有修改前 HEAD 冻结测量。下表倍数为两个中位数之比，不是逐对倍数中位数。\n",
    "|负载|后端|before s|after s|倍数|after warm 范围 s|\n|---|---|---:|---:|---:|---:|"]
    for workload in ("Small","Medium","Large","XL","BBBD0"):
        for backend in ("cpu-fast","torch-cuda","triton-cuda"):
            selection=[r for r in rows if r["workload"]==workload and r["backend"]==backend and not r["warmup"]]
            old=[r["wall_s"] for r in selection if r["mode"]=="materialized"]
            new=[r["wall_s"] for r in selection if r["mode"]=="streaming"]
            text.append(f"|{workload}|{backend}|{median(old):.4f}|{median(new):.4f}|{median(old)/median(new):.2f}×|{min(new):.4f}–{max(new):.4f}|")
    text += ["\n### BBBD0 阶段与资源\n",
    "|后端|模式|预计算 s|候选 s|微调 s|TD s|传输 s|RSS MiB 峰值|allocated/reserved MiB|\n|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for backend in ("cpu-fast","torch-cuda","triton-cuda"):
        for mode in ("materialized","streaming"):
            g=[r for r in rows if r["workload"]=="BBBD0" and r["backend"]==backend and r["mode"]==mode and not r["warmup"]]
            vals=[median(r["metadata"]["timings"].get(key,0) for r in g) for key in ("precompute_s","candidate_s","finetune_s","tdistance_s","transfer_s")]
            text.append(f"|{backend}|{mode}|"+"|".join(f"{v:.4f}" for v in vals)+
                f"|{max(r['peak_rss_bytes'] for r in g)/2**20:.1f}|{max(r['gpu_allocated'] or 0 for r in g)/2**20:.1f}/{max(r['gpu_reserved'] or 0 for r in g)/2**20:.1f}|")
    text += ["\n传输属于预计算的子集，不能重复相加。阶段是 wall 插桩，不是独占 kernel 时间。\n",
    "120 次（含 warmup）完整输出对照全部通过；比 all_bounds、完整 tdists、K、patterns、strengths。"
    "计时不采集候选 trace，逐轮科学验证另列。Small/Medium/Large/XL 为 seed=20261016 的"
    "合成输入，T/P/K 分别 450/16/20、900/32/30、1500/64/40、2100/64/50；"
    "BBBD0 是 1336×64、Kmax=131、dmin=5，其余 workload dmin=1。均保留 statewise=True、"
    "finetune=1、blocksize=50。\n",
    "原始 [交错 trials](../results/v02/accepted/paired/trials.json) 保存各次时间、完整阶段计时、"
    "RSS/VRAM 和验证。[修改前 Windows](../results/v02/baseline_windows/latency.json)、"
    "[修改前 WSL](../results/v02/baseline_wsl_run1/latency.json) 与早期 streaming 测量保留；"
    "它们部分与其他验证并行，故不作为最终加速比。RSS 为 5 ms 采样进程总量，配对进程"
    "保留 allocator 高水位，不能据此声称每种模式的净节省。VRAM 在每次调用前同步并重置峰值。\n",
    "## 科学验证与数据边界\n",
    "Windows CPU-only：197 passed、3 skipped；WSL 含真实 CUDA/Triton：245 passed、1 skipped。"
    "完整测试输出和 JUnit 见 results/v02 下 accepted_tests 命名文件。测试数是参数化用例数，不是生物学样本数。"
    "4 条真实 EEG 在 Full-Q 与 raw-prefix/moments 路线逐轮通过，低内存最大 TD 误差"
    "约 7.86e-13；完整 all_bounds、单/双 winner、单双 TD、接受选择和微调最终边界相同。"
    "完整记录 0 直接对现有 official-full trace；1–3 对已通过官方/历史核验的 v0.1 缓存，"
    "本次没有重新运行其数小时官方完整 fit。4 条 dev96 则新跑未改动的官方程序并逐次比较"
    "每个微调删除/窗口/winner/birth。旧完整缓存缺少逐移动字段，不能冒称补齐了该证据。\n",
    "[真实记录结果](../results/v02/real_accepted/validation.json) 仅含 hash、参数和比较结果，"
    "不含原始 EEG。私有文件通过 HPGSBS_FIXTURES 原位读取，未复制或上传。"
    "Fraction oracle 另外验证给定 binary64 U 的精确六矩与 dmin 分类；它不是 native BLAS"
    "运算轨迹证明。NaN/sentinel/tie、重复和跨行 stateopt、chunk 1/7/64、finetune -1/1/2、"
    "独立 y 和极大 dmin 均有独立测试。near-degenerate 低内存 TD 明确拒绝，而不是强制零方差。\n",
    "## Long-N\n",
    "以下为显式 statewise=False、finetune=0、blocksize=0、Kmax=16 的完整低内存拟合；"
    "不能外推为完整官方 statewise+finetune 长轨迹。候选用 raw-prefix bounded chunks，TD 用"
    "feature moments，所有输出 all_bounds 仍保存。预热后 3 次，同一进程不同尺寸顺序测量。\n",
    "|T×P|warm 中位 s|进程峰值 RSS MiB|selected K|selected/final TD 最大审计差|\n|---|---:|---:|---:|---:|"]
    long=read("results/v02/accepted/long/validation.json") or []
    for row in long:
        selected=row["trials"][-1]["algorithm"]["kmax"]
        # Checks include selected and final, deduplicated when both are K16.
        selected=min(c["k"] for c in row["checks"])
        text.append(f"|{row['shape'][0]}×{row['shape'][1]}|{row['steady_median_s']:.4f}|"
            f"{max(v['cpu_peak_rss_bytes'] for v in row['trials'])/2**20:.1f}|{selected}|"
            f"{max(c['error'] for c in row['checks']):.2e}|")
    text += ["\n50,000×64 的一张 FP64 T² 数组理论为 20 GB；本次未尝试该规模的 dense baseline。"
    "selected K=15 与最终 K=16 均单独用 bounded tiled two-pass TD 核验，未只检查最终划分。"
    "这些 oracle 支持固定划分 TD，不认证长 N 每个候选的完整官方轨迹；没有同规模官方加速比。"
    "[原始 long-N 记录](../results/v02/accepted/long/validation.json)。\n",
    "## GPU 常驻候选与集成范围\n",
    "实际 CUDA 组件输入相同 resident FP64 Q，比较 Numba 与 Triton，包含元数据准备/传输、"
    "最后六项摘要 D2H 和同步；Q 构建/上传排除在两边组件计时之外。每 case 预热后 10 次，"
    "不是完整 fit 加速比。单切 launch 开销较大，大规模无 gate 双切可受益。\n",
    "|T|状态数|搜索|Numba ms|resident ms|\n|---|---:|---|---:|---:|"]
    for r in read("results/v02/accepted/resident_component.json") or []:
        text.append(f"|{r['shape'][0]}|{r['states']}|{'double' if r['double'] else 'single'}|"
            f"{r['times']['numba']['median_s']*1000:.3f}|{r['times']['triton-resident']['median_s']*1000:.3f}|")
    text += ["\n选择性 resident 已作为显式 experimental 配置接入同一 controller；只处理无 stateopt gate"
    "且双候选数至少 20,000 的调用。单切、其他双切、顺序微调继续走 CPU。集成的完整 fit"
    "和真实 EEG trace 与组件测试分别验收；并不代表全流程 GPU-resident。\n"]
    resident=read("results/v02/accepted/suite/resident_fit/latency.json")
    if resident:
        text += ["|负载|预计算后端|selective resident 完整 fit s|相对默认 CPU streaming controller|\n|---|---|---:|---:|"]
        for r in resident:
            base=median(v["wall_s"] for v in rows if v["workload"]==r["workload"] and v["backend"]==r["backend"] and v["mode"]=="streaming" and not v["warmup"])
            text.append(f"|{r['workload']}|{r['backend']}|{r['steady_median_s']:.4f}|{base/r['steady_median_s']:.2f}×|")
        text.append("\n此表是后续同环境顺序测量，未与 streaming 再交错配对，不能据小幅差异选择新默认。")
    else: text.append("\n选择性 resident 完整 fit 计时尚待 suite 完成；此处没有填入预测值。")
    throughput=read("results/v02/accepted/suite/throughput.json")
    text += ["\n## 多记录吞吐\n"]
    if throughput:
        text += ["Small 450×16、K20，至多 64 条独立 seed=200+i 输入。CPU 固定 32 条，比较 1/2/4/8 workers；"
        "GPU 比 B=1/2/4/8/16/32，并在保守显存预算允许时测 B=64。包括建池、序列化、传输、同步和全部独立 controller，"
        "保存和输出核对不计时。RSS 为父进程加子进程总量的 10 ms 采样，包含共享页重复计数，"
        "不是精确物理 RAM。每 case 一次 warmup 后 3 次；所有输出检查 all_bounds/TD/K/patterns/strengths。\n",
        "|后端|workers|B|mode|fits/min 中位|wall s 中位|RSS MiB 峰值|allocated/reserved MiB|\n|---|---:|---:|---|---:|---:|---:|---:|"]
        keys=sorted(set((r["backend"],r["workers"],r["batch"],r["mode"]) for r in throughput if "iteration" in r))
        for b,w,n,m in keys:
            group=[r for r in throughput if r.get("backend")==b and r.get("workers")==w and r.get("batch")==n and r.get("mode")==m and not r.get("warmup",True)]
            if not group: continue
            alloc=max(r["gpu_allocated"] or 0 for r in group)/2**20; reserved=max(r["gpu_reserved"] or 0 for r in group)/2**20
            text.append(f"|{b}|{w}|{n}|{m}|{median(r['fits_per_minute'] for r in group):.1f}|"
                f"{median(r['wall_s'] for r in group):.3f}|{max(r['peak_process_tree_rss_bytes'] for r in group)/2**20:.1f}|{alloc:.1f}/{reserved:.1f}|")
        text.append("\nwall 是整个 batch 调用延迟；JSON 的 average_latency_s 定义为摊销 wall/B，不是每记录的完成延迟。单条 latency 另见完整 fit 表。每批 transfer、GPU precompute wall、CPU controller sum、OOM 后的实际预计算批次另存原始 JSON。GPU precompute wall 包含 launch/sync，不能当作纯 kernel 时间；实际 kernel/launch/sync profiler 另存 accepted/suite/profile。")
    else: text.append("吞吐测量尚未完成；不填写预测值。")
    ablation=read("results/v02/accepted/suite/ablation.json")
    if ablation:
        text += ["\n## 每项优化的消融\n",
        "同一 BBBD0 完整参数，四种模式交错，warmup 后各三次。仅在 benchmark 进程中"
        "暂时禁用 HP-GSBS cache/window，finally 恢复；未修改官方类。所有模式对比完整输出。\n",
        "|模式|完整 wall s|candidate s|finetune s|TD s|\n|---|---:|---:|---:|---:|"]
        for mode in ("materialized","stream-only","stream-window","stream-window-cache"):
            group=[r for r in ablation if r["mode"]==mode and not r["warmup"]]
            text.append(f"|{mode}|{median(r['wall_s'] for r in group):.4f}|"
                f"{median(r['metadata']['timings']['candidate_s'] for r in group):.4f}|"
                f"{median(r['metadata']['timings']['finetune_s'] for r in group):.4f}|"
                f"{median(r['metadata']['timings']['tdistance_s'] for r in group):.4f}|")
        text.append("\n单独优化的边际效果只依据相邻两行；不将多阶段倍率相乘。预计算同为 Full-Q + Prefix2D，阶段时间和 wall 的中位数不能机械相加。")
    memory=read("results/v02/accepted/suite/memory.json")
    if memory:
        text += ["\n## 新进程低内存 A/B\n",
        "相同 5,000×8、K16、单切、无微调、blocksize=0；每种模式独立进程，warmup 后三次。"
        "dense 先经保守 RAM 预算检查；没有在 50,000 上分配 dense 数组。\n",
        "|模式|完整 warm s|进程峰值 RSS MiB|\n|---|---:|---:|"]
        for r in memory:
            if "trials" in r:
                text.append(f"|{r['mode']}|{r['steady_median_s']:.4f}|{max(v['cpu_peak_rss_bytes'] for v in r['trials'])/2**20:.1f}|")
            else: text.append(f"|{r['mode']}|{r['status']}|未运行|")
        text.append("\n两条路径检查完整 all_bounds hash、TD 和 K。两边均采用同一受限科学配置。")
    components=read("results/v02/accepted/suite/components.json")
    if components:
        text += ["\n## 长状态首次双切的全候选组件核验\n",
        "|T×P|双候选总数|flush 次数|最大每候选分差|winner|\n|---|---:|---:|---:|---|"]
        for r in components:
            text.append(f"|{r['shape'][0]}×{r['shape'][1]}|{r['candidates']:,}|{r['flushes']:,}|{r['max_candidate_error']:.2e}|相同|")
        text.append("\n每次最多 256 个候选，同时用 Full-Q 独立计算该批全部分数；不保存完整 ledger。组件计时含逐候选 oracle 开销，不能当成生产 fit 加速比。")
    real=read("results/v02/real_accepted/validation.json") or []
    if real:
        text += ["\n## 普通 EEG 的低内存代价\n",
        "以下为 Windows 最终源码的单次 trace=True 科学验证 wall，包含 trace 构建，非预热后重复性能基准。"
        "同一记录两边保留相同科学参数；不可与 WSL trace=False 表拼接倍率。\n",
        "|EEG|Full-Q/Prefix2D s|raw-prefix/moments s|\n|---|---:|---:|"]
        for index in range(4):
            group={r["mode"]:r for r in real if r["record"]==index and "wall_s" in r}
            if len(group)==2:
                text.append(f"|{index}|{group['full']['wall_s']:.3f}|{group['low-memory']['wall_s']:.3f}|")
        text.append("\n低内存在这些普通高 K EEG 上更慢：反复重建状态矩及 prefix 区间评分仍有成本。其价值是扩展内存受限长 N，不能推荐为普通 EEG 默认。增量 TD 尚未在生产拟合中解决这一代价。")
    cold=read("results/v02/accepted/suite/cold.json")
    text += ["\n## 冷启动与精度\n"]
    if cold:
        text += ["独立 subprocess，隔离 Numba/Triton JIT cache；包含 import、生成 Small 输入和完整 fit；每项仅一次，非重复中位数。CUDA driver/OS 缓存未清除。\n",
        "|后端|mode|进程 wall s|\n|---|---|---:|"]
        for r in cold: text.append(f"|{r['backend']}|{r['mode']}|{r['process_wall_s']:.3f}|")
    precision=read("results/v02/accepted/suite/precision.json")
    if precision:
        text += ["\n|EEG|后端|FP32 all_bounds 相同|FP32 严格数值通过|最大 TD 差|\n|---|---|---|---|---:|"]
        for r in precision:
            v=r["report"]
            text.append(f"|{r['index']}|{r['backend']}|{v['bounds_equal']}|{v['passed']}|{v['max_t_error']:.3g}|")
    text += ["\nFP64 是默认。FP32 仅额外实验，resident kernel 明确拒绝 FP32。"
    "没有启用 TF32，未将最终边界相同当成 FP32 全数值等价。FP32 中的原表达式回退保留 FP64 Z，"
    "有专门回归测试，未因新增显式 Z 接口而把原回退降低精度。CPU 的 FP32 只指 interval Q，TD 仍为 FP64；"
    "GPU FP32 的 Q/R/prefix 均为 FP32，两者不是同一混合精度路线。\n",
    "## 阶段验收与建议合并范围\n",
    "|阶段|交付与边界|\n|---|---|",
    "|P0|最新 main/标签审计；原实现源码不覆盖；两份外部材料阅读及 hash；修改前基线和 profiling|",
    "|P1/P2|streaming argmax、window refinement、active-block cache，完整科学参数和逐移动 trace 验收；建议合并为普通 EEG 默认优化|",
    "|P3/P4|无 Q bounded chunk + moments TD；真实 EEG 完整 fit 和 Long-N 受限 fit；保留显式 experimental，近退化失败关闭|",
    "|P5|单/双切及远距离移动操作级统计验证；未整合完整 fit，不宣称完整加速；实验模块|",
    "|P6|实际 CUDA bounded resident summary；显式选择性集成；无 Q GPU/GPU TD/GPU finetune 未实现|",
    "|P7|CPU workers 与 GPU batch 吞吐测量；不引入证据不足的 auto；不以神经结果选择计算路线|",
    "|P8|逐阶段提交、公开证据、Draft PR；由用户审核后决定合并，未自动 merge|",
    "\n普通 EEG 优先合并 P1/P2；P3/P4 保持 opt-in，P5 保持组件实验，P6 选择性 GPU 保持 opt-in。"
    "新增维护成本主要是两套 reducer、低内存数值失败域和可选 Triton kernel；保留物化函数和"
    "Prefix2D 可以单独回退和审查。没有修改其他研究仓库或私有数据。\n",
    "## 重现\n",
    "从本分支 CPU-only `pip install -e '.[test]'` 后运行 `pytest tests -q`；"
    "私有 fixture 不在仓库，需设置 `HPGSBS_FIXTURES` 指向已有冻结目录。"
    "CUDA 在已记录的 WSL 环境运行。所有 v02 benchmark 要求新的输出目录，禁止覆盖证据。\n",
    "```bash\nPYTHONPATH=src python benchmarks/v02_paired.py --fixtures /path/to/frozen/fixtures --output results/v02/new_pair\n"
    "PYTHONPATH=src python benchmarks/v02_validate.py --long --output results/v02/new_long\n"
    "PYTHONPATH=src python benchmarks/v02_validate.py --fixtures /path/to/frozen/fixtures --output results/v02/new_real\n"
    "PYTHONPATH=src python benchmarks/v02_suite.py --fixtures /path/to/frozen/fixtures --output results/v02/new_suite\n```\n"]
    Path("docs/v02_results_CN.md").write_text("\n".join(text),encoding="utf-8")


if __name__=="__main__": main()
