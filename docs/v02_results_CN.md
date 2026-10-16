> 历史报告：对应修复前 source 5478fd6 / public commit 9309874。当前 0.2.0.dev0 验收见 v02_merge_readiness.md；本页数字不冒充本轮结果。

# v0.2 科学一致性与性能报告

日期：2026-10-16（北京时间）。本分支可独立审核；未合并 main，未改写 v0.1.0。

算法、复杂度、失效条件和配置见 [逐项算法审查](v02_algorithm_review_CN.md)。

## 配对完整拟合

同一 WSL2 / Ryzen 7 7840H / RTX 4050 Laptop 6 GB，Python 3.10、NumPy 2.2.6、PyTorch 2.11.0+cu128、Triton 3.6.0。FP64，BLAS/Numba 各 1 线程，TF32 关闭，trace=False。每格 warmup 一次，随后三对交错运行，两个顺序轮换；本次没有并行运行其他 HP-GSBS 计时任务。桌面和系统频率仍可能波动，三次测量不代表普遍速度保证。

before 为保留的 v0.1 `full_scores` + 原物化 reducer，禁用新增局部缓存；after 为streaming + window refinement + active-block cache。Q/TD、科学参数、输入均相同。保留同一控制器的对照避免跨 OS/跨硬件混算。最新 main 与 v0.1 的源码/测试相同，另有修改前 HEAD 冻结测量。下表倍数为两个中位数之比，不是逐对倍数中位数。

|负载|后端|before s|after s|倍数|after warm 范围 s|
|---|---|---:|---:|---:|---:|
|Small|cpu-fast|0.1646|0.1389|1.19×|0.1358–0.1436|
|Small|torch-cuda|0.1099|0.0711|1.55×|0.0649–0.0765|
|Small|triton-cuda|0.1044|0.0644|1.62×|0.0580–0.0688|
|Medium|cpu-fast|0.4983|0.4109|1.21×|0.3805–0.4190|
|Medium|torch-cuda|0.2734|0.1252|2.18×|0.1168–0.1590|
|Medium|triton-cuda|0.2579|0.1486|1.74×|0.1481–0.1617|
|Large|cpu-fast|1.5252|1.1869|1.28×|1.0861–1.3204|
|Large|torch-cuda|0.4673|0.2001|2.34×|0.1933–0.2035|
|Large|triton-cuda|0.5141|0.2252|2.28×|0.2115–0.2547|
|XL|cpu-fast|3.9688|2.3483|1.69×|2.3083–2.4454|
|XL|torch-cuda|1.9767|0.3539|5.59×|0.3335–0.3632|
|XL|triton-cuda|2.0029|0.4044|4.95×|0.3882–0.5501|
|BBBD0|cpu-fast|3.8028|1.8409|2.07×|1.8409–1.8634|
|BBBD0|torch-cuda|2.8802|1.1960|2.41×|1.0673–1.2915|
|BBBD0|triton-cuda|3.0057|1.1797|2.55×|1.0587–1.2611|

### BBBD0 阶段与资源

|后端|模式|预计算 s|候选 s|微调 s|TD s|传输 s|RSS MiB 峰值|allocated/reserved MiB|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|cpu-fast|materialized|0.9173|1.1009|1.4087|0.3630|0.0000|1297.6|0.0/0.0|
|cpu-fast|streaming|0.9131|0.0715|0.5008|0.3224|0.0000|1297.6|0.0/0.0|
|torch-cuda|materialized|0.0745|1.0671|1.3051|0.3595|0.0356|1338.8|138.7/504.0|
|torch-cuda|streaming|0.2151|0.0769|0.5358|0.3284|0.0360|1325.1|138.7/504.0|
|triton-cuda|materialized|0.1649|1.0833|1.3361|0.3539|0.0357|1434.3|110.1/504.0|
|triton-cuda|streaming|0.1657|0.0779|0.5325|0.3336|0.0422|1434.3|110.1/504.0|

传输属于预计算的子集，不能重复相加。阶段是 wall 插桩，不是独占 kernel 时间。

120 次（含 warmup）完整输出对照全部通过；比 all_bounds、完整 tdists、K、patterns、strengths。计时不采集候选 trace，逐轮科学验证另列。Small/Medium/Large/XL 为 seed=20261016 的合成输入，T/P/K 分别 450/16/20、900/32/30、1500/64/40、2100/64/50；BBBD0 是 1336×64、Kmax=131、dmin=5，其余 workload dmin=1。均保留 statewise=True、finetune=1、blocksize=50。

原始 [交错 trials](https://github.com/JustinZHAO-05/HP-GSBS/blob/9309874678ae4e861711fd3bc4b96e332b5bd8ae/results/v02/accepted/paired/trials.json) 保存各次时间、完整阶段计时、RSS/VRAM 和验证。[修改前 Windows](https://github.com/JustinZHAO-05/HP-GSBS/blob/9309874678ae4e861711fd3bc4b96e332b5bd8ae/results/v02/baseline_windows/latency.json)、[修改前 WSL](https://github.com/JustinZHAO-05/HP-GSBS/blob/9309874678ae4e861711fd3bc4b96e332b5bd8ae/results/v02/baseline_wsl_run1/latency.json) 与早期 streaming 测量保留；它们部分与其他验证并行，故不作为最终加速比。RSS 为 5 ms 采样进程总量，配对进程保留 allocator 高水位，不能据此声称每种模式的净节省。VRAM 在每次调用前同步并重置峰值。

## 科学验证与数据边界

Windows CPU-only：197 passed、3 skipped；WSL 含真实 CUDA/Triton：245 passed、1 skipped。完整测试输出和 JUnit 见 results/v02 下 accepted_tests 命名文件。测试数是参数化用例数，不是生物学样本数。4 条真实 EEG 在 Full-Q 与 raw-prefix/moments 路线逐轮通过，低内存最大 TD 误差约 7.86e-13；完整 all_bounds、单/双 winner、单双 TD、接受选择和微调最终边界相同。完整记录 0 直接对现有 official-full trace；1–3 对已通过官方/历史核验的 v0.1 缓存，本次没有重新运行其数小时官方完整 fit。4 条 dev96 则新跑未改动的官方程序并逐次比较每个微调删除/窗口/winner/birth。旧完整缓存缺少逐移动字段，不能冒称补齐了该证据。

[真实记录结果](https://github.com/JustinZHAO-05/HP-GSBS/blob/9309874678ae4e861711fd3bc4b96e332b5bd8ae/results/v02/real_accepted/validation.json) 仅含 hash、参数和比较结果，不含原始 EEG。私有文件通过 HPGSBS_FIXTURES 原位读取，未复制或上传。Fraction oracle 另外验证给定 binary64 U 的精确六矩与 dmin 分类；它不是 native BLAS运算轨迹证明。NaN/sentinel/tie、重复和跨行 stateopt、chunk 1/7/64、finetune -1/1/2、独立 y 和极大 dmin 均有独立测试。near-degenerate 低内存 TD 明确拒绝，而不是强制零方差。

## Long-N

以下为显式 statewise=False、finetune=0、blocksize=0、Kmax=16 的完整低内存拟合；不能外推为完整官方 statewise+finetune 长轨迹。候选用 raw-prefix bounded chunks，TD 用feature moments，所有输出 all_bounds 仍保存。预热后 3 次，同一进程不同尺寸顺序测量。

|T×P|warm 中位 s|进程峰值 RSS MiB|selected K|selected/final TD 最大审计差|
|---|---:|---:|---:|---:|
|5000×8|0.1306|201.9|16|1.78e-15|
|10000×16|0.2324|213.4|16|8.88e-16|
|20000×32|0.4633|247.8|16|8.88e-16|
|50000×64|1.6693|418.9|15|1.15e-14|

50,000×64 的一张 FP64 T² 数组理论为 20 GB；本次未尝试该规模的 dense baseline。selected K=15 与最终 K=16 均单独用 bounded tiled two-pass TD 核验，未只检查最终划分。这些 oracle 支持固定划分 TD，不认证长 N 每个候选的完整官方轨迹；没有同规模官方加速比。[原始 long-N 记录](https://github.com/JustinZHAO-05/HP-GSBS/blob/9309874678ae4e861711fd3bc4b96e332b5bd8ae/results/v02/accepted/long/validation.json)。

## GPU 常驻候选与集成范围

实际 CUDA 组件输入相同 resident FP64 Q，比较 Numba 与 Triton，包含元数据准备/传输、最后六项摘要 D2H 和同步；Q 构建/上传排除在两边组件计时之外。每 case 预热后 10 次，不是完整 fit 加速比。单切 launch 开销较大，大规模无 gate 双切可受益。

|T|状态数|搜索|Numba ms|resident ms|
|---|---:|---|---:|---:|
|450|1|single|0.013|1.376|
|450|1|double|0.477|1.335|
|450|5|single|0.012|1.295|
|450|5|double|0.105|1.967|
|1336|1|single|0.020|1.465|
|1336|1|double|4.362|1.661|
|1336|14|single|0.016|1.644|
|1336|14|double|0.350|1.931|
|2100|1|single|0.020|1.248|
|2100|1|double|11.523|2.990|
|2100|21|single|0.023|1.869|
|2100|21|double|0.982|2.969|

选择性 resident 已作为显式 experimental 配置接入同一 controller；只处理无 stateopt gate且双候选数至少 20,000 的调用。单切、其他双切、顺序微调继续走 CPU。集成的完整 fit和真实 EEG trace 与组件测试分别验收；并不代表全流程 GPU-resident。

|负载|预计算后端|selective resident 完整 fit s|相对默认 CPU streaming controller|
|---|---|---:|---:|
|Small|torch-cuda|0.0935|0.76×|
|Small|triton-cuda|0.0686|0.94×|
|Medium|torch-cuda|0.1373|0.91×|
|Medium|triton-cuda|0.1180|1.26×|
|Large|torch-cuda|0.2068|0.97×|
|Large|triton-cuda|0.2901|0.78×|
|XL|torch-cuda|0.4184|0.85×|
|XL|triton-cuda|0.4754|0.85×|
|BBBD0|torch-cuda|1.1215|1.07×|
|BBBD0|triton-cuda|1.1403|1.03×|

此表是后续同环境顺序测量，未与 streaming 再交错配对，不能据小幅差异选择新默认。

## 多记录吞吐

Small 450×16、K20，至多 64 条独立 seed=200+i 输入。CPU 固定 32 条，比较 1/2/4/8 workers；GPU 比 B=1/2/4/8/16/32，并在保守显存预算允许时测 B=64。包括建池、序列化、传输、同步和全部独立 controller，保存和输出核对不计时。RSS 为父进程加子进程总量的 10 ms 采样，包含共享页重复计数，不是精确物理 RAM。每 case 一次 warmup 后 3 次；所有输出检查 all_bounds/TD/K/patterns/strengths。

|后端|workers|B|mode|fits/min 中位|wall s 中位|RSS MiB 峰值|allocated/reserved MiB|
|---|---:|---:|---|---:|---:|---:|---:|
|cpu-fast|1|32|materialized|258.7|7.422|680.8|0.0/0.0|
|cpu-fast|1|32|streaming|343.7|5.587|681.2|0.0/0.0|
|cpu-fast|2|32|materialized|815.8|2.354|1473.6|0.0/0.0|
|cpu-fast|2|32|streaming|953.8|2.013|1474.2|0.0/0.0|
|cpu-fast|4|32|materialized|1266.1|1.516|2268.6|0.0/0.0|
|cpu-fast|4|32|streaming|1557.0|1.233|2267.4|0.0/0.0|
|cpu-fast|8|32|materialized|1628.5|1.179|3851.7|0.0/0.0|
|cpu-fast|8|32|streaming|1939.6|0.990|3849.4|0.0/0.0|
|torch-cuda|1|1|materialized|420.6|0.143|1126.2|22.7/42.0|
|torch-cuda|1|1|streaming|703.4|0.085|1130.9|22.7/42.0|
|torch-cuda|1|2|materialized|537.4|0.223|1140.3|35.6/62.0|
|torch-cuda|1|2|streaming|1076.3|0.111|1140.4|35.6/62.0|
|torch-cuda|1|4|materialized|679.8|0.353|1152.8|61.2/84.0|
|torch-cuda|1|4|streaming|1341.4|0.179|1159.0|61.2/84.0|
|torch-cuda|1|8|materialized|634.5|0.756|1183.9|112.6/156.0|
|torch-cuda|1|8|streaming|1504.7|0.319|1196.2|112.6/156.0|
|torch-cuda|1|16|materialized|646.4|1.485|1233.6|215.2/368.0|
|torch-cuda|1|16|streaming|1506.7|0.637|1208.7|215.2/368.0|
|torch-cuda|1|32|materialized|640.9|2.996|1283.4|424.2/768.0|
|torch-cuda|1|32|streaming|1456.0|1.319|1283.4|424.2/768.0|
|torch-cuda|1|64|materialized|592.9|6.476|1432.4|834.7/1568.0|
|torch-cuda|1|64|streaming|1427.5|2.690|1432.2|834.7/1568.0|
|triton-cuda|1|1|materialized|398.1|0.151|1192.4|19.5/1568.0|
|triton-cuda|1|1|streaming|590.9|0.102|1194.0|19.5/1568.0|
|triton-cuda|1|2|materialized|480.6|0.250|1194.0|29.2/1568.0|
|triton-cuda|1|2|streaming|764.0|0.157|1194.1|29.2/1568.0|
|triton-cuda|1|4|materialized|551.1|0.435|1206.6|48.4/1568.0|
|triton-cuda|1|4|streaming|1076.0|0.223|1206.6|48.4/1568.0|
|triton-cuda|1|8|materialized|599.3|0.801|1256.4|86.9/1568.0|
|triton-cuda|1|8|streaming|1556.7|0.308|1256.4|86.9/1568.0|
|triton-cuda|1|16|materialized|639.7|1.501|1305.9|163.8/1568.0|
|triton-cuda|1|16|streaming|1560.3|0.615|1306.0|163.8/1568.0|
|triton-cuda|1|32|materialized|632.0|3.038|1355.9|320.4/1568.0|
|triton-cuda|1|32|streaming|1421.0|1.351|1355.9|320.4/1568.0|
|triton-cuda|1|64|materialized|677.0|5.672|1505.2|627.6/1568.0|
|triton-cuda|1|64|streaming|1401.0|2.741|1505.4|627.6/1568.0|

wall 是整个 batch 调用延迟；JSON 的 average_latency_s 定义为摊销 wall/B，不是每记录的完成延迟。单条 latency 另见完整 fit 表。每批 transfer、GPU precompute wall、CPU controller sum、OOM 后的实际预计算批次另存原始 JSON。GPU precompute wall 包含 launch/sync，不能当作纯 kernel 时间；实际 kernel/launch/sync profiler 另存 accepted/suite/profile。

## 每项优化的消融

同一 BBBD0 完整参数，四种模式交错，warmup 后各三次。仅在 benchmark 进程中暂时禁用 HP-GSBS cache/window，finally 恢复；未修改官方类。所有模式对比完整输出。

|模式|完整 wall s|candidate s|finetune s|TD s|
|---|---:|---:|---:|---:|
|materialized|2.9001|0.9143|0.8886|0.3356|
|stream-only|1.7764|0.1963|0.4906|0.3149|
|stream-window|1.5738|0.1841|0.3091|0.3049|
|stream-window-cache|1.3740|0.0592|0.3031|0.2949|

单独优化的边际效果只依据相邻两行；不将多阶段倍率相乘。预计算同为 Full-Q + Prefix2D，阶段时间和 wall 的中位数不能机械相加。

## 新进程低内存 A/B

相同 5,000×8、K16、单切、无微调、blocksize=0；每种模式独立进程，warmup 后三次。dense 先经保守 RAM 预算检查；没有在 50,000 上分配 dense 数组。

|模式|完整 warm s|进程峰值 RSS MiB|
|---|---:|---:|
|materialized|10.4848|1596.8|
|low-memory|0.1379|201.5|

两条路径检查完整 all_bounds hash、TD 和 K。两边均采用同一受限科学配置。

## 长状态首次双切的全候选组件核验

|T×P|双候选总数|flush 次数|最大每候选分差|winner|
|---|---:|---:|---:|---|
|1200×32|718,201|2,806|2.78e-17|相同|
|5000×8|12,492,501|48,799|2.78e-17|相同|

每次最多 256 个候选，同时用 Full-Q 独立计算该批全部分数；不保存完整 ledger。组件计时含逐候选 oracle 开销，不能当成生产 fit 加速比。

## 普通 EEG 的低内存代价

以下为 Windows 最终源码的单次 trace=True 科学验证 wall，包含 trace 构建，非预热后重复性能基准。同一记录两边保留相同科学参数；不可与 WSL trace=False 表拼接倍率。

|EEG|Full-Q/Prefix2D s|raw-prefix/moments s|
|---|---:|---:|
|0|4.265|38.802|
|1|2.437|30.145|
|2|3.274|62.747|
|3|2.123|33.632|

低内存在这些普通高 K EEG 上更慢：反复重建状态矩及 prefix 区间评分仍有成本。其价值是扩展内存受限长 N，不能推荐为普通 EEG 默认。增量 TD 尚未在生产拟合中解决这一代价。

## 冷启动与精度

独立 subprocess，隔离 Numba/Triton JIT cache；包含 import、生成 Small 输入和完整 fit；每项仅一次，非重复中位数。CUDA driver/OS 缓存未清除。

|后端|mode|进程 wall s|
|---|---|---:|
|cpu-fast|materialized|3.986|
|cpu-fast|streaming|4.971|
|torch-cuda|materialized|8.862|
|torch-cuda|streaming|9.234|
|triton-cuda|materialized|11.623|
|triton-cuda|streaming|12.366|

|EEG|后端|FP32 all_bounds 相同|FP32 严格数值通过|最大 TD 差|
|---|---|---|---|---:|
|0|cpu-fast|True|True|0|
|0|torch-cuda|True|False|4.68e-05|
|0|triton-cuda|True|False|4.68e-05|
|1|cpu-fast|True|True|0|
|1|torch-cuda|True|False|0.00011|
|1|triton-cuda|True|False|0.00011|
|2|cpu-fast|True|True|0|
|2|torch-cuda|True|False|0.000354|
|2|triton-cuda|True|False|0.000354|
|3|cpu-fast|True|True|0|
|3|torch-cuda|True|False|4.02e-05|
|3|triton-cuda|True|False|4.02e-05|

FP64 是默认。FP32 仅额外实验，resident kernel 明确拒绝 FP32。没有启用 TF32，未将最终边界相同当成 FP32 全数值等价。FP32 中的原表达式回退保留 FP64 Z，有专门回归测试，未因新增显式 Z 接口而把原回退降低精度。CPU 的 FP32 只指 interval Q，TD 仍为 FP64；GPU FP32 的 Q/R/prefix 均为 FP32，两者不是同一混合精度路线。

## 阶段验收与建议合并范围

|阶段|交付与边界|
|---|---|
|P0|最新 main/标签审计；原实现源码不覆盖；两份外部材料阅读及 hash；修改前基线和 profiling|
|P1/P2|streaming argmax、window refinement、active-block cache，完整科学参数和逐移动 trace 验收；建议合并为普通 EEG 默认优化|
|P3/P4|无 Q bounded chunk + moments TD；真实 EEG 完整 fit 和 Long-N 受限 fit；保留显式 experimental，近退化失败关闭|
|P5|单/双切及远距离移动操作级统计验证；未整合完整 fit，不宣称完整加速；实验模块|
|P6|实际 CUDA bounded resident summary；显式选择性集成；无 Q GPU/GPU TD/GPU finetune 未实现|
|P7|CPU workers 与 GPU batch 吞吐测量；不引入证据不足的 auto；不以神经结果选择计算路线|
|P8|逐阶段提交、公开证据、Draft PR；由用户审核后决定合并，未自动 merge|

普通 EEG 优先合并 P1/P2；P3/P4 保持 opt-in，P5 保持组件实验，P6 选择性 GPU 保持 opt-in。新增维护成本主要是两套 reducer、低内存数值失败域和可选 Triton kernel；保留物化函数和Prefix2D 可以单独回退和审查。没有修改其他研究仓库或私有数据。

## 重现

从本分支 CPU-only `pip install -e '.[test]'` 后运行 `pytest tests -q`；私有 fixture 不在仓库，需设置 `HPGSBS_FIXTURES` 指向已有冻结目录。CUDA 在已记录的 WSL 环境运行。所有 v02 benchmark 要求新的输出目录，禁止覆盖证据。

```bash
PYTHONPATH=src python benchmarks/v02_paired.py --fixtures /path/to/frozen/fixtures --output results/v02/new_pair
PYTHONPATH=src python benchmarks/v02_validate.py --long --output results/v02/new_long
PYTHONPATH=src python benchmarks/v02_validate.py --fixtures /path/to/frozen/fixtures --output results/v02/new_real
PYTHONPATH=src python benchmarks/v02_suite.py --fixtures /path/to/frozen/fixtures --output results/v02/new_suite
```
