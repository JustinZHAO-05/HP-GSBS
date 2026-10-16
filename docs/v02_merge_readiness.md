> 本页是合并前科学/调度验收历史。下文未获发布授权、未合并等表述记录当时状态；本轮用户已另行授权正式发布，最新状态见 [release verification](release_verification_v0.2.0.md)。

# HP-GSBS PR #1：v0.2 merge readiness

结论：**Approve（在下述成熟度与验证域内建议合并）**。仅准备审查；未执行 merge、tag、release 或 PyPI 发布。

[源码 CI（9baff92，四矩阵成功）](https://github.com/JustinZHAO-05/HP-GSBS/actions/runs/38886420614) · [源码 diff 审查入口](https://github.com/JustinZHAO-05/HP-GSBS/pull/1/files)。最终交付 HEAD 的 CI 在 PR 描述中更新。

原科学/性能验收的算法、tests、计时 benchmark 冻结源码：`9baff923fea285d448fb3078c82943f365369b3b`；原最终交付为 `002bf2a`。下面新增的调度修复另有源码与证据身份，旧 manifest 和历史测量未覆盖或重标。最终交付 HEAD 和其 CI 链接另记录在 PR 描述，避免文档自引用 SHA。

[PR #1](https://github.com/JustinZHAO-05/HP-GSBS/pull/1) · [问题登记表](v02_merge_audit.md) · [API/预算/成熟度](v02_api_scope.md) · [原性能验收 manifest](../results/v02/merge_9baff92/manifest.json)

## 调度补充验收：C22–C24

修复及测试源码：`7da26110e040820150412d6f47e7d1930926e26c`。
[独立补充 manifest](../results/v02/scheduler_7da2611/manifest.json) 绑定本轮源码和两平台 JUnit；
[源码 CI 四矩阵通过](https://github.com/JustinZHAO-05/HP-GSBS/actions/runs/38918277397)。
证据核验工具在后续交付中增加本轮 scope 支持；库与测试仍为上述快照。

|平台|passed|skipped|failure/error|
|---|---:|---:|---:|
|Windows|315|3|0/0|
|WSL，实际 CUDA/Triton 和四条完整冻结 EEG|380|1|0/0|

新增 27 项跨平台测试与 4 项实际 GPU 测试全部通过：24 个恢复组合覆盖四后端的旧签名、
旧软件版本、failed、cancelled、缺失状态和成功状态；原文件逐字节及 mtime 保持。
另覆盖配置默认值规范化、版本/计算策略失效、混合策略分桶的两种输入顺序。
实际 Torch/Triton 各两种顺序、每组六条记录，latency batch=1，throughput 按各自比例分组，
所有输出与未修改官方 reference 对照通过。显存比例的具体规划调用也在无 CUDA 的回归中验证。

恢复策略：签名 v1 包含软件版本、完整默认配置及输入身份。旧签名返回明确 failed receipt，
可加载作历史分析；执行当前实现需新输出目录，不自动复用或覆盖。CPU/GPU 共用 success 检查。
相同开发版本内任意源码修改不承诺自动失效；输入仍依赖路径/size/mtime，非内容寻址。

仅两个库文件变化：batch/scheduler.py、batch/bucketing.py；core、backends、config、API 与原验收完全一致，
补充核验工具验证这一边界和历史证据文件 hash。此次重跑全量正确性测试，未重跑旧性能矩阵；
下文耗时和加速比仍专属 9baff92 测量，不声称为 7da2611 新测量，也不外推混合配置吞吐收益。

```bash
python benchmarks/v02_verify_delivery.py --manifest results/v02/scheduler_7da2611/manifest.json
```

## 科学与工程验收

本节及后续性能表保留原 9baff92 验收记录；最新调度与全量测试以此前补充节为准。

reference 为未修改 statesegmentation 0.0.6。保留原科学参数、候选 gate、严格 >、NaN、未评分 -1、双切 K/birth、顺序 refinement、完整 tdists 与自定义 K 选择。输入为原冻结 EEG，只读快照；未改预处理、BBBD 数据或统计结果。

|平台|tests|passed|skipped|failure/error|
|---|---:|---:|---:|---:|
|windows|291|288|3|0/0|
|wsl|350|349|1|0/0|

Windows skip 为 CUDA 依赖模块和可选外部审计 checkout；WSL skip 为可选外部审计 checkout。实际 WSL CUDA/Triton 和四条私有完整 EEG 均运行。CLI/API、旧结果加载、save/load、manifest resume、实际中断恢复、注入 OOM halving 和 fresh-process Numba cache 回归均包含在 tests 中。

本轮真实输入验收 24 项全部 passed：四条 full64/frontal32、不同 session/video 的完整冻结路径，CPU/Torch/Triton FP64 和 CPU low-memory；另四条 dev96 各两种模式与新鲜未修改官方逐移动 trace 严格比较。

|完整记录|模式|backend|wall s（非 warm latency）|最大 TD 差|
|---|---|---|---:|---:|
|0|full|cpu-fast|3.655|2.43e-13|
|0|full|torch-cuda|6.684|9.59e-14|
|0|full|triton-cuda|2.480|9.59e-14|
|0|low-memory|cpu-fast|35.769|8.88e-15|
|1|full|cpu-fast|2.550|0|
|1|full|torch-cuda|1.986|3.13e-13|
|1|full|triton-cuda|1.287|3.13e-13|
|1|low-memory|cpu-fast|26.941|3.06e-13|
|2|full|cpu-fast|3.734|0|
|2|full|torch-cuda|2.599|2.77e-13|
|2|full|triton-cuda|2.759|2.77e-13|
|2|low-memory|cpu-fast|55.940|7.85e-13|
|3|full|cpu-fast|2.154|0|
|3|full|torch-cuda|1.863|1.85e-13|
|3|full|triton-cuda|1.457|1.85e-13|
|3|low-memory|cpu-fast|33.183|1.36e-13|

完整旧 fixture 缺 after-insertion/per-move 字段，因此 full-search-trace 通过不等于完整官方逐移动验证；报告明确 moves_compared=false。dev96 新鲜官方观察包含 deleted/window/winner/birth/after_delete/after_insert。原版 full EEG 的全部逐移动轨迹仍未重新计算，未把缺失字段伪装成通过。七网络源空间输入在提供的冻结集内不可用；未生成源定位。

## 单次性能：真实源码快照交错比较

相同解释器/硬件、X hash、K/statewise/blocksize/fine/dmin、FP64、TF32 off、BLAS/Numba=1。三个独立常驻子进程分别载入真实 v0.1.0、9309874、最终源码；每 workload/backend 轮转顺序，2 次 warmup 后 3 次正式样本；没有并行 fit。共 225 次，全部 bounds/K/TD/patterns/strengths 对照通过。RSS 为进程当前值，峰值另见 paired/long/throughput。

|workload|backend|v0.1.0 s|old accepted s|final s|v0.1/final|old/final|
|---|---|---:|---:|---:|---:|---:|
|Small|cpu-fast|0.1235|0.1068|0.1060|1.16×|1.01×|
|Small|torch-cuda|0.0905|0.0721|0.0745|1.21×|0.97×|
|Small|triton-cuda|0.0850|0.0643|0.0577|1.47×|1.11×|
|Medium|cpu-fast|0.4231|0.3602|0.3594|1.18×|1.00×|
|Medium|torch-cuda|0.2027|0.1271|0.1294|1.57×|0.98×|
|Medium|triton-cuda|0.2205|0.1410|0.1417|1.56×|0.99×|
|Large|cpu-fast|1.2682|1.0707|0.9342|1.36×|1.15×|
|Large|torch-cuda|0.4950|0.2307|0.2209|2.24×|1.04×|
|Large|triton-cuda|0.5057|0.2519|0.2572|1.97×|0.98×|
|XL|cpu-fast|3.9172|2.1357|2.1516|1.82×|0.99×|
|XL|torch-cuda|2.1961|0.6992|0.3596|6.11×|1.94×|
|XL|triton-cuda|2.1311|0.7758|0.5592|3.81×|1.39×|
|BBBD0|cpu-fast|3.0056|1.4829|1.5711|1.91×|0.94×|
|BBBD0|torch-cuda|2.3270|0.9063|0.7896|2.95×|1.15×|
|BBBD0|triton-cuda|2.3370|0.9225|0.8193|2.85×|1.13×|

Small=450×16/K20，Medium=900×32/K30，Large=1500×64/K40，XL=2100×64/K50，BBBD0=1336×64/K131；BBBD0 dmin=5，其余 dmin=1，均 statewise=True/fine=1/blocksize=50。不同 session/trace/precision 的数据不拼接倍率。旧 materialized 保留可比实现，新的 budget/preflight 本身也有成本。

## Low-memory 与 Long-N

raw X/Z 双前缀、bounded CP scoring、resumable fixed buffer、C-order reducer 分离。没有全 candidate ledger、candidate×T×P 张量或低内存 dense fallback。候选计数 O(T+K+gate)，但双切仍 Θ(T²)，每候选 O(P)。普通高 K EEG 更慢，继续推荐 Full-Q。短带与行范数复用有严格存储上限，超过上限仍流式。

|shape|restricted fit median s|peak RSS MiB|selected K|最大 fixed-partition TD error|
|---|---:|---:|---:|---:|
|[5000, 8]|0.1077|191.1|16|1.78e-15|
|[10000, 16]|0.1843|204.5|16|8.88e-16|
|[20000, 32]|0.3745|247.5|16|8.88e-16|
|[50000, 64]|1.4439|469.5|15|1.15e-14|

以上为 K16、single、fine=0、block=0、dmin=1 的完整受限 fit，非全量 long-N 官方搜索认证。selected/final 划分单独用两遍 tiled pair oracle 验收。仅该验证脚本显式放宽 oracle 到 3e9 pair evaluations，T≤50k 时两遍总计严格低于 2.5e9；生产 fallback 默认仍为 50m。未在 50k 物化 Q/R，也未穷举 50k 初始双切。默认候选上限 50m 会明确拒绝该域。

5k×8 与 1200×32 首次双切全候选组件有独立 Full-Q per-candidate oracle；仅 bounded chunk，记录全部候选计数/最大误差/尾批，不能把组件审计时间当完整 fit 加速。Memory fresh-process A/B 在保守 RAM admission 允许时运行；所有拒绝和预算上限有明确记录。

## 数值域与增量研究

moments 六矩实数恒等通过 Fraction、独立 pairwise、Prefix2D 验证；FP64 方差相减不是误差证明。两遍回退先预算实际 W/A pair 数，近退化仍 fail-closed，不 clamp variance，不物化 N²。数值研究分别记录 native NumPy、FP64 U 高精度 pair、原始 FP64 X 的 longdouble 目标和 moments；longdouble 不是区间包络或官方轨迹证书。common-anchor 可降低矩消去，但不能消除 native BLAS trajectory 的不确定性，因此未擅自放宽 near-degenerate 拒绝。

|完整 controller 原型|direct moments s|incremental s|direct/incremental|strict move parity|
|---|---:|---:|---:|---|
|small|0.4214|0.3942|1.07×|passed|
|BBBD0|36.7490|34.9397|1.05×|passed|

增量模块确实进入过完整 CPU controller 原型：single/double 纯 proposal，finally rollback；选择并完成顺序微调后 commit；每 32 个接受划分从当前区域重建。删除/插入/远移/外侧邻接迁移、160 次反向轨迹和周期=2 的完整轨迹均验证。只保留 active interval/adjacency，set 查失效，不保留被拒候选。

仍标 research-only、不开放 GSBSConfig：完整高 K 性能只覆盖一条 EEG，复杂移动的误差保障仍为工程检测；尚无多 recording、独立 Y 完整增量验收与稳健吞吐收益。即使个别场景更快，也不自动成为正式分析 backend。晋级需扩大这些域，不能把原型报告等同 production 发布。

## GPU resident 与低内存原型

resident 保留 device 坐标生成/有限 program summaries/稳定最早 flat/NaN、Inf、runner、gate 语义。新增跨 program、grid stride、host-device mismatch/元数据范围测试。内部 HostQ 来自同次预计算并只读，组件默认全面核验 host/device；计时的 trusted 内部路径不重复拷贝整张 Q。

|workload|backend|CPU streaming controller s|selective resident s|stream/resident|
|---|---|---:|---:|---:|
|Small|torch-cuda|0.0801|0.0772|1.04×|
|Small|triton-cuda|0.0638|0.0671|0.95×|
|Medium|torch-cuda|0.1222|0.1364|0.90×|
|Medium|triton-cuda|0.1582|0.1709|0.93×|
|Large|torch-cuda|0.1925|0.1978|0.97×|
|Large|triton-cuda|0.2261|0.2350|0.96×|
|XL|torch-cuda|0.3836|0.3952|0.97×|
|XL|triton-cuda|0.4153|0.4282|0.97×|
|BBBD0|torch-cuda|1.0726|1.1162|0.96×|
|BBBD0|triton-cuda|1.1140|1.1123|1.00×|

阈值 resident_min_candidates=20k 为显式设备相关 opt-in 参数；无 gate 大双切才走 resident，单切/其余 gate/顺序 finetune 仍 CPU。未根据本设备测量发布通用 auto。组件表包含元数据 H2D、launch、summary D2H/sync；完整 fit 表包含预计算，不能用 kernel 时间替代。profiler 原件独立归档，主树保留表与 metadata。

no-Q GPU prefix、GPU fixed-partition moments 原型实际运行；记录 CP、输出 transfer 和 launch 成本，但没有完整 stable reducer/guard/TD/refinement 集成。GPU incremental 与 GPU finetune 延期，保留 CPU 顺序语义；组件正确/快不能直接推导完整 fit 科学等价或加速。

## 批量、冷启动与精度

吞吐测 Small、真实形状高 K、不同 T/P，CPU 1/2/4/8 workers 与 GPU B=1/2/4/8/16/32/64。每配置 1 次 warmup + 3 次，计时含建池/序列化/同步；逐记录输出核对在计时结束后执行。Generator 实际 yield 时间单独记录，wall/B 仅称摊销时间。RSS 为进程树采样加总（共享页可能重复），VRAM allocated/reserved 单独记录。大 batch admission 拒绝是硬件边界，不是完成测量。

|workload|backend|workers/B|fits/min median|batch wall s|median last completion s|peak RSS MiB|
|---|---|---|---:|---:|---:|---:|
|Small|cpu-fast|1/32|274.3|7.000|7.000|683.6|
|Small|cpu-fast|2/32|839.6|2.287|2.254|1467.5|
|Small|cpu-fast|4/32|1362.2|1.410|1.379|2260.2|
|Small|cpu-fast|8/32|1766.8|1.087|1.046|3825.5|
|Small|torch-cuda|1/1|646.7|0.093|0.093|1129.1|
|Small|torch-cuda|1/2|942.2|0.127|0.127|1135.4|
|Small|torch-cuda|1/4|1342.0|0.179|0.179|1147.8|
|Small|torch-cuda|1/8|1418.4|0.338|0.338|1203.8|
|Small|torch-cuda|1/16|1554.3|0.618|0.617|1253.2|
|Small|torch-cuda|1/32|1459.9|1.315|1.290|1352.6|
|Small|torch-cuda|1/64|1399.3|2.744|2.695|1452.1|
|Small|triton-cuda|1/1|658.6|0.091|0.091|1304.3|
|Small|triton-cuda|1/2|782.6|0.153|0.153|1304.4|
|Small|triton-cuda|1/4|904.5|0.265|0.265|1203.1|
|Small|triton-cuda|1/8|1451.5|0.331|0.330|1240.3|
|Small|triton-cuda|1/16|1491.8|0.644|0.643|1315.0|
|Small|triton-cuda|1/32|1529.9|1.255|1.243|1364.7|
|Small|triton-cuda|1/64|1340.3|2.865|2.797|1613.0|
|highK|cpu-fast|1/8|24.4|19.656|19.656|1358.8|
|highK|cpu-fast|2/8|74.3|6.460|6.417|3052.4|
|highK|cpu-fast|4/8|125.7|3.819|3.761|4858.2|
|highK|cpu-fast|8/8|154.9|3.098|3.042|8458.3|
|highK|torch-cuda|1/1|42.9|1.398|1.398|1333.6|
|highK|torch-cuda|1/2|41.4|2.900|2.887|1361.0|
|highK|torch-cuda|1/4|40.5|5.929|5.900|1429.2|
|highK|torch-cuda|1/8|41.2|11.640|11.556|1647.4|
|highK|triton-cuda|1/1|41.9|1.432|1.431|1347.2|
|highK|triton-cuda|1/2|38.8|3.090|3.090|1374.6|
|highK|triton-cuda|1/4|44.6|5.382|5.358|1483.8|
|highK|triton-cuda|1/8|43.1|11.136|11.059|1702.0|
|variable|cpu-fast|1/32|307.9|6.235|6.235|1402.0|
|variable|cpu-fast|2/32|837.6|2.292|2.250|3290.5|
|variable|cpu-fast|4/32|1194.8|1.607|1.552|5178.9|
|variable|cpu-fast|8/32|1401.7|1.370|1.278|8955.9|
|variable|torch-cuda|1/1|584.3|0.103|0.102|1402.5|
|variable|torch-cuda|1/2|613.9|0.195|0.195|1402.5|
|variable|torch-cuda|1/4|613.0|0.392|0.391|1402.5|
|variable|torch-cuda|1/8|680.2|0.706|0.705|1402.6|
|variable|torch-cuda|1/16|900.9|1.066|1.065|1402.7|
|variable|torch-cuda|1/32|1101.3|1.743|1.743|1402.7|
|variable|torch-cuda|1/64|1266.4|3.032|3.032|1402.7|
|variable|triton-cuda|1/1|704.7|0.085|0.085|1402.7|
|variable|triton-cuda|1/2|642.6|0.187|0.186|1436.4|
|variable|triton-cuda|1/4|564.8|0.425|0.425|1436.5|
|variable|triton-cuda|1/8|707.8|0.678|0.678|1436.5|
|variable|triton-cuda|1/16|868.3|1.106|1.105|1436.5|
|variable|triton-cuda|1/32|1016.3|1.889|1.889|1436.5|
|variable|triton-cuda|1/64|1297.7|2.959|2.959|1436.5|

准入拒绝 6 个大 batch 配置，明细保留。每种 workload 均运行 CPU/Torch/Triton 文件 manifest 完成与 resume，以及数组 fit_many API。OOM retry 是真实 CUDA 环境内的注入回归测试，未故意耗尽整个设备制造 OOM。没有据单次结果设通用最优 worker/batch。

独立 cold 脚本隔离 Numba/Triton cache，含 import+Small 完整 fit；每项只有一次，不是中位数，不清 OS/driver cache。新增 reducer/count/fill 编译有冷启动成本。

|record|FP32 backend|bounds same|strict numeric pass|max TD error|
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

FP32 保持 experimental；CPU Q32/TD64 与 GPU Q/R/prefix32 不同，原表达式/TD exact 回退保持 FP64，TF32 未启用。负结果不隐藏。

## 公开仓库与版本

0.2.0.dev0 同步 pyproject、__version__、README、CITATION 与测试；不创建正式 release。v0.1.0 tag 仍为 922a69c76ba9c39964efd3c3a7f973625ae5b1f1。原 9309874 evidence 在本地 ZIP 完成 CRC/逐文件 hash 后精简，并有 public Git 固定 commit 检索；旧报告明确历史身份，不冒充本轮数字。

新证据原件先存私有持久审计目录，再对公开副本删本地路径/hostname。重复大数组（含真实 EEG patterns）不进入公开 revision trials；归档原件 hash 与公开变换 receipt 保留，比较结论与误差数值不变。原 profiler chrome traces 独立归档，保留 summary 表。旧历史中的本地路径不会因删除当前文件消失，未 force rewrite；未发现需要轮换的凭据或新公开私有原始 EEG/NPZ/subject manifest。

## Merge gates 与决定

M0：默认科学/完整 EEG、reducer/回退、资源拒绝、版本/公开副本、CPU CI、实际 CUDA、CLI/save/load/resume、source hash、v0.1 tag 均须通过后才 Ready for Review。M1：experimental 在已测域内显式启用，未认证域拒绝，research-only 不开放正式配置。M2：默认对 v0.1 的收益见真实快照表；所有慢样本/冷启动/FP32 负结果与高 K low-memory 代价保留。M3：更新 PR 后只准备审查，不 merge。

默认 v0.2 在已冻结 BBBD 域可替换 v0.1，仍推荐先保存原结果/配置/hash 后复跑核对。Long-N 能力仅为上述受限 fit 和组件；resident 仅设备相关 opt-in，不能推荐全场景 GPU resident。整个 PR 可在此边界内合并；正式 v0.2.0 发布仍需用户另行批准，并完成批准后的 main 安装、CI、CUDA smoke、旧结果加载验证。

重现（私有 fixture 自备，不入库）：

```bash
PYTHONPATH=src HPGSBS_FIXTURES=/private/frozen pytest tests -q
PYTHONPATH=src python benchmarks/v02_verify_delivery.py --manifest results/v02/merge_9baff92/manifest.json
```

后续扩大：全部完整 EEG 的新鲜官方逐移动轨迹、source networks、增量独立 Y/多 recording 性能、GPU no-Q 全路径、严格数值误差界/interval certificate、跨 GPU 自动阈值。上述均未标记已完成。

已确认的保守 false refusal：一组 pair 方差近零、另一组正常时，native TD=24.4540385213，但 moments 拒绝认证。见最终证据 numeric_false_refusal.json。它不产生错误 success，但属于已知适用范围限制；更精细的 Welch 条件数/误差界保护延期，不把该输入称为数学退化或完整支持。

## Maturity matrix

|模块|归属|适用范围/晋级条件|
|---|---|---|
|CPU/Torch/Triton，FP64 Full-Q + streaming + Prefix2D，window/cache|Stable|已冻结科学验证域；正式 BBBD 默认|
|raw-prefix/lazy Q、feature-moments|Experimental / opt-in|内存受限；预算/近退化拒绝；不保证高 K 更快|
|selective resident|Experimental / opt-in|实际 CUDA 通过；本机未见稳定完整 fit 收益，默认禁用|
|FP32|Experimental / opt-in|GPU 严格 TD 有差异，不能替代 FP64 科学验收|
|incremental full-controller prototype|Research-only|继续扩大多记录/Y/长轨迹数值与稳健性能验证后再开放 API|
|GPU no-Q prefix / fixed-partition moments|Research-only|组件有测量；完整 gate/reducer/guard/TD/finetune 仍缺|
|GPU incremental / GPU finetune 全流程|Disabled / Deferred|未实现；不假称 GPU resident 完成|

## 补充测量与剩余边界

正式吞吐表来自 throughput_fit_only；其 reproduce.py 在计时后校验输出，并记录实际 yield 时间。
原 throughput 数据含消费者验收开销，Small 全部及部分 high K 已测；发现检查主导 wall 后中止重复诊断。
jobs.json 保留 returncode=-2 和处置理由，原始数据/日志保留；没有把它改成成功的完整吞吐任务。
高 K 是同一冻结 EEG 的重复独立拟合任务，不是多个参与者。补充工具代码随证据保存并由 manifest hash 绑定。

增量表是 trace=True、raw-prefix、dmin=5 的完整 controller（不含 Q/TD 初始化）计时；不与默认 latency 混算。
额外 16 个 equal-nonzero-strength / repeated-cancellation 的 fine=0/1/2/-1、双排序严格移动轨迹全部通过。
workload_identities.json 和 throughput_inputs.json 记录输入身份、shape、参数与 hash，raw inputs 不发布。

默认 source 校验与 GitHub CPU CI 覆盖库代码；补充 profiler / fit-only 工具单独保存可审查源代码。
resident 额外 profiler 在 BBBD0 每 fit 18 次 GPU candidate 调用；事件表记录 H2D/D2H、aten::to 和同步。
元数据未复用：候选/微调后划分不断变化，当前收益不足以支持额外状态缓存协议；保留 active-block cache，未引入未经验证的 device-state reuse。
GPU raw-prefix 原型的 endpoints 由 CPU 提供；仅验证 interval score，不宣称已实现 no-Q device candidate winner。

|GPU prefix component T×P / C|CPU ms|GPU ms|最大 score 差|
|---|---:|---:|---:|
|[1200, 32] / 256|0.301|3.090|1.39e-17|
|[5000, 8] / 256|0.202|0.919|3.47e-18|
|[5000, 64] / 4096|2.634|1.128|6.94e-18|

包含 endpoint H2D、score D2H、同步，排除预先 prefix 上传。大 C/P 有组件收益，小 C 明显受 launch 主导；缺 gate/reducer/数值失败集成，未发布完整 GPU low-memory backend。GPU moments 仅 dmin=1 fixed-partition stats，最大统计差约9.1e-13，未认证完整 TD/搜索。

独立进程 5k×8 同科学配置 memory A/B：

|mode|warm median s|peak RSS MiB|
|---|---:|---:|
|materialized|9.7540|1598.0|
|low-memory|0.1104|190.6|

消融（完整 BBBD0，同组 warmup 后三次）：

|mode|wall median s|
|---|---:|
|materialized|3.3638|
|stream-only|1.8035|
|stream-window|1.6014|
|stream-window-cache|1.4288|

Cold 每项为一次独立过程，包含 import 和 JIT，不能当稳态中位数：

|backend|mode|process wall s|
|---|---|---:|
|cpu-fast|materialized|5.392|
|cpu-fast|streaming|5.414|
|torch-cuda|materialized|10.092|
|torch-cuda|streaming|10.119|
|triton-cuda|materialized|12.978|
|triton-cuda|streaming|13.462|

CPU BBBD0 集中复验：两次预热后各七次，轮换顺序，全部输出通过。旧 accepted=1.4245s，final=1.4496s，old/final=0.983。初组约6%差异不能单独当作确定回退；本轮增加了候选计数/预算保护，未声称这些保护本身加速。完整样本保留，未挑最好一次。

重跑纯拟合吞吐和 resident profiler（新 output 目录）：

```bash
PYTHONPATH=src python results/v02/merge_9baff92/throughput_fit_only/reproduce.py --fixtures /private/frozen --output results/v02/new_fit_only
PYTHONPATH=src python results/v02/merge_9baff92/profile/resident_reproduce.py --fixtures /private/frozen --output results/v02/new_profile
```
