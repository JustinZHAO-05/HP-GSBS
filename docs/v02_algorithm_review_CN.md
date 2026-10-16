# v0.2 算法审查与接入边界

本分支从 2026-10-16 获取的 `origin/main=f85ea0b` 开始，保留 v0.1.0
标签、冻结 fixture、历史 benchmark 和 reference。v0.1.0 与该 main 的
`src/` 和 `tests/` 没有差异。正式科学 reference 仍为未改动的
`statesegmentation==0.0.6`，不是外部优化笔记。

## 外部材料审查

完整阅读两份用户提供的技术材料：

|材料|SHA256|
|---|---|
|GSBS_CPU_FAST_ALGORITHM_CN.md|22f684f4637d8a08fefe22a2719a777f4d8ca0f153e3f1fa7f593b375fb29482|
|GSBS_GPU_FAST_ALGORITHM_CN.md|f245faba4dc38e2e29f11905ffa0aaded2054884e5887e0e09c506a480c0d05e|

可访问附件目录没有材料列出的 B5/B6/B7/B8、doubleprefix 或 Torch 原始
ZIP/源码证据包。因此没有独立复跑其历史性能数字，也没有复制这些实现。
它们的性能表只作为研究背景，不参与本分支的加速比。以下实现与测试均从
现有 HP-GSBS 和安装的官方源码独立完成。外部冻结源码 hash 与本库实际
官方版本不能混为一谈；每次结果仍保留 `upstream.source_sha256`。

## 候选评分与归约

原 `full_scores()` 保留。新增 `stream_q()` 对同一候选顺序和同一表达式
边算边归约。未评分 flat=0 的分数为 -1；严格大于才替换，精确并列保留
最早 flat，首个 NaN 优先于所有数值，出现非 -1 与“存在候选”分别记录。
全部分数等于 -1 时 double 返回 None；分数全部小于 -1 时 sentinel
仍可胜出。NaN winner 的 runner 诊断定义为 NaN，避免把旧 runner 误作
有效分差。没有 nanargmax、epsilon 或新的候选剪枝。

频数 gate 独立保留 `A[i]+B[0]>=1`、`A[i]+B[j]>=2`，包括重复首列、
跨行满足条件、None 与空 `(0,2)` 的区别。CPU 单元测试用独立 NumPy
矩阵比较任意分数、负分、Inf、NaN、tie；完整拟合再与官方比较。

Full-Q 的候选评分工作仍为 O(T+ΣL²)，每候选 O(1)。流式归约的额外
工作区是 O(T+K)，不再分配 double 的 block_length² 分数矩阵。
完整 Q 和 Prefix2D TD 仍然有二次存储，不能称完整默认路线为线性内存。
near tie 或非有限值触发原 means/zscore 表达式重评；现在候选逐块重访，
不保存 near-candidate 列表，也不因数值异常物化完整双切矩阵。
容差仍是原工程保护，未宣称任意输入上的严格误差界。

在 Windows 新进程回归中发现 Numba 0.68 的 cached generator lowering
注册缺失。Full-Q reducer 因此融合相同候选枚举循环，不依赖已缓存的
generator；供 Python bounded chunks 消费的 generator 禁用磁盘缓存。
新增隔离 cache 的 subprocess 测试按 lazy→full FP64→full FP32 启动
三个独立进程。原失败日志保留；最终性能表在此修复后重新测量。

## 微调与缓存

先删除当前边界，再对临时完整划分计算未变化状态贡献和实际目标父段的
两个新区间；`stream_window()` 只访问真实窗口内合法位置。负 finetune
允许目标父段远离被删除边界，未套用仅适合原左右邻居的公式。
原 strengths、排序、窗口端点、顺序、birth 和 all_bounds 索引全部保留。

官方 observer 只用 `sys.settrace` 观察未改动的方法。新 trace 在每次
微调记录 deleted、window、winner、birth、after_delete；再结合 winner
和 birth 可重建回插状态，after 记录整轮最终边界。保存格式向后兼容；
旧 trace 缺少新字段时比较器不会伪造缺失的逐移动证据。

局部 cache 在一次同步 fit 的固定 X/Q/精度生命周期内使用。键为
`(block start, block end, all internal cuts, statewise)`，只保留当前有效
block。分裂、合并、移动或分组变化都会改变键并失效。未缓存全局 winner，
未跨输入/fit/精度复用。每轮仍重建 boundopt/stateopt 并全局重评。
输入在同步计算期间必须保持不变。物化兼容路径禁用此 cache，可用于 A/B。
未声称此缓存使含双切和微调的完整 GSBS 达到 N log N。

## 双前缀与 bounded chunk

`q_strategy='streaming'` 保存 raw-X 前缀和实际 Z 前缀；支持显式传入 Z。
区间 raw sum 重新去空间均值后，与区间 Z sum 点积并除以
`T sqrt(P-1) ||centered raw sum||`。从原模板逐行 Pearson 相关求和，
区间长度在模板空间标准化中消掉，得到该式。只存 normalized Gram
会丢失 raw 幅度信息，因而没有采用。

候选生成、chunk scoring、summary reduction 独立；每批最多 C 个端点
四元组和 C×P 特征临时量，未保存候选 ledger。保留现有 centered-prefix
`lazy` 路径作为兼容选项。raw-prefix 与 centered-prefix 的浮点轨迹不同，
分别通过完整 trace 验收，不将实数恒等写成逐位相同。

主要空间 O(TP+T+K+CP)。双切仍可有 Θ(T²) 个候选；去掉分数矩阵没有
让完整枚举变成线性时间。Kmax×T 的 all_bounds 输出空间还须另计。

## Feature-moments TD 的独立推导

V 取独立 y 或 x。U 为每行 ddof=1 zscore 后再去残余行均值，h=Ptd-1。
相关为 `r_ij=U_i dot U_j/h`。展开 `||ΣU||²`，删除对角并除以 2，得到
within 的 sum；展开 `||ΣUUᵀ||F²` 得到相关平方和，删除 `Σ||U||⁴` 后
同样除以 2。不同相邻状态的 Cartesian 积没有重复和对角，不除以 2。
非相邻状态不参与。对 `0<j-i<dmin` 且类别为 W/A 的 pair 删除其
`(1,r,r²)`，不是删除全部短带后再忽略状态类别。

Fraction 测试直接验证六矩恒等，普通浮点测试独立比较 Prefix2D/官方
covariance。固定划分采用当前/前一状态矩，空间 O(TP+P²)，主要运算
O(TP²+KP²+T dmin P)。没有 T×P×P 二阶前缀。

Welch 使用 unbiased variance；within<2 返回原 guard 的 0，between<2
为原定义的 NaN。`Q-S²/n` 风险触发 tiled two-pass；每 tile 最多
tile_size²，只枚举实际 W/A 合格 pair，不调用 full covariance。
two-pass 仍依赖 BLAS 分块轨迹，不是 NumPy 原运算逐位认证。

对极小 pair 方差和非有限统计明确抛出 FloatingPointError，并记录
unresolved 诊断；不 clamp、不加 epsilon、不伪造零 TD、不静默启动
官方二次内存路径。近退化拒绝阈值是保守工程限制，不是前向误差定理。
输入行相对偏置病态时，该低内存路线也拒绝整条 dense-reference 回退。
默认 TD 仍为 Prefix2D；feature-moments 明确标记 experimental。

## 增量 TD：操作级实验

单切 L/R：L×R 从 W 转 A，V×R 和 L×W 从 A 转 ignored，直接得到
用户给定的 ΔW/ΔA。双切 L/M/R：三个内部交叉块全部离开 W，仅 L×M
和 M×R 加入 A；V×M、V×R、L×W、M×W 离开 A。端点没有外邻时相应项
不存在。此类别迁移给出了单切和双切公式的独立推导。

`IncrementalTDistance` 用结果划分的 within 区间和 adjacent 区间对作为
贡献键，只增删变化键；每个贡献独立扣除 dmin 短带。这同时支持 split、
double split、delete、insert、distant move 和回滚，不依赖单调性假设。
active moments 与 region cache 在每次更新后清除失效条目。

已验证操作级六矩与独立 Prefix2D oracle；未接入生产 controller，未声称
完整 fit 等价或加速。当前 list bookkeeping 可为 O(K²)，矩累计加减存在
漂移，活动矩缓存为 O(KP²)。因此不能借用受限单调单切的 N log N 界。

## GPU 组件与生产路线

原 Torch/Triton 预计算后端复用新的 CPU 搜索和微调，实际 CUDA 完整
拟合与四条 EEG 回归保留。默认没有增加每 K 的设备同步。

另有独立实验 `resident_candidates.candidate_summary()`：Q 常驻 CUDA，
device 按 flat 生成候选，融合 gate、lookup、score、NaN/tie/sentinel
归约；最多 128 个 program summaries，无二次候选数组。grid-stride
显式携带最早 flat，跨 program 完成顺序不影响 winner；只有最终六个
诊断回到 host。outside 标量及分区元数据当前仍由 CPU 准备。

此组件只支持 FP64，实际 RTX 4050/Triton 运行了 gate、sentinel、NaN、
Inf、tie、跨 grid-stride 测试。初次编译错误和第二次 NaN runner 诊断
差异的原始失败日志均保留，第三次 11/11 通过。实际组件测量发现单切
launch/sync 成本较高，而大规模无 gate 双切有收益，因此增加显式实验
选项 `candidate_backend='triton-resident'`：仅无 stateopt gate 且至少
20,000 个双候选的调用启用；其余搜索和微调沿用 CPU streaming。
该阈值是可审查的实验调度策略，不是普遍最优界；计数只依赖计算拓扑。
合成与四条 EEG 的完整 trace 和端到端测量单独列入最终验收记录。
没有实现无 Q 的 GPU prefix kernel、GPU TD 或 GPU finetuning，也没有
将此混合选项描述成全流程 GPU-resident GSBS。

真实分块集成还发现 NumPy int64 标量无法由 Triton 参数 binder specialize；
入口统一转换 Python int，并用 NumPy 标量与四条完整 EEG 回归覆盖。
该失败与修复的输出分开保存，没有删除失败记录。

## 策略与兼容性

推荐普通 EEG：FP64、Full-Q、streaming candidates、Prefix2D TD。
可用 `candidate_backend='materialized'` 调回保留的物化搜索，且禁用局部
cache。低内存需显式同时选择 `q_strategy='streaming'` 和
`td_backend='feature-moments'`；默认科学参数不会自动改变。

`configs/long_memory_explicit.json` 是本次长序列测量的**受限科学配置**，
不是将任意已有 statewise/finetune 任务迁移到它的自动策略。
小 N 低内存完整配置已测单双切和微调；大 N 首次双切仍有平方候选成本。
未加入未经校准的 auto，也未改变 CPU-only 安装或旧 CLI 参数。
原 GSBSResult 文件可继续加载；增量组件须显式导入，GPU resident 须显式
选择配置，均不进入默认热路径。分支未发布新 tag/release，是否合并由
用户审查决定。
