# 0.2.0.dev0 compute policy

默认科学路线为 FP64 / full Q / streaming candidates / prefix2d TD，支持
cpu-fast、torch-cuda、triton-cuda。CUDA 为预计算加 CPU 顺序控制；并非全流程 resident。

|模式|成熟度|用途与晋级条件|
|---|---|---|
|上述默认、window、active-block cache|stable|冻结/官方及实际 EEG、CUDA 回归域内使用；不是任意浮点输入误差定理|
|raw-prefix streaming / lazy Q|experimental，显式 opt-in|节约 Q 内存；仍可能平方候选数，非通用加速|
|feature-moments TD|experimental，显式 opt-in|节约相关矩阵/前缀内存；近退化 fail-closed；普通 EEG 不保证更快|
|selective Triton resident|experimental，显式 opt-in|仅 full Q、prefix2d、FP64；参数 resident_min_candidates 是设备相关计算阈值|
|FP32|experimental|GPU TD 可偏离 FP64；边界相同不等于严格数值通过|
|incremental TD|research-only，未开放 GSBSConfig|完整 controller 原型有 proposal/rollback/commit/rebase；需高 K 稳定收益及更广数值域才能升级|
|no-Q GPU / GPU moments|research-only 组件|仅 benchmark 原型；完整稳定归约、gate、fallback、TD/finetune 集成仍待完成|
|GPU incremental / GPU finetune 全流程|deferred|launch、状态依赖、兼容性与维护成本需进一步验证|

`result.metadata.maturity` 和 `compute_config` 记录实际设置。实验模式不能自动用于
正式 BBBD 结果。没有根据神经结果选择策略，也没有未经测量的 auto backend。

CPU fit 对 X/Y 作只读快照；Q/Z/scorer/cache 只在一个同步 fit 内有效。不同 fit、
精度或输入不会共享 cache。直接使用内部 CandidateSearch 时输入/scorer 必须保持不变。

计算预算默认：`max_candidates=50_000_000`、`max_fallback_elements=50_000_000`、
`max_workspace_bytes=2 GiB`、`max_pair_evaluations=50_000_000`。
工作区估算保守且含输出矩阵，不能代替实测 RSS/VRAM；GPU batch 另有 memory_fraction
准入和 OOM halving。需要更大预算时由调用者显式调整，不改变 scientific config。

超限抛出 `ResourceBudgetExceeded`，带 diagnostics；异常 fit 不生成 success result。
低内存 TD 对非有限/近退化统计拒绝认证，无 dense 回退或 variance clamp。
现有近退化检测保守到任一 W/A 组方差极小时即拒绝，包含某些另一组方差正常、
数学上 Welch 可定义的输入。该 false refusal 明确保留为 experimental 限制；
最终证据 numeric_false_refusal.json 给出固定反例，不能称全部重复模式均已支持。
近并列原式回退记录 reason、gap、first_problematic_candidate、实际重评数和元素数。
候选预算只拒绝执行，绝不剪枝、换 K 或跳过双切。官方 reference 后端保持原版行为，
不受优化后端预算控制。

CPU raw-prefix 可用 `fit(x,c,progress=callback,cancel=predicate)`：每 chunk 回调，
取消抛出 `SearchCancelled`。full-Q 每搜索域前后检查，非每候选检查；CUDA/reference
暂不支持回调。没有单 fit checkpoint：保存未完成 controller/trace/数值状态的维护成本
尚不合理；manifest 已完成记录支持恢复，未完成 fit 明确重新运行。

Manifest 恢复使用执行签名 v1：输入及可选 Y/time-map 的路径、大小、mtime，
HP-GSBS 软件版本、backend，以及经过 `GSBSConfig` 补全默认值的全部配置。
省略默认值与显式填写默认值具有相同身份；软件版本或计算策略变化必须重跑。
结果 metadata 保存 `execution_identity` 供审计；这不是仅凭科学参数复用结果的授权。
只有签名一致且 `status="success"` 的结果可跳过，CPU/GPU 共用这一判断。
失败、取消、缺失状态或无法验证的旧签名均返回 failed receipt，保留原文件。
v0.1.0 和旧版 v0.2 的无版本签名不会自动迁移：可用 `GSBSResult.load` 分析历史结果，
要确认执行了当前实现则使用新的输出目录重新运行 manifest。该签名仍依赖文件 stat，
不是输入内容 hash；它也不承诺跨相同开发版本的任意源码修改自动失效。

GPU 自动 batch 分桶同时隔离 `mode` 和 `memory_fraction`。
`mode="latency"` 的自动 batch 为 1；throughput 使用各自分桶的显存比例规划，
显式 batch_size 仍由调用者指定。显存比例是准入估算，OOM halving 仍保留。

`compare(...,level=...)` 支持 output-only、full-search-trace、finetune-move-trace、
numeric-stress；后三级缺有效 trace 则失败，move 级别缺任一逐移动字段也失败。
每次移动观察 after_delete 和实际 after_insert。numeric-stress 只规定严格 trace
和数值比较门槛，压力数据集由 tests 提供。默认 auto 兼容旧调用，报告实际 compared
和 skipped reason；旧 fixture 不能凭空提供逐移动证据。

软件开发版本为 0.2.0.dev0；正式 v0.2.0 tag、release、PyPI 需要另获批准。
v0.1.0 历史引用可从原 tag 读取。最终验收见 v02_merge_readiness.md。
