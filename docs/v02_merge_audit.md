# PR #1 v0.2 merge audit

审计起点：`9309874678ae4e861711fd3bc4b96e332b5bd8ae`，远端无新增提交、review 或 comment。
reference 为未修改 statesegmentation 0.0.6。所有处置以最终证据 manifest 和 readiness 报告为准。
“确认”表示源码可直接确认或已有最小复现；“风险”不表示已发生科学错误。

|ID|文件/函数；类别；严重性|证据与根因|处理、回归入口|benchmark / commit / disposition|
|---|---|---|---|---|
|C01|candidate_search.best；memory；blocker|确认：原表达式回退无工作上限|候选数、模板元素、工作区预算；异常携带 first problematic candidate、gap、actual count、reason/strategy；test_merge_guards|待最终验收|
|C02|streaming_candidates；correctness；blocker|确认：near-tie 重评分全为 -1 时 double 返回坐标；固定 repeated X/negative supplied Z 反例|所有回退统一 None；window NaN runner；独立 NumPy dense oracle、chunk/gate/sentinel 测试|待最终验收|
|C03|feature_moments；numerical；experimental blocker|确认：raw Q-S²/n 消去；近退化拒绝是兼容性保护，非误差定理|保留 two-pass fail-closed，增加两遍 pair 预算；高精度研究与负结果另报|待最终验收|
|C04|incremental_tdistance/run_search；integration；experimental|确认：旧模块不接完整 fit|纯候选 transaction/rollback、commit after finetune、32 次周期重建、set 失效查找；完整轨迹原型测试|性能决定是否开放；待测|
|C05|stream_prefix；performance/memory；experimental|确认：双切 Θ(T²)，没有线性时间保证|O(T+K+gate) 精确计数、候选预算、chunk 回调/取消；异常无 success result；不剪枝|待最终验收|
|C06|resident_candidates；correctness/performance；experimental|风险：跨程序/runner/元数据；旧阈值不通用|全 summary oracle、形状/边界/host-device 检查，内部匹配 Q 只读，显式阈值参数|待实际 CUDA 和交错完整 fit|
|C07|GPU low-memory；integration；scope|确认：无 Q/TD/finetune 完整 GPU 路线缺失|组件原型、transfer/launch 成本评估；无完整 fit 宣称|待测后明确延期|
|C08|reference observer；validation；high|确认：旧 full fixture 缺 per-move|观察实际 after insertion；新 official trace 严格验收；旧缓存只证明现存字段|待全 EEG/可行 official 补充|
|C09|tests；validation；high|风险：非高斯/罕见配置未充分覆盖|预算、病态、独立 y、small P、-1 fine、顺序及事务测试|待最终测试|
|C10|FP32；numerical；high|确认：GPU FP32 TD 旧结果有 4e-5–3.54e-4 差异|保持 experimental；FP64 原式 Z 和 exact 回退不降精度|保留负结果并复跑|
|C11|batch/benchmarks；performance；medium|确认：旧吞吐只含 Small/同形|加入高 K、变形、array/manifest、completion latency、resume、OOM tests|待最终测量|
|C12|public evidence；security；release blocker|确认：旧日志暴露用户/worktree/机器路径；尚未发现凭据|归档原件后删除重复大日志；最终副本脱敏并扫描；不改公开历史|待扫描 receipt|
|C13|version/citation；release；blocker|确认：分支仍称 0.1.0|统一 0.2.0.dev0，历史 tag 引用保留于 tag；无 release|待版本测试/commit|
|C14|results/v02；release；medium|确认：约 107k 新增行多为重复执行记录|CRC/SHA256 外部归档及 Git 9309874 持久检索；主树保留摘要/必要 receipt|待 archive receipt|
|C15|acceptance evidence；validation；blocker|确认：代码将变，旧 accepted 不可当新证据|新目录 + final source SHA/hash manifest；可选 manifest verifier|待最终验收|
|C16|validation.compare；validation；blocker|确认：缺 trace 仍能 passed|显式 output/full-search/move/numeric 级别；缺字段 fail；报告 compared/skipped|test_merge_guards 已通过；待最终验收|
|C17|config/api/batch；memory/integration；high|确认：无统一资源拒绝，组合验证分散|早期兼容性检查，保守 precompute 估算；GPU buckets 纳入 scorer/TD/thread/budget|待最终验收|
|C18|API/metadata/docs；release；blocker|确认：experimental FP64 仍有笼统 validated 标签|maturity 明确；incremental API 由性能门槛决定；没有通用 auto|待最终 readiness|
|C19|candidate_chunks 编译原型；correctness；blocker（已弃用）|确认：Windows fatal 0xc0000374；具体 Numba 内存所有权根因未证实|不发布 compiled generator；改 resumable fixed-buffer fill，17 个定向测试及 fresh-process 回归通过|research_failures.json；最终 source 重验|
|C20|CPU/prefix/window/resident runner；correctness；blocker|确认：n=2 single / all scored below -1 时，只有一个真实未评分 slot，虚拟 sentinel 错占 runner；winner 不变|按真实未评分 slot 数归约；GPU 记录 scored maximum，排除重复虚拟 sentinel；n=1 矩阵 runner=-Inf|新增 56 CPU + 8 CUDA 小域回归；定向 CPU 153 / CUDA 26 passed；最终源码重验|

计算预算均为执行策略；不足时明确失败，绝不换 K、删候选或跳过 double。
同步 fit 对输入作只读快照，cache 不再依赖调用者在拟合期间不修改原数组。
未合并、未发布、未改 main/tag/reference/私有 EEG。
