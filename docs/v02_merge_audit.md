# PR #1 v0.2 merge audit

审计起点：`9309874678ae4e861711fd3bc4b96e332b5bd8ae`，远端无新增提交、review 或 comment。
reference 为未修改 statesegmentation 0.0.6。所有处置以最终证据 manifest 和 readiness 报告为准。
“确认”表示源码可直接确认或已有最小复现；“风险”不表示已发生科学错误。

|ID|文件/函数；类别；严重性|证据与根因|处理、回归入口|benchmark / commit / disposition|
|---|---|---|---|---|
|C01|candidate_search.best；memory；blocker|确认：原表达式回退无工作上限|候选数、模板元素、工作区预算；异常携带 first problematic candidate、gap、actual count、reason/strategy；test_merge_guards|见最终处置与证据|
|C02|streaming_candidates；correctness；blocker|确认：near-tie 重评分全为 -1 时 double 返回坐标；固定 repeated X/negative supplied Z 反例|所有回退统一 None；window NaN runner；独立 NumPy dense oracle、chunk/gate/sentinel 测试|见最终处置与证据|
|C03|feature_moments；numerical；experimental blocker|确认：raw Q-S²/n 消去；近退化拒绝是兼容性保护，非误差定理|保留 two-pass fail-closed，增加两遍 pair 预算；高精度研究与负结果另报|见最终处置与证据|
|C04|incremental_tdistance/run_search；integration；experimental|确认：旧模块不接完整 fit|纯候选 transaction/rollback、commit after finetune、32 次周期重建、set 失效查找；完整轨迹原型测试|见最终处置与证据|
|C05|stream_prefix；performance/memory；experimental|确认：双切 Θ(T²)，没有线性时间保证|O(T+K+gate) 精确计数、候选预算、chunk 回调/取消；异常无 success result；不剪枝|见最终处置与证据|
|C06|resident_candidates；correctness/performance；experimental|风险：跨程序/runner/元数据；旧阈值不通用|全 summary oracle、形状/边界/host-device 检查，内部匹配 Q 只读，显式阈值参数|见最终处置与证据|
|C07|GPU low-memory；integration；scope|确认：无 Q/TD/finetune 完整 GPU 路线缺失|组件原型、transfer/launch 成本评估；无完整 fit 宣称|见最终处置与证据|
|C08|reference observer；validation；high|确认：旧 full fixture 缺 per-move|观察实际 after insertion；新 official trace 严格验收；旧缓存只证明现存字段|见最终处置与证据|
|C09|tests；validation；high|风险：非高斯/罕见配置未充分覆盖|预算、病态、独立 y、small P、-1 fine、顺序及事务测试|待最终测试|
|C10|FP32；numerical；high|确认：GPU FP32 TD 旧结果有 4e-5–3.54e-4 差异|保持 experimental；FP64 原式 Z 和 exact 回退不降精度|保留负结果并复跑|
|C11|batch/benchmarks；performance；medium|确认：旧吞吐只含 Small/同形|加入高 K、变形、array/manifest、completion latency、resume、OOM tests|见最终处置与证据|
|C12|public evidence；security；release blocker|确认：旧日志暴露用户/worktree/机器路径；尚未发现凭据|归档原件后删除重复大日志；最终副本脱敏并扫描；不改公开历史|见最终处置与证据|
|C13|version/citation；release；blocker|确认：分支仍称 0.1.0|统一 0.2.0.dev0，历史 tag 引用保留于 tag；无 release|见最终处置与证据|
|C14|results/v02；release；medium|确认：约 107k 新增行多为重复执行记录|CRC/SHA256 外部归档及 Git 9309874 持久检索；主树保留摘要/必要 receipt|见最终处置与证据|
|C15|acceptance evidence；validation；blocker|确认：代码将变，旧 accepted 不可当新证据|新目录 + final source SHA/hash manifest；可选 manifest verifier|见最终处置与证据|
|C16|validation.compare；validation；blocker|确认：缺 trace 仍能 passed|显式 output/full-search/move/numeric 级别；缺字段 fail；报告 compared/skipped|test_merge_guards 已通过；见最终处置与证据|
|C17|config/api/batch；memory/integration；high|确认：无统一资源拒绝，组合验证分散|早期兼容性检查，保守 precompute 估算；GPU buckets 纳入 scorer/TD/thread/budget|见最终处置与证据|
|C18|API/metadata/docs；release；blocker|确认：experimental FP64 仍有笼统 validated 标签|maturity 明确；incremental API 由性能门槛决定；没有通用 auto|见最终处置与证据|
|C19|candidate_chunks 编译原型；correctness；blocker（已弃用）|确认：Windows fatal 0xc0000374；具体 Numba 内存所有权根因未证实|不发布 compiled generator；改 resumable fixed-buffer fill，17 个定向测试及 fresh-process 回归通过|research_failures.json；最终 source 重验|
|C20|CPU/prefix/window/resident runner；correctness；blocker|确认：n=2 single / all scored below -1 时，只有一个真实未评分 slot，虚拟 sentinel 错占 runner；winner 不变|按真实未评分 slot 数归约；GPU 记录 scored maximum，排除重复虚拟 sentinel；n=1 矩阵 runner=-Inf|新增 56 CPU + 8 CUDA 小域回归；定向 CPU 153 / CUDA 26 passed；最终源码重验|

计算预算均为执行策略；不足时明确失败，绝不换 K、删候选或跳过 double。
同步 fit 对输入作只读快照，cache 不再依赖调用者在拟合期间不修改原数组。
未合并、未发布、未改 main/tag/reference/私有 EEG。

## 最终处置（source 9baff92；证据 merge_9baff92）

全部行均以新 manifest / readiness 的实测值为准。Windows 288 passed/3 skip，WSL 349 passed/1 skip，另外16严格移动对照。
初始登记中的“待”由下表取代；残余限制没有被改写成无限域正确性声明。

|ID|修复 commit|回归 / benchmark 入口|最终 disposition 与残余风险|
|---|---|---|---|
|C01|13cdac6/c50e035|test_merge_guards；long、components、预算异常|关闭：显式工作/内存拒绝、诊断/取消；预算不足不返回近似结果|
|C02|13cdac6/c50e035/9baff92|独立 NumPy oracle；56小域case；8实际CUDA小域case；streaming tests|关闭：None、空重评、真实 sentinel/runner、NaN/tie/C-order 均验证|
|C03|13cdac6/deea9f8|feature_moments/Fraction；research numeric targets；numeric_false_refusal|保护范围内关闭；任一组近零方差的保守 false refusal 明确保留；更细误差界延期，experimental|
|C04|13cdac6/deea9f8|事务/160次移动/周期重建；research 16次完整controller|完整集成原型已尝试且轨迹通过；收益仅约5–7%，多记录/Y域不足，research-only，不开放config|
|C05|13cdac6/deea9f8|bounded fill/fresh process/chunk1/7/64/256；12.49m candidates|关闭：编译buffer填充、精确计数、预算/进度/取消；平方复杂度保留，没有候选剪枝|
|C06|13cdac6/9baff92|全部 summary、真实EEG、80配对fit、resident profile|显式experimental；本机无稳定全局收益；阈值可配，不设通用auto；未验证metadata reuse不接入|
|C07|deea9f8 + research evidence|GPU prefix与fixed-partition moments原型|组件已测；GPU no-Q完整搜索/TD/incremental/finetuning延期，API明确拒绝不支持组合|
|C08|13cdac6|real24项；dev96新鲜official8项；supplement16|分层关闭；完整EEG旧fixture缺官方逐移动证据；source networks not available，不伪造|
|C09|13cdac6/9baff92|全suite+supplemental trace checks|关闭于声明域：重复/偏置/抵消/smallP/singleton/大dmin/Y/-1fine/预算压力；非任意浮点证明|
|C10|13cdac6|FP64 Z guard；precision12项|FP32保持experimental；GPU严格TD不通过属预期负结果，不计作FP64 gate通过|
|C11|deea9f8 + supplemental evidence|throughput_fit_only完整三workload；manifest/resume/OOM|关闭：纯fit计时/实际yield；旧含验收诊断中止并保留；大batch准入skip明确；不称通用最优|
|C12|deea9f8/9baff92 + delivery|archive/sanitize/compact scan receipt|当前公开副本脱敏、密钥/身份扫描；原历史路径仍存在，未force rewrite；原件私有归档|
|C13|13cdac6/deea9f8|version/maturity test；CITATION|关闭：0.2.0.dev0；v0.1.0 tag/release原封保留；新release未授权|
|C14|deea9f8 + delivery|archive_receipt、public_compaction receipt|关闭：原ZIP CRC/逐文件hash验证；Git9309874持久检索；保留慢/失败/FP32与本轮诊断|
|C15|9baff92 + delivery|source.json/manifest/verify_delivery|关闭：新证据目录、实际HEAD与source hashes；最终交付仅报告/证据，最终CI另绑PR HEAD|
|C16|13cdac6|missing/malformed trace tests；strict real/dev checks|关闭：full-search/move/numeric-stress缺证据失败；输出级别独立，skipped reason明确|
|C17|13cdac6|组合/预算/输入快照/batch tests|关闭：提前拒绝不支持组合；Q/TD/输出空间保守预算；GPU buckets隔离相关设置|
|C18|13cdac6/deea9f8 + docs|version/maturity test；v02_api_scope|关闭：默认stable；lowmemory/resident/FP32 opt-in；incremental/GPU原型research-only|
|C19|deea9f8|research_failures；fresh-process regression|关闭：未发布崩溃的compiled-generator原型，固定buffer实现经过Windows/Linux重验|
|C20|9baff92|153 CPU/26 CUDA定向；349 WSL全suite|关闭：真实未评分slot计数；n=1 runner=-Inf，单unscored域不复制sentinel|

完整科学/性能结论、CI与最终交付HEAD见 v02_merge_readiness.md 和 PR #1 描述。未执行 merge 或 release。

|C21|公开 scanner/sanitizer；release；high|确认：无边界的 drive 正则把 https 的 s:/ 误当盘符；非数值计算问题|加入词边界限制，URI/Windows 路径定向检查；从已归档原件重新脱敏；公开扫描零发现|本交付证据工具 commit；delivery_tool_changes.json；关闭|

C21 仅更改 benchmarks/v02_evidence.py 的公开复制处理和 scripts/compact_public_evidence.py；库、tests 和实际计时实现仍与9baff92完全一致。旧测量 hash 与最终工具 hash 分别保留，不冒充重跑。
