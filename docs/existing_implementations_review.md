# 现有实现审查

2026-10-16 已下载并完整检查相关评分与调度函数，审查的是固定提交，不等于将 fork 认定为可靠高速版本。

|实现|证据和风险|复用判断|
|---|---|---|
|[cwilson51](https://github.com/cwilson51/statesegmentation)，ecc3d05615259096cb9d3a3f44524cf12c86e4de|joblib 按双候选首坐标分发，并返回整行评分；支持 statewise。每行反复构造均值/标准化；传递 X,Z,xmeans,states。默认只读 memmap 时写 xmeans 会失败。小数组序列化缓存/worker 复用又可能让 xmeans 修改污染下一任务；使用线程后端也不安全。未见完整科学路径测试或端到端 benchmark。|返回行再收集的模式可借鉴；不复用修改共享 xmeans 的评分核心。|
|[官方 PR #2](https://github.com/lgeerligs/statesegmentation/pull/2)，2ae170fa29afa88f2331fc89d2c0f8713cfaa35a|ProcessPoolExecutor(32) 将普通数组传给 worker；worker 修改的是进程私有 wdists，返回 None，parent 没有合并评分，因而 stateopt=None 路径得到 None 并丢失双边界搜索。反复复制和建池代价也未解决。|不采用；若修复需返回评分并回填、避免固定 32 workers，再完整验证。|
|[Andesha 默认分支](https://github.com/Andesha/statesegmentation)，a75d2f144e740ca347b1a3c5035d1936ce192f0a|当前默认分支与官方相同。PR 分支必须单独审查，不能把默认分支描述成含 PR 改动。|作为相同 reference 来源，无额外加速。|
|独立 FastGSBS|搜索精确名称 FastGSBS 及 GitHub 组合未确认独立可核实项目地址。|未找到，不阻塞独立实现。|

HP-GSBS 多进程以独立记录为粒度，worker 返回完整结果或原子保存结果；不允许 worker 静默修改父进程数组。BLAS 和 Numba 线程数显式限制。后续风险复现见 tests/integration/test_fork_risks.py（若未执行，不能将静态发现写成实测）。
