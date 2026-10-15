# 架构

`fit` 统一输入验证、BLAS 线程限制与后端选择。GSBSConfig 分离算法参数和计算策略。GSBSResult 保留官方 GSBS 输出对象和原始 X，因此所有 get_* 方法、未选 K 的 state patterns 与 strengths 仍可直接调用。保存为无 pickle NPZ，附带计算配置、源代码版本、路径轨迹及实际 backend。

reference 直接调用 statesegmentation 0.0.6；可选 sys.settrace 仅观察 fit 局部变量，不修改官方类或模块。CPU 使用前缀 Q、Numba 评分、区域 t-distance 与顺序微调。CUDA 批量预计算 Q、R、二维前缀，并一次传回 CPU。每条记录有独立 controller，不把不同记录强制成相同 K 或插入路径。Triton 替换 Q 构造，并提供独立 fused 候选 kernel 作性能对照。

当前 GPU 后端是混合实现：没有将每轮控制器、finetune 或全部候选搜索搬到 GPU。其性能优势必须端到端测量，不以 kernel 时间代替。CPU-only 安装不加载 torch/triton。

批处理采用精确 T/P/dmin/dtype 分桶，避免 padding；不同 Kmax 可共享同一预计算批次。自动批次按空闲显存比例、16 个二次矩阵及 workspace 估计，不只计算 Q。CUDA OOM 缩半重试。manifest 驱动每任务单独保存，失败独立记录，完成结果原子提交且不覆盖；恢复匹配输入路径/stat 和配置，不依赖日志尾行。输入内容在保留相同大小和 mtime 的情况下被改写无法被 stat 签名识别，科研冻结输入不得修改。

多进程以记录为粒度。数组 API 会一次序列化每条记录；文件 manifest worker 自己读文件，避免父进程传大数组。每 worker 限制 BLAS/Numba 线程，n_jobs 由用户指定。fit_many 返回原输入顺序；manifest 日志按完成顺序，结果可立即用于下游。
