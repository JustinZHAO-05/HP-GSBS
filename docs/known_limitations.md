# 已知限制与未完成项

> 本页保留 v0.1.0 的验收范围。v0.2 分支的新增能力、数值拒绝域与未完成项见
> [算法审查](v02_algorithm_review_CN.md)和[实测报告](v02_results_CN.md)。

- 新跑完整官方 reference 已保存，并完成逐轮候选/微调路径比较，详见 saved_validation.json。
- 2021 期刊最终全文下载端点受 403 限制；已核对官方代码、2020 原始预印本方法及 2022 方法/附录全文，未声称 2021 期刊全文阅读完成。
- 没有合法冻结的七网络源空间输入；只测试了 sensor full64 / frontal32 及合成低维输入，不做源重建或重新定义网络。
- GPU 是混合实现；没有 GPU 常驻逐轮控制器、persistent search、CUDA Graphs 或多 GPU。当前只实现 Full Q；CPU lazy 也仍有二次 t-distance 存储。
- Triton 实际运行并验证了 Q 融合 kernel、单/双候选 kernel；生产使用 Q kernel，候选 kernel 仅测试/微基准，未宣称已经实现全部 A–E 优化。Triton 仅在部分短记录负载优于 PyTorch。
- FP32 不通过严格 t-distance 容差；TF32 不开放 GSBS fit 模式。没有把 FP64 CPU 与 TF32 GPU 混为等价性能比较。
- 分桶按完全相同 T/P，不使用 padding；尚未做数千条、多个 GPU、源空间 P 很大的生产压力测试。manifest 在 GPU 路线用 mmap 和逐批 materialization；自动显存模型是保守估算，不保证永不 OOM。
- 恢复签名使用输入路径、大小和 mtime，加配置/backend；冻结输入不得原地改写并保留相同 stat。结果原子写、不覆盖，但未实现多调度器共享一个输出目录的分布式锁。
- 性能数据为本机实际桌面/WSL 环境，包含后台 reference 长任务的资源竞争；三次中位数不是跨机器结论。CPU RSS 为采样进程高水位，不是单次新增内存。首次调用和真正冷启动分别保存。
- 完成开发与算法一致性验证，不代表 BBBD 的测量有效性门、行为假设或正式论文研究完成；没有自动替换现有研究后端。
