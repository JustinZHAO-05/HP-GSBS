# v0.2 known limitations

- Default FP64 equivalence is supported within the documented synthetic and frozen EEG domain, not proven for every floating-point input. Old full EEG fixtures do not all contain fresh official per-move traces; dev96 fresh traces provide separate coverage.
- There are no supplied frozen seven-network source-space inputs. Sensor full64/frontal32 evidence does not certify source-space reconstruction or network definitions.
- Full Q and Prefix2D retain O(T²) memory. Streaming candidates remove the complete score matrix but do not remove Q/TD/output-history costs or quadratic candidate counts.
- Raw-prefix/lazy Q, Feature-Moments and selective resident are experimental. Moments conservatively refuses some near-degenerate yet mathematically defined Welch cases; no variance clamp is used to force a result. Ordinary high-K EEG often runs faster with Full Q.
- Incremental transaction/rollback/commit is a research-only controller prototype. GPU no-Q prefix and fixed-partition moments are components; no full no-Q GPU fit, GPU incremental TD, GPU sequential finetuning or full-resident controller is released.
- Selective resident requires CUDA FP64, Full Q and Prefix2D. Its component speedup did not establish reliable complete-fit gains; it remains off by default. Triton is not universally fastest.
- GPU FP32 TD fails strict FP64 tolerances on the validated EEG set even where boundaries agree. CPU FP32 Q with FP64 TD and GPU FP32 Q/R/prefix are different mixed-precision routes. TF32 is not a fitting mode.
- Restricted 50k single-cut/no-finetuning fits validate selected/final fixed-partition TD, not every official long-N double-search/finetuning decision.
- GPU buckets use exact shape and compatible compute/resource policy without padding. The memory planner is an estimate; OOM halving remains necessary. Thousands of recordings, multiple GPUs and very large source P are not production-certified; optimal backend/batch selection is not automatic.
- Resume checks version/schema/backend/config and file stat identity plus success. It is not a content hash or arbitrary code-tree hash. No distributed lock coordinates multiple schedulers sharing an output directory. Dev0 and stable caches deliberately differ; see [migration](v02_migration.md).
- There is no single-fit checkpoint. CPU cancellation/progress support is bounded by documented chunk/search boundaries, not every backend operation.
- Performance is measured on the documented laptop/WSL environment; ranges are observed repeated fits, not confidence intervals. Historical v0.1 baseline and v0.2 source comparisons have different contracts.
- Software validation/release does not validate BBBD measurement gates, behavior/memory hypotheses or a paper's conclusions, and does not automatically replace any research backend.
