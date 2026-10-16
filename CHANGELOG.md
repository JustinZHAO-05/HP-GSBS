# Changelog

## 0.2.0 — 2026-10-16 (UTC)

### Added

- Streaming candidate reducer and bounded candidate/original-expression fallback execution.
- Window-local refinement, active-block cache and strict scientific trace-validation levels.
- Experimental Feature-Moments TD, raw-prefix bounded chunks and selective Triton resident reduction.
- ResourceBudgetExceeded/SearchCancelled with explicit failure rather than approximate success.
- Versioned manifest execution identity and expanded numerical/edge-case tests.
- Research-only incremental TD transaction/rollback/commit and GPU prefix/fixed-partition moments prototypes.
- Bilingual v0.2 homepage, machine-readable performance summary, reproducible PNG/SVG and migration guide.

### Changed

- Default candidates use streaming instead of materializing the complete double-score matrix.
- Finetuning evaluates its actual search window while preserving order and birth labels.
- GPU automatic batches safely separate different mode/memory_fraction policies.
- Manifest identity includes normalized complete config, software version and existing file identities; dev/stable versions intentionally do not share execution caches.

### Fixed

- Sentinel/runner, NaN/tie, empty-rescore and unscored candidate edge cases.
- Numba fresh-process cache and GPU metadata scalar handling.
- Accidental reuse of legacy execution results and inconsistent CPU/GPU success checks.
- Potentially unbounded long-N candidate/numerical fallback work, now budgeted.

### Known limitations

- FP32, low-memory TD and selective resident remain experimental; moments conservatively refuses some mathematically defined near-degenerate inputs.
- Long-N full statewise/double search with finetuning is not fully certified against official per-move traces.
- Incremental TD remains research-only. GPU is not a fully resident controller.
- Seven-network source-space inputs and automatic optimal CPU/GPU selection remain unvalidated/unimplemented.
- Performance remains tied to the documented hardware and 9baff92 measurement snapshot; later scheduler/version changes were not rebenchmarked.

## 0.1.0 — 2026-10-15 (UTC)

Initial CPU, PyTorch CUDA and Triton CUDA routes; unmodified statesegmentation 0.0.6 reference, independent batching, atomic results, CLI, synthetic examples and FP64 validation. The original [release](https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.1.0), tag and assets are preserved. Beijing publication date was 2026-10-16.
