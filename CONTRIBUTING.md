# Contributing

Changes should improve useful scientific computation while preserving official search semantics. Install `python -m pip install -e ".[test]"` and run `python -m pytest tests -q`. GPU tests require CUDA PyTorch/Triton; hosted CPU CI does not certify GPU parity.

For algorithm changes, include a small independent reference test and compare the full search trace. Keep statewise double insertion, official block strategy, t-distance selection, sequential finetuning, and boundary numbering intact. Separate approximation experiments from the default FP64 route.

Performance claims need synchronized end-to-end wall times, identical inputs/parameters/precision, a cold/warm distinction, and memory measurements. Report regressions as well as improvements. Kernel timings alone do not establish fit speedups.

Issue reports should include `python -m hpgsbs doctor`, backend, precision, input shape, algorithm configuration, and the first divergent K/candidate where available. Prefer a synthetic reproducer. Private EEG, subject manifests and fitted research outputs are not part of this repository.

The maintainer reports an implementation against a pinned official reference, not a proof of universal numerical equivalence. Source-space validation and larger production stress tests remain opportunities for evidence-backed contributions.

For v0.2, use strict `finetune-move-trace` comparisons when collecting new reference evidence; missing required trace fields must fail. Keep Feature-Moments/raw-prefix/resident/FP32 experiments labeled separately from default FP64. Test mixed resource policies and non-success/legacy cache artifacts when changing scheduling. Version changes intentionally invalidate execution caches.

Historical source/evidence manifests are immutable. Use `benchmarks/v02_verify_delivery.py --source-revision COMMIT` with the corresponding manifest, as shown in docs/v02_migration.md. New timing must use new output directories and explicit source identities. Reproduce README values with `python scripts/plot_v02_performance.py --check`; no long official benchmark rerun is required for documentation-only changes.
