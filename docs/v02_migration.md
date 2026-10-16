# Migrating to HP-GSBS v0.2.0

The public `fit`, `fit_many`, `GSBSConfig` and `GSBSResult` interfaces retain the GSBS scientific configuration and output methods. Default candidate search changes from materialized scores to streaming reduction, with local refinement and a per-fit active-block cache. Default FP64 / full Q / prefix2d TD remains the recommended starting point for ordinary EEG. Kmax/dmin, preprocessing and the study's K-selection function must come from the original protocol.

## Saved results and execution identity

v0.1 NPZ results remain loadable with `GSBSResult.load`. Loading an old scientific result does not establish that v0.2 generated it. Manifest resume now hashes signature schema, software version, backend, a complete GSBSConfig with defaults, and input/Y/time-map paths, sizes and mtimes. Only a matching key and `status="success"` permit a skip. Unversioned, failed, cancelled, missing-status or differently configured artifacts produce failed receipts; the file is preserved. Select a fresh output directory to execute the new version.

The transition **0.2.0.dev0 → 0.2.0** also changes the signature. Development results are historical scientific outputs, not same-version stable execution caches. This deliberate invalidation is provenance behavior, not a numerical regression. Do not hand-edit metadata to bypass it. Stat identities do not detect content changes that preserve path/size/mtime, and the key is not a hash of arbitrary edits within an unchanged development version.

GPU auto batches isolate `mode` and `memory_fraction` along with the existing shape and compute settings. Auto latency uses one record. Throughput uses its own bucket's fraction and may halve on OOM. Explicit batch size remains caller-controlled. This does not establish a globally optimal backend or batch size.

## Explicit experiments and failures

- CPU low memory: `q_strategy="streaming"`, `candidate_backend="streaming"`, `td_backend="feature-moments"`. Raw-prefix bounded chunks and moments avoid Q/TD quadratic storage, but candidates and output history can still be expensive. Ordinary high-K EEG is often slower; near-degenerate moments can conservatively refuse.
- Selective GPU candidate reduction: `candidate_backend="triton-resident"`, CUDA FP64, full Q, prefix2d. This is an opt-in experiment; most controller/finetuning work stays on CPU.
- FP32 is experimental. GPU TD does not meet strict FP64 tolerances across the validated EEG set. No TF32 fitting mode is exposed.
- Incremental TD transaction/rollback/commit and GPU no-Q/fixed-partition moments are research prototypes, not public stable complete-fit routes.

`ResourceBudgetExceeded` and `SearchCancelled` mean no successful complete result was produced. Inspect the error/diagnostics and select a justified budget or different explicit compute strategy; do not lower scientific Kmax or disable double search merely to make a run complete. CPU raw-prefix cancellation is chunk-based; CUDA/reference callbacks remain unsupported. There is no single-fit checkpoint.

## Evidence snapshots

| Identity | Meaning |
|---|---|
| `9baff923fea285d448fb3078c82943f365369b3b` | Original v0.2 scientific and performance measurement source |
| `7da26110e040820150412d6f47e7d1930926e26c` | C22–C24 scheduler implementation and regression source |
| `3c40c1c4860944160f0ffaefca56c1433c7d29f8` | Reviewed scheduler delivery including public evidence |
| PR final HEAD / squash merge / release commit / tag | Separate release identities, recorded in the final release verification receipt |

Historical manifests are immutable. Verify their recorded source against a fixed Git commit rather than changing hashes after documentation/version edits:

```bash
python benchmarks/v02_verify_delivery.py --manifest results/v02/merge_9baff92/manifest.json --source-revision 002bf2af70d8a4be2aa23bf7feb54d82500dd4ac
python benchmarks/v02_verify_delivery.py --manifest results/v02/scheduler_7da2611/manifest.json --source-revision 3c40c1c4860944160f0ffaefca56c1433c7d29f8
```

Run these from a repository clone containing the historical commits, not an extracted Python sdist. The verifier checks the requested commit's source hashes and the preserved evidence files; it does not claim to rerun those measurements. Publication of this software does not switch the backend of any existing BBBD analysis.
