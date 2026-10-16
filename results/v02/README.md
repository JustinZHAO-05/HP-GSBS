# v0.2 evidence index

Primary performance evidence is under `accepted/`: all jobs run sequentially
against the final source snapshot recorded by `accepted/provenance.json`.
Other run directories are retained development evidence, including failures;
do not mix their measurements into the final A/B ratios.

- `accepted/paired/`: interleaved same-input, same-scientific-config complete fits.
- `accepted/suite/`: CPU/GPU throughput, isolated-cache cold starts, actual CUDA
  profiler output, selective resident full fits, fresh-process memory A/B,
  FP32 diagnostics, bounded all-candidate long double audit and ablations.
- `accepted/long/`: restricted long-N full fits and selected/final partition oracles.
- `accepted_tests_windows.*`, `accepted_tests_wsl.*`: final full regression suites;
  private fixtures are read in place via `HPGSBS_FIXTURES` and are not distributed.
- `real_accepted/validation.json`: final-source four full EEG paths, both Full-Q and low-memory;
  fresh official dev96 runs additionally compare every refinement operation.
- `baseline_windows/`, `baseline_wsl_run1/`: before-edit latest-main measurements.
- `release_tests_windows.txt`: retained cached Numba generator lowering failure;
  fixed by fused enumeration and the fresh-process regression test.
- `resident_tests_wsl*.txt`: initial kernel compilation/diagnostic failures and
  later passing component tests; `suite_final/resident-fit.log` retains the
  NumPy scalar specialization failure fixed before final accepted measurements.

No EEG matrix, NPZ fixture or private raw recording is included here. Reports
contain hashes, scientific parameters, derived comparison summaries and timings.
RSS is sampled process total; throughput RSS sums parent/child RSS and may count
shared pages more than once. CUDA precompute wall and transfer overlap in scope.
The standalone component tables are not full-fit speedups. See the Chinese
reports in `docs/` for the precise certification and completion boundaries.
