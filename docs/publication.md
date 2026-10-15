# Public release and reproducibility

The public repository is a clean source snapshot. The local development history includes research manifests, so it is not uploaded wholesale. Core source files are copied without algorithm changes.

Included: source, synthetic examples, tests, documentation, aggregate performance/validation receipts, environment package versions, and the generated promotional image.

Excluded: EEG arrays, per-record fitted NPZ files, subject-to-file manifests, research cache paths, virtual environments, third-party checkouts, raw profiler traces, and local execution history. Environment paths in public receipts are replaced with placeholders; numeric measurements and comparison outcomes are retained. The full-reference receipt omits the original research source-spec manifest.

The 103/134 test counts in the scientific validation report describe the original local runs with private fixtures. A clean public checkout has additional skips. GitHub-hosted CI tests synthetic CPU behavior only; CUDA/Triton validation receipts come from the documented RTX 4050 machine.

The synthetic latency workload in `benchmarks/single_fit.py` uses seed 20261016; batch workloads use seed 200+i. Reproduce those on your hardware before choosing a backend. Exact timings depend on drivers, threads, power state, compilation caches and competing work.

Scripts `freeze_reference.py`, `validate_saved.py`, `precision_bench.py` and full-record portions of `delivery_bench.py` require local research fixtures and are not used by the dataset-free quick start. They document the validation procedure; the input matrices are not redistributed.
