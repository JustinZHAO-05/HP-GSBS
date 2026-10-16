# Scheduler follow-up: C22–C24

Library/test source: `7da26110e040820150412d6f47e7d1930926e26c`.
This supplements the unchanged `merge_9baff92` acceptance; it does not relabel its measurements.

|Platform|Passed|Skipped|Failures/errors|
|---|---:|---:|---:|
|Windows / Python 3.11|315|3|0/0|
|WSL / Python 3.10 / actual CUDA and Triton|380|1|0/0|

The WSL suite includes the four private full EEG fixtures. Windows skips optional CUDA
modules and the external audit checkout; WSL skips the external audit checkout.
The 23 warnings on each platform come from deliberate numerical stress cases
(precision loss, constant inputs and non-finite arithmetic); they are retained in the logs.

New coverage: 24 four-backend cache cases (exact legacy-signature construction,
old software version, failed/cancelled/missing status, success); canonical default
configuration and version/policy changes; two CPU-only mixed-policy planner orderings;
four actual Torch/Triton mixed-policy orderings with reference comparisons.
Legacy test files are constructed from valid saved results with the original hash
formula; these tests do not claim to rerun v0.1.0 performance measurements.

Reproduction from the repository root, in environments with the documented dependencies:

```bash
PYTHONPATH=src HPGSBS_FIXTURES=/private/frozen python -m pytest tests -q --junitxml=tests.xml
python benchmarks/v02_verify_delivery.py --manifest results/v02/scheduler_7da2611/manifest.json
```

Use the same environment variables in PowerShell with its native syntax. Tests create
synthetic manifests in temporary directories; the private frozen EEG fixtures are read
in place. No input arrays, result NPZs or participant manifests are published here.
Public JUnit/log copies redact machine paths and hostname, preserving case names and
all numerical results. Original hashes are recorded in the manifest; originals remain
in the private audit archive. `environment.json` records the tested package versions.

The execution key includes signature schema, software version, canonical GSBSConfig,
backend and input stat identities. It is not a content hash or an arbitrary source-tree
hash. Existing unversioned caches are preserved and rejected for automatic execution
resume; load them for analysis or select a new output directory to run the new code.
Both CPU and GPU resume require success. GPU automatic buckets isolate mode and
memory_fraction; explicit batch_size retains its caller-specified behavior.

Only batch/scheduler.py and batch/bucketing.py changed in the library. The verifier
checks this against the baseline manifest and confirms all baseline evidence files
are unchanged. The added verification scope and reports follow the library/test
snapshot; their hashes are explicit in this supplement. Historical performance
matrices were not rerun, and no mixed-policy speedup is claimed.
