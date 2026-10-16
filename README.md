<div align="center">

![HP-GSBS: neural signals passing through three computational routes into aligned temporal states](docs/assets/hpgsbs-hero.png)

# HP-GSBS

**High-Performance Greedy State Boundary Search · v0.2.0**

Find temporal states in neural recordings with faster computation and auditable decisions.

[![CPU tests](https://github.com/JustinZHAO-05/HP-GSBS/actions/workflows/tests.yml/badge.svg)](https://github.com/JustinZHAO-05/HP-GSBS/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-22b8a8.svg)](LICENSE)
[![Precision](https://img.shields.io/badge/Scientific_default-FP64-153c60)](#scientific-consistency)

[Quick start](#quick-start) · [Performance](#measured-performance) · [Migration](docs/v02_migration.md) · [Downloads](https://github.com/JustinZHAO-05/HP-GSBS/releases) · [简体中文](README.zh-CN.md)

</div>

> Release preparation: the PR package is still `0.2.0.dev0`; the separate main release commit will set `0.2.0`. Version-specific download links below become available with the formal release.

## What is it for?

GSBS segments a **time × features** matrix into successive states with relatively stable spatial activity patterns. HP-GSBS implements the original GSBS scientific definition efficiently for EEG, other multivariate neural time series and collections of independent recordings. Supply already-preprocessed matrices and the algorithm parameters from your study design.

FP64 is the default. Single/double boundary insertion, blockwise search, t-distance decisions, sequential finetuning, boundary birth labels and special K indexing are preserved. Preprocessing, artifact policy, K selection and downstream statistics remain part of the study. The original GSBS method and upstream implementation belong to their original authors; see [attribution and citation](#attribution-and-citation).

| Route | Implementation | Use |
|---|---|---|
| **CPU FastGSBS** · `cpu-fast` | NumPy/BLAS interval statistics and Numba streaming search | CPU-only fitting and independent process workers |
| **TensorGSBS** · `torch-cuda` | Batched PyTorch CUDA precomputation and the shared CPU streaming/refinement controller | NVIDIA GPU fitting with PyTorch |
| **TurboGSBS** · `triton-cuda` | Triton interval-score kernel and the same CPU controller | An alternative GPU route; measure your workload |
| **Official reference** · `reference` | Unmodified `statesegmentation.GSBS` 0.0.6 | Independent scientific comparisons |

The GPU routes are **hybrid GPU precomputation + CPU sequential search**, not fully GPU-resident GSBS. All three optimized routes benefit from the new streaming/refinement controller. Triton is not universally fastest, and each recording keeps an independent search path.

## What's new in v0.2.0

- **Stable default:** streaming candidate reduction avoids materializing the complete double-candidate score matrix; window-local refinement and an active-block cache reduce repeated work. Birth labels, finetuning order, original sentinel/tie/NaN/stateopt behavior and t-distance decisions remain intact.
- **Bounded execution:** candidate, original-expression fallback, pair-evaluation and workspace budgets refuse excessive work explicitly. CPU cancellation raises `SearchCancelled`; no partial approximate fit is labeled successful.
- **Auditability:** strict output/search/finetune-move/numeric comparison levels, expanded numerical tests, and versioned manifest execution signatures. GPU batches isolate different mode/memory policies; CPU/GPU resume both require success.
- **Experimental opt-in:** raw-prefix bounded chunks, Feature-Moments t-distance, selective Triton resident candidate reduction and FP32. These have narrower validation domains and are not automatic replacements for the default.
- **Research-only:** incremental TD transaction/rollback/commit, GPU no-Q prefix and GPU fixed-partition moments. These prototypes are not stable complete-fit public API backends.

See [CHANGELOG](CHANGELOG.md), [release notes](docs/release_notes_v0.2.0.md) and [API scope](docs/v02_api_scope.md).

## Measured performance

![v0.1 to v0.2 median fit time, observed repeat ranges and workload-specific speedup for CPU, PyTorch and Triton](docs/assets/performance-v0.2.0.png)

[SVG](docs/assets/performance-v0.2.0.svg) · [Unrounded derived data](docs/assets/performance-v0.2.0.json) · [Plot script](scripts/plot_v02_performance.py) · [Original receipts](results/v02/merge_9baff92/revisions/trials.json)

Ryzen 7 7840H · RTX 4050 Laptop 6 GB · WSL2 Ubuntu 22.04 · FP64 · TF32 off. The actual v0.1.0, pre-review v0.2 (`9309874`) and measured v0.2 (`9baff92`) source snapshots ran interleaved: **two warmups and three timed fits per workload/backend/snapshot**, controlled single BLAS/Numba threads, synchronized CUDA, and identical input/scientific settings within each workload. Validation and serialization are outside fit wall time.

The table compares v0.1.0 with `9baff92`; speedup is the ratio of unrounded medians. Ranges are three observed fit times, not confidence intervals. BBBD0 is one real recording fitted repeatedly, not three participants. The later C22–C24 scheduler fixes and formal version change **did not rerun this performance matrix**.

<!-- V02_TABLE_START -->
| Workload | Backend | v0.1.0 (s) | v0.2 snapshot (s) | v0.2 range (s) | Speedup |
|---|---|---:|---:|---:|---:|
| Small | CPU | 0.1235 | 0.1060 | 0.0918–0.1110 | 1.16× |
| Small | PyTorch | 0.0905 | 0.0745 | 0.0713–0.0767 | 1.21× |
| Small | Triton | 0.0850 | 0.0577 | 0.0550–0.0594 | 1.47× |
| Medium | CPU | 0.4231 | 0.3594 | 0.3218–0.3648 | 1.18× |
| Medium | PyTorch | 0.2027 | 0.1294 | 0.1127–0.2284 | 1.57× |
| Medium | Triton | 0.2205 | 0.1417 | 0.1278–0.1572 | 1.56× |
| Large | CPU | 1.2682 | 0.9342 | 0.9200–0.9438 | 1.36× |
| Large | PyTorch | 0.4950 | 0.2209 | 0.2123–0.2647 | 2.24× |
| Large | Triton | 0.5057 | 0.2572 | 0.2038–0.3187 | 1.97× |
| XL | CPU | 3.9172 | 2.1516 | 2.1451–2.3305 | 1.82× |
| XL | PyTorch | 2.1961 | 0.3596 | 0.3377–0.3660 | 6.11× |
| XL | Triton | 2.1311 | 0.5592 | 0.4573–0.5790 | 3.81× |
| BBBD0 | CPU | 3.0056 | 1.5711 | 1.4586–1.5922 | 1.91× |
| BBBD0 | PyTorch | 2.3270 | 0.7896 | 0.7769–0.9666 | 2.95× |
| BBBD0 | Triton | 2.3370 | 0.8193 | 0.7551–0.8731 | 2.85× |

<!-- V02_TABLE_END -->

Workloads: Small 450×16/K20, Medium 900×32/K30, Large 1500×64/K40, XL 2100×64/K50, BBBD0 1336×64/K131. All use statewise detection, finetune=1, strength order and blocksize=50; synthetic dmin=1, BBBD0 dmin=5. The shared FP64 Full-Q/Prefix2D scientific configuration is unchanged. The figure shows log seconds and linear speedup with a 1× reference.

The complete three-snapshot cells remain in the derived JSON. A separate focused historical CPU repeat found 1.4245 s pre-review versus 1.4496 s corrected v0.2 median; that modest overhead is not hidden or mixed into this matrix. Cold start, batching and experimental low-memory fits have separate timing scopes. The restricted 50,000×64, K16, single-cut/no-finetuning experiment took 1.4439 s median and 469.5 MiB peak RSS; this is not full long-N official double-search certification. See [performance and limits](docs/performance.md).

### Historical official GSBS baseline

The [v0.1 historical performance report](docs/performance_v0.1.md), [original PNG](docs/assets/performance-comparison.png), [SVG](docs/assets/performance-comparison.svg) and [script](scripts/plot_performance.py) remain available. Their official-GSBS speedups use a different measurement protocol: synthetic medians/ranges and a single BBBD0 fit per backend. Do not combine those thousands-fold ratios with the v0.1→v0.2 comparison above.

## Scientific consistency

| Evidence | Scope |
|---|---|
| Scheduler snapshot `7da2611`, Windows | **315 passed / 3 skipped** |
| Same snapshot, WSL with actual CUDA/Triton | **380 passed / 1 skipped** |
| Four complete frozen EEG recordings | Full64/frontal32 inputs; default FP64 complete outputs agree with validated official paths |
| Fresh official dev96 comparisons | Eight full-search/finetuning-move comparisons in the original v0.2 acceptance |
| C22–C24 regressions | Version/default identity, legacy and non-success refusal, mixed GPU policies and input order |

These are test counts, not participant counts. [Readiness](docs/v02_merge_readiness.md) and [parity details](docs/reference_parity.md) distinguish historical scientific snapshots, the scheduler supplement and release checks. Full official per-move traces were **not newly generated for every full EEG recording**. Source-space seven-network inputs are unavailable. GPU FP32 TD differs at strict FP64 tolerances even when boundaries match; FP32 remains experimental. Strict trace levels fail if required evidence is missing.

Near ties use original-expression rescoring. Ill-conditioned inputs may fall back to the official route with the reason recorded; resource/numerical guards can refuse work. This is evidence within a tested domain, not universal floating-point equivalence. Software equivalence does not validate BBBD behavior or memory hypotheses. Private EEG and individual fitted outputs are not distributed.

## Quick start

### Install

Requires **Python 3.10+**; recorded CI covers 3.10/3.11. Install the official CPU-only wheel from [v0.2.0 Release](https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.2.0):

```bash
python -m pip install https://github.com/JustinZHAO-05/HP-GSBS/releases/download/v0.2.0/hpgsbs-0.2.0-py3-none-any.whl
python -m hpgsbs doctor
```

Or install the [Python sdist](https://github.com/JustinZHAO-05/HP-GSBS/releases/download/v0.2.0/hpgsbs-0.2.0.tar.gz) with `python -m pip install hpgsbs-0.2.0.tar.gz`. GitHub's automatic source ZIP/tar.gz is a repository snapshot, not the Python sdist. **No PyPI publication is claimed**; do not assume `pip install hpgsbs` retrieves this project.

For examples/tests, clone the repository, `git checkout v0.2.0`, create and activate a virtual environment, then:

```bash
python -m pip install -e ".[test]"
python examples/quickstart.py --output examples/generated
python -m pytest tests -q
```

CPU installation needs NumPy, SciPy, Numba, threadpoolctl, psutil, tqdm and pinned `statesegmentation==0.0.6`; it does not install Torch/Triton. Optional dependencies and constraints are in [pyproject.toml](pyproject.toml). For NVIDIA CUDA choose a compatible wheel using the [official PyTorch selector](https://pytorch.org/get-started/locally/). The measured WSL/Linux environment used:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install triton==3.6.0
python -m hpgsbs doctor
python examples/quickstart.py --backend torch-cuda --output examples/generated/torch
python examples/quickstart.py --backend triton-cuda --output examples/generated/triton
```

PyTorch CUDA does not require the Triton backend. Triton is supported here on Linux/WSL, not native Windows; see [Triton installation](https://triton-lang.org/main/getting-started/installation.html). AMD and Apple GPU backends are not implemented. Missing CUDA dependencies cause an explicit error; choose `cpu-fast` explicitly.

### Fit, inspect, validate and save

```python
import numpy as np
from hpgsbs import GSBSConfig, GSBSResult, fit, fit_many
from hpgsbs.validation import compare
from hpgsbs.core.budgets import ResourceBudgetExceeded, SearchCancelled

x = np.random.default_rng(42).normal(size=(40, 8))  # synthetic demo
config = GSBSConfig(
    kmax=8, dmin=1,  # demo only: use your study's Kmax/dmin for EEG
    statewise_detection=True, finetune=1, finetune_order=True,
    blocksize=50, dtype="float64", q_strategy="full",
    candidate_backend="streaming", td_backend="prefix2d", trace=True,
)
try:
    result = fit(x, config, backend="cpu-fast")
    reference = fit(x, config, backend="reference")
    report = compare(reference, result, level="finetune-move-trace")
    if not report["passed"]:
        raise RuntimeError(report)
    bounds_at_6 = result.get_bounds(6)
    full_tdistance_curve = result.tdists
    result.save("my_fit.npz")  # refuses overwrite
    restored = GSBSResult.load("my_fit.npz")
    batch = fit_many([x, x.copy()], config, backend="cpu-fast")
except (ResourceBudgetExceeded, SearchCancelled) as exc:
    print("No successful fit produced:", exc)
except FileExistsError:
    print("Choose a new output filename; the existing result is preserved.")
```

Use `backend="torch-cuda"` or `"triton-cuda"` with the same config when available. `fit(..., y=y)` searches X and evaluates TD on held-out Y. `save(..., times=times)` retains the original time map. Comparison levels include `output-only`, `full-search-trace`, `finetune-move-trace` and `numeric-stress`; tracing is required for strict trace levels. Save/load preserves the compatible GSBS methods and all-K outputs.

Low-memory and resident modes are explicit experiments:

```python
from dataclasses import replace
low_memory = replace(config, q_strategy="streaming", td_backend="feature-moments")
# CPU only; often slower for ordinary high-K EEG; near-degenerate stats may refuse.
resident = replace(config, candidate_backend="triton-resident")
# CUDA FP64 + full Q + prefix2d only; selective candidates, not full GPU residency.
```

Use research-defined Kmax/dmin and preprocessing. These examples do not change BBBD settings. [Migration](docs/v02_migration.md) describes output compatibility and experimental limits.

## Batch, resume and CLI

`fit_many` supports independent recordings and per-record configs. Exact shape/compute-policy buckets also isolate `mode` and `memory_fraction`. Auto latency uses B=1; throughput plans from that bucket's fraction; explicit batch_size overrides auto planning. OOM halves the attempted batch. CPU `n_jobs` controls independent process workers; Windows process calls belong under `if __name__ == "__main__":`.

Manifest tasks follow [examples/tasks.json](examples/tasks.json); paths resolve from the running directory. Results commit atomically without overwrite. Resume requires **matching versioned execution identity and status=success**: signature schema, software version, backend, normalized full config, input path/size/mtime, and optional Y/time-map identities. Failed/cancelled/missing-status or legacy results are not silently skipped.

Old v0.1 results remain readable with `GSBSResult.load()`, but do not prove execution by v0.2. Use a new output directory when identity differs. **0.2.0.dev0→0.2.0 changes the execution signature**: development caches are not automatically reused as stable-version execution caches. This is provenance policy, not a numerical error. The signature is not a content/source-tree hash; stat-preserving changes and arbitrary edits within one development version are not detected.

```bash
python -m hpgsbs fit examples/generated/input.npy --kmax 12 --output another_fit.npz
python -m hpgsbs fit-batch examples/tasks.json --backend cpu-fast --workers 4 --output examples/generated/batch-v020
python -m hpgsbs validate examples/generated/input.npy --kmax 12 --backend cpu-fast
python -m hpgsbs compare examples/generated/fit.npz another_fit.npz
```

## How the acceleration works

Spatial sample z-scores (`ddof=1`) and time-prefix sums turn each interval's match score into reusable sufficient statistics. Full Q gives O(1) interval lookups; a streaming reducer traverses legal single/double candidates without keeping the full score matrix. Window refinement evaluates the actual local search domain; active-block cache entries remain valid only within the synchronous fit. Prefix2D TD preserves unbiased variance, `dmin` and Welch statistics. Default storage remains O(T²); streaming candidate reduction does not make Full Q linear-memory.

[Mathematics](docs/mathematics.md) · [Architecture](docs/architecture.md) · [Technical report](docs/technical_report.md) · [BBBD integration](docs/bbbd_integration.md) · [Limitations](docs/known_limitations.md) · [Contributing](CONTRIBUTING.md)

## Attribution and citation

**[Yanzhe Zhao](https://github.com/JustinZHAO-05) — Main contributor and maintainer** (Tianjin University).

> Zhao, Y. (2026). *HP-GSBS: High-Performance Greedy State Boundary Search* (Version 0.2.0) [Computer software]. GitHub. https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.2.0

Use [CITATION.cff](CITATION.cff) and cite the original GSBS papers and [upstream implementation](https://github.com/lgeerligs/statesegmentation):

- Geerligs et al. (2021), [Detecting neural state transitions underlying event segmentation](https://doi.org/10.1016/j.neuroimage.2021.118085).
- Geerligs et al. (2022), [A partially nested cortical hierarchy of neural states underlies event segmentation in the human brain](https://doi.org/10.7554/eLife.77430).

[v0.1.0](https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.1.0) and its fixed source `922a69c` remain intact. Code is [MIT licensed](LICENSE), with upstream attribution in [NOTICE](NOTICE). The [hero artwork](docs/assets/README.md) is AI-generated conceptual art, not a scientific measurement. Algorithm software validation and release do not establish downstream research conclusions.
