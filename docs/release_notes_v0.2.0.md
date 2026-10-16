# HP-GSBS v0.2.0 — Faster Streaming Search and Low-Memory Extensions

HP-GSBS accelerates original Greedy State Boundary Search for multivariate neural recordings. FP64 is the default; the unmodified statesegmentation 0.0.6 implementation remains the scientific reference.

## Stable changes

Streaming candidates avoid the complete double-score matrix. Window-local refinement and active-block caching reuse work while retaining boundary birth labels, sequential finetuning, blockwise search, original sentinel/tie/NaN/stateopt semantics and single/double t-distance selection. Bounded resource/fallback work, cancellation and strict trace levels make refusal and verification explicit.

Manifest execution signatures include software version and normalized configuration. Legacy/unversioned results remain readable but cannot masquerade as new executions. CPU/GPU resume both require success, and GPU automatic buckets respect mode/memory_fraction independently. **0.2.0.dev0 and 0.2.0 have different execution-cache identities**; use a fresh output directory for stable-version fits.

## Measured performance

Ryzen 7 7840H / RTX 4050 Laptop 6 GB / WSL2 / FP64 / TF32 off. Actual v0.1.0, pre-review v0.2 and 9baff92 v0.2 snapshots ran interleaved, with two warmups and three timed fits per condition; validation excluded. The complete five-workload, three-backend [table](v02_performance_table.md), [figure](assets/performance-v0.2.0.png) and [receipts](../results/v02/merge_9baff92/revisions/trials.json) retain medians and repeat ranges. Ratios use unrounded medians.

BBBD0: CPU 3.0056→1.5711 s (1.91×), PyTorch 2.3270→0.7896 s (2.95×), Triton 2.3370→0.8193 s (2.85×). XL: 1.82× CPU, 6.11× PyTorch, 3.81× Triton. These are workload-specific historical measurements of 9baff92, not newly run timings of the release commit. The later scheduler/version changes did not rerun this matrix. The earlier official-GSBS thousand-fold baseline has a separate protocol and is not combined with these ratios.

## Scientific evidence and maturity

The scheduler snapshot passed 315 Windows tests (3 skips) and 380 WSL tests (1 skip), including actual CUDA/Triton and four complete frozen EEG inputs. Original v0.2 acceptance includes fresh official dev96 per-move comparisons. Full official per-move traces were not newly generated for every full EEG recording. See [parity](reference_parity.md) and [release verification](release_verification_v0.2.0.md) for snapshot-specific checks.

Raw-prefix low memory, Feature-Moments TD, selective Triton resident candidates and FP32 remain **experimental opt-in**. Moments can conservatively refuse near-degenerate inputs, including some mathematically defined Welch cases. GPU FP32 TD is not strictly FP64-equivalent. Incremental transaction/rollback/commit, GPU no-Q prefix and GPU fixed-partition moments are **research-only**. The GPU architecture still uses CPU sequential control. Full long-N double/finetuning trajectories, seven-network source-space validation and automatic backend selection remain unfinished. The restricted 50k single-cut/no-finetuning result does not certify full long-N GSBS.

## Installation and migration

Python 3.10+. CPU-only installation does not require Torch/Triton:

```bash
python -m pip install https://github.com/JustinZHAO-05/HP-GSBS/releases/download/v0.2.0/hpgsbs-0.2.0-py3-none-any.whl
python -m hpgsbs doctor
```

Alternatively download `hpgsbs-0.2.0.tar.gz` from this release and `python -m pip install hpgsbs-0.2.0.tar.gz`. This Python sdist differs from GitHub's automatic source archives. SHA256SUMS.txt accompanies the assets. PyPI publication is not part of this release.

For tests/examples, clone the [v0.2.0 source](https://github.com/JustinZHAO-05/HP-GSBS/tree/v0.2.0), install `.[test]`, and run `python examples/quickstart.py`. Install compatible NVIDIA CUDA PyTorch separately; Triton is supported here on Linux/WSL. The measured environment used PyTorch 2.11.0+cu128 and Triton 3.6.0. No AMD/Apple GPU backend is implemented. [English README](../README.md) and [中文 README](../README.zh-CN.md) give commands; [migration](v02_migration.md) explains defaults, output compatibility and cache invalidation.

## Citation and attribution

**Yanzhe Zhao — Main contributor and maintainer**, Tianjin University.

Zhao, Y. (2026). *HP-GSBS: High-Performance Greedy State Boundary Search* (Version 0.2.0) [Computer software]. GitHub. https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.2.0

Original GSBS method: Geerligs et al. (2021), [NeuroImage, 118085](https://doi.org/10.1016/j.neuroimage.2021.118085); Geerligs et al. (2022), [eLife, e77430](https://doi.org/10.7554/eLife.77430). Upstream [statesegmentation](https://github.com/lgeerligs/statesegmentation) attribution is retained in NOTICE; software metadata is in [CITATION.cff](../CITATION.cff). No unverified DOI is claimed.

This release updates algorithm software only. It does not change BBBD data, preprocessing, research parameters, behavior/quiz statistics, source-network definitions or scientific conclusions. No private EEG, individual result NPZ, participant manifest or environment is included.
