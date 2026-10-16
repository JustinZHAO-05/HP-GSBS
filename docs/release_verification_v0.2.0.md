# v0.2.0 release verification

**Published and verified.** The formal [GitHub Release](https://github.com/JustinZHAO-05/HP-GSBS/releases/tag/v0.2.0) was published on 2026-10-16 at 16:03:15 UTC (2026-10-17 Asia/Shanghai). It is neither a draft nor a prerelease. The annotated tag remains fixed at the release-preparation commit. Later main changes only add release evidence and DOI metadata.

## Premerge documentation checks

Library and tests are unchanged from reviewed delivery 3c40c1c. Windows full suite: 315 passed / 3 skipped. WSL actual CUDA/Triton, reference-parity and batch integration: 173 passed. Both README Python examples and the expanded quickstart (strict move trace, fit_many, save/load, get_bounds and tdists) passed. The READMEs have 74 valid internal links/anchors; existing external links returned HTTP 200, and future v0.2 assets are checked after publication. The new PNG was visually inspected; SVG uses the same plot and all 15 data rows/45 source cells are receipt-derived. Historical plots and both acceptance manifests are unchanged.

The [premerge receipt and JUnit files](../results/releases/v0.2.0/premerge.json) preserve source/evidence hashes and redacted logs. Historical manifests pass fixed-commit verification at 002bf2a and 3c40c1c. No long official benchmark was rerun.

## Immutable acceptance identities

- Performance/scientific measurement: `9baff923fea285d448fb3078c82943f365369b3b`.
- Scheduler implementation/tests: `7da26110e040820150412d6f47e7d1930926e26c`.
- Reviewed scheduler delivery: `3c40c1c4860944160f0ffaefca56c1433c7d29f8`.
- Original v0.1.0 tag target: `922a69c76ba9c39964efd3c3a7f973625ae5b1f1`.

All original manifests and measurements remain immutable. New plots/tables derive from unrounded receipts. Fixed-commit source verification is documented in [migration](v02_migration.md). Release checks exercise CPU-only wheel/sdist, CLI, save/load, default FP64/reference comparison, real CUDA/Triton, frozen EEG and versioned resume; they do not rerun the historical long official performance matrix.

## Squash merge and stable-version preparation

PR #1 final head: `25e0eeeb41aad4ae88f5c0705becb203901f42b8`. Squash merge commit: `2810e8deaecaacb894a36cc6248589782b6367f6`. The feature branch is retained. A separate main commit changes package/CFF/version assertions to 0.2.0 and adds four explicit dev0→stable cache-refusal cases; it does not alter the fitting algorithms. CFF release date uses UTC (2026-10-16).

The exact release commit/tag, package SHA256 values, installation/GPU checks and public-release status are supplied by the post-release verification asset. Release-preparation metadata does not assert completion before those checks execute.

## Completed release and post-release checks

- Release-preparation commit / annotated tag target: `b86b409169aa173d4a361d8cbcc633add1bfacf6`.
- [Main release CI](https://github.com/JustinZHAO-05/HP-GSBS/actions/runs/38928134338) and [tag CI](https://github.com/JustinZHAO-05/HP-GSBS/actions/runs/38928854651): all four Ubuntu/Windows × Python 3.10/3.11 jobs passed in each run.
- Stable full suite: Windows **319 passed / 3 skipped**; WSL actual CUDA/Triton **384 passed / 1 skipped**. Numerical stress cases intentionally produce 23 warnings per full suite; there are no failures/errors.
- Isolated build, wheel/sdist `twine check`, metadata/license checks, package privacy inventory and clean CPU-only installations passed. Neither CPU environment contains Torch/Triton. Both passed strict synthetic official move comparison, fit_many, save/load, doctor, expanded quickstart and CLI fit-batch/resume.
- Actual v0.1.0 wheel code and 0.2.0.dev0 source code created synthetic results before release installation. Both remain readable with equivalent output. Stable manifest resume refuses those caches and preserves their NPZ bytes. Failed/cancelled/status-less results are also refused.
- The installed release wheel passed CPU/PyTorch/Triton strict synthetic move checks and 12 full frozen EEG comparisons (4 recordings × 3 backends). These compare complete frozen outputs/paths; they are not fresh official per-move full-EEG fits.
- Public wheel/sdist were downloaded after release, matched the built SHA256 values below, and were reinstalled non-editably. The CPU checks and examples passed again; a separate installed public wheel passed actual GPU and all 12 frozen comparisons again. The original v0.1.0 tag and all three asset digests remain unchanged; feature branch `25e0eee` is retained.
- Both README examples, internal paths/anchors and external release/download links passed. All 15 performance rows still match the 9baff92 receipts; no historical performance matrix was rerun.

| Asset | SHA256 |
|---|---|
| hpgsbs-0.2.0-py3-none-any.whl | `43f1819b7d8c31adea736345210550b0a5aae4a247e77e363c6650b1b5db749b` |
| hpgsbs-0.2.0.tar.gz | `e877b7b9e99df2efa7bd7ebd75350e47b8e72733a2d3779b2f49894f5dfef278` |

The [machine-readable receipt](../results/releases/v0.2.0/post_release.json), redacted full-suite logs/JUnit and portable installed-package probe accompany this report. SHA256SUMS.txt in the Release covers the two distribution packages. Post-release evidence has its own checksum file so those original package assets are not replaced.

## Archival DOI and scope

[Zenodo v0.2.0](https://doi.org/10.5281/zenodo.23422148) was created automatically by the existing integration. Its GitHub related identifier points to v0.2.0. The downloaded archive matches the recorded MD5; pyproject, CFF, package version, scheduler and README bytes match the tag. The DOI is added only to subsequent main metadata and release notes; the original tag/package CFF remains valid without that later DOI.

No PyPI upload, private EEG publication or research-data/parameter change occurred. Raw-prefix/Feature-Moments TD/selective resident candidates/FP32 remain experimental; incremental TD and GPU no-Q/fixed-partition moments remain research-only. Full long-N double/finetuning trajectories, broader incremental validation, GPU-resident control and seven-network source-space validation remain research work. Hardware-specific performance ratios are historical measurements, not universal guarantees or new release-commit benchmarks.
