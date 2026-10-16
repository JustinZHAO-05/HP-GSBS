# v0.2.0 release verification

Release preparation is authorized. The final PR HEAD, squash merge commit, separate stable-version commit, annotated tag, CI URLs, package hashes and post-release results are recorded as execution proceeds. No release success is inferred from a preparation entry.

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
