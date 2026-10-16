# BBBD / existing-study integration

Supply the already-preprocessed time × features matrix and inherit every original algorithm parameter. The measured BBBD fixtures used dmin=5 and their original per-video Kmax; do not replace them with the package defaults.

Create a task manifest following `examples/tasks.json`, using your own `.npy` paths. Add `times` for the original time map and `y` for held-out data if needed. Run `python -m hpgsbs fit-batch your_tasks.json --backend cpu-fast --workers 4 --output your_outputs` from the directory against which those paths resolve. The repository contains no EEG or subject manifests.

NPZ outputs include all_bounds, tdists, bounds, deltas, states, state_patterns, strengths and state durations in samples. When a time map is supplied, the original times and boundary timestamps are also saved. `GSBSResult.load(...).get_bounds(k)` preserves the original special K indexing. Feed the full tdists into your existing K-selection function rather than silently switching to default nstates.

Do not change filtering, artifact handling, feature definitions, quiz/behavior data, LOSO alignment, statistical hypotheses or source localization as part of a backend switch. Frozen matrices and time maps remain necessary; boundary caches cannot reconstruct signal values. Do not concatenate discontinuous segments or assume stimulus/video clock alignment.

Each source network is an independent task with P>=2 original parcel/source features. A single averaged source signal is not a valid GSBS spatial-pattern input. No seven-network source data were validated in this release.

The local validation ran four complete records and verified resume behavior. Public examples reproduce the software workflow on synthetic inputs. Adoption in a study requires validation on its own inputs and parameters.

Use FP64, full Q, streaming candidates and prefix2d as the ordinary EEG starting policy, while inheriting all original scientific parameters. The example config is illustrative, not a replacement research protocol. A backend upgrade requires study-specific validation and never automatically changes existing analyses.

Manifest execution keys include software version, complete normalized config, backend and X/Y/time-map stat identities. v0.1 and dev0 results remain loadable but are not stable 0.2.0 execution caches; choose new output directories. Only matching successful outputs may be skipped. See [migration](v02_migration.md). Experimental low-memory/resident/FP32 choices must remain explicit.
