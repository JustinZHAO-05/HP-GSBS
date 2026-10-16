# Example configurations

- `development.json`: synthetic demonstration, K20/dmin1. It is not a BBBD research prescription.
- `bbbd_record0.json`: illustrative retained record-0 K131/dmin5 scientific settings. Use each study record's actual frozen parameters; do not apply this file to every recording.
- `long_memory_explicit.json`: experimental CPU raw-prefix/Feature-Moments, K16, single-cut and no finetuning. This deliberately restricted benchmark is not full statewise EEG validation.

The default compute policy is FP64 / full Q / streaming candidates / prefix2d TD. `kmax` and `dmin` must come from research design. Low-memory/resident/FP32 are explicit experiments; never select them from neural outcomes. See [API scope](../docs/v02_api_scope.md) and [migration](../docs/v02_migration.md).
