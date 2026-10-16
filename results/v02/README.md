# v0.2 evidence

The pre-repair evidence is immutable at Git commit `9309874`. Its original files
were archived with CRC and individual SHA256 verification before removing duplicate
logs/profiler traces from the current tree. See archive_receipt.json and
historical_summary.json, which retain slow/failed/FP32 results. No numerical values
were improved or replaced. Historical documents link to the original commit.

New acceptance evidence is stored separately and selected by an explicit manifest.
Private input arrays and participant manifests are never published. Historical Git
contains local paths; this cleanup does not erase history and does not force push.
