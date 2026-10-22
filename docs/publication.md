# Research paper, software and reproducibility materials

The article **HP-GSBS: Accelerating greedy state boundary search with reusable interval statistics**, by Yanzhe Zhao and Yongzhang Ji, describes the interval-statistics formulation and its numerical, EEG, performance and fMRI evaluations. The [current author manuscript and Online Resource 1](https://github.com/JustinZHAO-05/HP-GSBS-SoftwareX-Reproducibility/tree/main/manuscript) are prepared for *Neuroinformatics*. The [arXiv preprint](https://arxiv.org/abs/2610.25898) remains the preferred paper citation until a journal DOI is available.

| Resource | Identity | Contents |
|---|---|---|
| Algorithm software | HP-GSBS 0.2.0; commit `b86b409169aa173d4a361d8cbcc633add1bfacf6`; [software DOI](https://doi.org/10.5281/zenodo.23422148) | Python API/CLI, CPU and CUDA backends, synthetic examples, tests and software reports |
| Current research materials | [Independent reproducibility repository](https://github.com/JustinZHAO-05/HP-GSBS-SoftwareX-Reproducibility) and its release/commit identifiers | Six figure sources, controlled observations and tables, four EEG inputs and saved outputs, Sherlock application and reproduction scripts |
| Earlier research archive | Reproducibility v1.0.1; [version DOI](https://doi.org/10.5281/zenodo.23457342) | The fixed five-figure archive published under its original title |
| Reproducibility archive series | [All-versions DOI](https://doi.org/10.5281/zenodo.23457341) | Version history for the independent research archive |

The software release, its version DOI and its scientific defaults remain fixed. Documentation updates provide navigation to the research materials. `CITATION.cff` retains the preprint as the preferred article citation and the software's own version DOI.

## Datasets and saved analyses

The [BBBD collection](https://github.com/JustinZHAO-05/HP-GSBS-SoftwareX-Reproducibility/tree/main/data/eeg) supplies all four frozen analysis matrices, 5.12-Hz time maps, recording-specific configurations and saved FP64 outputs. Recording 1 includes a complete original GSBS output. Recordings 2–4 retain the historically official-validated CPU reference chain. Their recorded preprocessing and input/file hash definitions accompany the data.

The [Sherlock application](https://github.com/JustinZHAO-05/HP-GSBS-SoftwareX-Reproducibility/tree/main/data/fmri) supplies group and 16 participant segmentations, configuration, fit times, boundary-alignment statistics and figure arrays. The versioned Figshare input is acquired with the public download-and-check script; new analyses use the public HP-GSBS 0.2.0 release.

Original code is MIT licensed. BBBD/Sherlock inputs and applicable derived materials retain CC BY 4.0 attribution and source credit in the research repository's `DATA_SOURCES.md` and `THIRD_PARTY_NOTICES.md`.

## Historical software evidence

Software CI exercises synthetic CPU behavior; CUDA/Triton receipts come from the documented RTX 4050 machine. Original test counts and performance reports retain their recorded environments and measurement scopes. Research-fixture scripts such as `freeze_reference.py`, `validate_saved.py`, `precision_bench.py` and full-record portions of `delivery_bench.py` document the original local validation workflow. The independently packaged research inputs and portable rerun scripts are linked above.

See the [documentation index](index.md), [performance scopes](performance.md) and [v0.2.0 release verification](release_verification_v0.2.0.md).
