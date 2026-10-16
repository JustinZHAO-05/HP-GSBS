"""Paired FP64 wall-time measurements; save every trial before the next long fit.

One process, no concurrent fits, no tracing; warmup + three trials by default.
--single-pass selects one measured fit per backend for specified workloads.
BBBD0 requires the private frozen results/fixtures/0.npy (not distributed).
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import threading
import numpy as np
import psutil
from hpgsbs import GSBSConfig, fit
from hpgsbs.doctor import doctor
from hpgsbs.validation import compare


WORKLOADS = {"Medium": (900, 32, 30), "Large": (1500, 64, 40),
             "XL": (2100, 64, 50), "BBBD0": (1336, 64, 131)}
BACKENDS = ("reference", "cpu-fast", "torch-cuda", "triton-cuda")


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--single-pass", nargs="*", choices=WORKLOADS, default=[],
                        help="Run these workloads once per backend, without a warmup or repeats")
    parser.add_argument("--output", default="results/benchmarks/completed_reference_latency.json")
    args = parser.parse_args()
    if "BBBD0" in args.workloads and not Path("results/fixtures/0.npy").is_file():
        raise FileNotFoundError("BBBD0 needs the private frozen results/fixtures/0.npy; "
                                "for public inputs use --workloads Medium Large XL")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    ledger = output.with_name(output.stem + "_trials.json")
    if output.exists() or ledger.exists():
        raise FileExistsError("Choose a new output path; completed measurements are never overwritten")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This paired comparison requires CUDA and the tested Triton environment")
    environment = doctor()
    environment.pop("executable", None)
    run = dict(started_at=datetime.now(timezone.utc).isoformat(), environment=environment,
               protocol="Sequential same-process FP64; default warmup then 3 rounds; single-pass workloads once per backend; trace=False; 1 BLAS/Numba thread; no output I/O inside fit timing",
               single_pass_workloads=args.single_pass,
               trials=[])
    save(ledger, run)
    aggregates = []
    for name in args.workloads:
        T, P, kmax = WORKLOADS[name]
        x = (np.load("results/fixtures/0.npy", allow_pickle=False).astype(np.float64)
             if name == "BBBD0" else np.random.default_rng(20261016).normal(size=(T, P)))
        assert x.shape == (T, P)
        fingerprint = hashlib.sha256(x.tobytes()).hexdigest()
        config = GSBSConfig(kmax, dmin=5 if name == "BBBD0" else 1, dtype="float64", trace=False)
        baseline = fit(x, config, "cpu-fast")
        single = name in args.single_pass
        for iteration in range(1 if single else 4):
            for backend in BACKENDS:
                print("START", name, backend, "single" if single else ("warmup" if iteration == 0 else f"trial={iteration}"),
                      datetime.now(timezone.utc).isoformat(), flush=True)
                gpu = backend.endswith("cuda")
                if gpu:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                done = threading.Event()
                process = psutil.Process(); peak = [process.memory_info().rss]
                def monitor():
                    while not done.wait(.005):
                        peak[0] = max(peak[0], process.memory_info().rss)
                worker = threading.Thread(target=monitor, daemon=True); worker.start()
                start = perf_counter()
                try:
                    result = fit(x, config, backend)
                    if gpu: torch.cuda.synchronize()
                    wall = perf_counter() - start
                finally:
                    done.set(); worker.join()
                validation = compare(baseline, result)
                trial = dict(workload=name, iteration=iteration, wall_s=wall,
                             cpu_peak_rss_bytes=peak[0], **result.metadata,
                             input_sha256=fingerprint, validation_against_cpu=validation)
                if gpu:
                    trial.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                run["trials"].append(trial); save(ledger, run)
                print("DONE", name, backend, iteration, f"{wall:.6f}s", "parity", validation["passed"], flush=True)
                if not validation["passed"]:
                    raise RuntimeError(f"Parity failure: {name} {backend}: {validation}")
                if iteration == 0 and backend == "reference":
                    result.save(output.parent / f"completed_{name}_reference.npz")
        for backend in BACKENDS:
            trials = [r for r in run["trials"] if r["workload"] == name and r["backend"] == backend]
            measured = trials if single else trials[1:]
            median = float(np.median([r["wall_s"] for r in measured]))
            aggregates.append(dict(workload=name, backend=backend, precision="float64", shape=[T, P],
                                   algorithm=config.algorithm(), compute_config=asdict(config), input_sha256=fingerprint,
                                   first_call_s=trials[0]["wall_s"],
                                   **({"single_fit_s": median} if single else {"steady_median_s": median}),
                                   measurement_kind="single-fit" if single else "median-of-3",
                                   fits_per_minute=60/median, validation_against_cpu=trials[-1]["validation_against_cpu"],
                                   note=("Single fit; no separate warmup; no variability estimate. " if single else "Warmup excluded. ") + "Same process and paired rounds; no fit tracing or serialization inside wall time. RSS is sampled process total. Setup/import may precede fit; not true cold-start.", trials=trials))
        save(output, aggregates)
        print("WORKLOAD_COMPLETE", name, flush=True)
    run["completed_at"] = datetime.now(timezone.utc).isoformat(); save(ledger, run)


if __name__ == "__main__":
    main()
