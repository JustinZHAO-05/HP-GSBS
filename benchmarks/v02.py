"""Versioned, reproducible A/B measurements; never overwrites an existing run.

Private fixtures are read in place, never copied into this repository.
Run with PYTHONPATH=src and the same interpreter for both revisions.
"""
import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
from hpgsbs import fit, GSBSConfig, GSBSResult
from hpgsbs.benchmark import benchmark
from hpgsbs.validation import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--revision", help="Explicit Windows worktree HEAD when Linux git cannot read its gitdir")
    p.add_argument("--fixtures", required=True)
    p.add_argument("--backends", nargs="+", default=["cpu-fast"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--q-strategy", default="full")
    p.add_argument("--td-backend", default=None)
    p.add_argument("--candidate-backend", default=None)
    p.add_argument("--workloads", nargs="+", default=["Small", "Medium", "Large", "XL", "BBBD0"])
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    fixtures = Path(a.fixtures)
    rows = []
    metadata = dict(head=a.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    python=platform.python_version(), platform=platform.platform(), arguments=vars(a),
                    source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path("src").rglob("*.py")})
    (out / "environment.json").write_text(json.dumps(metadata, indent=2))
    for name, t, features, k in [("Small",450,16,20),("Medium",900,32,30),
                                ("Large",1500,64,40),("XL",2100,64,50),
                                ("BBBD0",1336,64,131)]:
        if name not in a.workloads:
            continue
        x = (np.load(fixtures / "0.npy").astype(float) if name == "BBBD0"
             else np.random.default_rng(20261016).normal(size=(t,features)))
        extra = {}
        if a.td_backend: extra["td_backend"] = a.td_backend
        if a.candidate_backend: extra["candidate_backend"] = a.candidate_backend
        c = GSBSConfig(k, dmin=5 if name == "BBBD0" else 1, q_strategy=a.q_strategy, **extra)
        cached = fixtures / "0_full_reference.npz" if name == "BBBD0" else fixtures.parent / "benchmarks" / f"completed_{name}_reference.npz"
        ref = GSBSResult.load(cached) if cached.exists() else None
        for backend in a.backends:
            print("START", name, backend, flush=True)
            measured = benchmark(x, c, backend, repeats=a.repeats)
            result = fit(x, c, backend)
            validation = compare(ref, result) if ref is not None else {"status":"no cached official reference"}
            row = dict(workload=name, input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                       config=asdict(c), validation=validation, **measured)
            rows.append(row)
            (out / "latency.json").write_text(json.dumps(rows, indent=2))
            print("DONE", name, backend, measured["steady_median_s"], validation, flush=True)
            if ref is not None and not validation["passed"]:
                raise AssertionError(validation)
            if name == "BBBD0" and backend == "cpu-fast":
                profiler = cProfile.Profile()
                profiler.runcall(fit, x, c, backend)
                profiler.dump_stats(str(out / "bbbd0.prof"))
                import pstats
                with (out / "profile.txt").open("w") as f:
                    pstats.Stats(profiler, stream=f).strip_dirs().sort_stats("cumtime").print_stats(45)


if __name__ == "__main__":
    main()
