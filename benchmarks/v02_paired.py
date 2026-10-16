"""Sequential interleaved A/B; materialized is the retained v0.1 search path.

Do not run other HP-GSBS benchmark processes concurrently with this script.
"""
import argparse,hashlib
from dataclasses import replace,asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from hpgsbs import fit,GSBSConfig,GSBSResult
from hpgsbs.benchmark import benchmark
from hpgsbs.validation import compare
from hpgsbs.doctor import doctor


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",required=True); p.add_argument("--output",required=True)
    p.add_argument("--backends",nargs="+",default=["cpu-fast","torch-cuda","triton-cuda"])
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); rows=[]
    environment=doctor(); environment.pop("executable",None)
    environment["source_hashes"]={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path("src").rglob("*.py")}
    (out/"environment.json").write_text(json.dumps(environment,indent=2))
    f=Path(a.fixtures)
    for name,t,p,k in [("Small",450,16,20),("Medium",900,32,30),("Large",1500,64,40),
                        ("XL",2100,64,50),("BBBD0",1336,64,131)]:
        x=np.load(f/"0.npy").astype(float) if name=="BBBD0" else np.random.default_rng(20261016).normal(size=(t,p))
        c=GSBSConfig(k,dmin=5 if name=="BBBD0" else 1)
        for backend in a.backends:
            expected=fit(x,replace(c,candidate_backend="materialized"),backend)
            for iteration in range(4):
                for mode in (["materialized","streaming"] if iteration%2==0 else ["streaming","materialized"]):
                    config=replace(c,candidate_backend=mode)
                    # benchmark(repeats=1) would add an unwanted warmup each pair.
                    import psutil,threading
                    done=threading.Event(); peak=[psutil.Process().memory_info().rss]
                    def monitor():
                        while not done.wait(.005): peak[0]=max(peak[0],psutil.Process().memory_info().rss)
                    thread=threading.Thread(target=monitor,daemon=True); thread.start()
                    gpu=backend.endswith("cuda")
                    if gpu:
                        import torch
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    start=perf_counter()
                    try:
                        r=fit(x,config,backend)
                        if gpu: torch.cuda.synchronize()
                        wall=perf_counter()-start
                    finally: done.set(); thread.join()
                    report=compare(expected,r)
                    row=dict(workload=name,backend=backend,mode=mode,iteration=iteration,
                        warmup=iteration==0,wall_s=wall,peak_rss_bytes=peak[0],report=report,
                        gpu_allocated=torch.cuda.max_memory_allocated() if gpu else None,
                        gpu_reserved=torch.cuda.max_memory_reserved() if gpu else None,metadata=r.metadata)
                    rows.append(row); (out/"trials.json").write_text(json.dumps(rows,indent=2))
                    print(name,backend,mode,iteration,wall,report["passed"],flush=True)
                    if not report["passed"]: raise AssertionError(report)


if __name__=="__main__": main()
