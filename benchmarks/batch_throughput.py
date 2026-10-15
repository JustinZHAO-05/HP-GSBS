import argparse
from dataclasses import replace
from time import perf_counter
from pathlib import Path
import json
import numpy as np
from hpgsbs import fit,fit_many,GSBSConfig


def main():
    p=argparse.ArgumentParser(); p.add_argument("--backend",default="cpu-fast"); p.add_argument("--T",type=int,default=450)
    p.add_argument("--P",type=int,default=16); p.add_argument("--kmax",type=int,default=20)
    p.add_argument("--workers",type=int,default=1); p.add_argument("--batch-sizes",default="1,2,4,8,16,32")
    p.add_argument("--output",required=True); a=p.parse_args()
    c=GSBSConfig(a.kmax,n_jobs=a.workers,mode="throughput")
    gpu=a.backend.endswith("cuda")
    if gpu:
        import torch
    rows=[]
    for b in map(int,a.batch_sizes.split(',')):
        xs=[np.random.default_rng(200+i).normal(size=(a.T,a.P)) for i in range(b)]
        # Warmup and validation use same precision/config; no timed reference mixed in.
        expected=[fit(x,replace(c,n_jobs=1)) for x in xs]
        fit_many(xs,c,a.backend,batch_size=b)
        for repeat in range(3):
            if gpu: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start=perf_counter(); results=fit_many(xs,c,a.backend,batch_size=b)
            if gpu: torch.cuda.synchronize()
            elapsed=perf_counter()-start
            row=dict(batch_size=b,repeat=repeat,workers=a.workers,backend=a.backend,shape=[a.T,a.P],kmax=a.kmax,
                     wall_s=elapsed,fits_per_minute=60*b/elapsed,bounds_equal=all(np.array_equal(r.all_bounds,e.all_bounds) for r,e in zip(results,expected)),
                     gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None,
                     gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu else None)
            rows.append(row); print(row,flush=True)
            Path(a.output).write_text(json.dumps(rows,indent=2))


if __name__=="__main__": main()
