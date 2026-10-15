"""Kernel timing and profiler trace; do not substitute these for end-to-end fit."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from hpgsbs import fit,GSBSConfig
from hpgsbs.backends.triton_cuda import candidate_scores
from hpgsbs.core.candidate_search import full_scores
from time import perf_counter


def main():
    prefix=sys.argv[1] if len(sys.argv)>1 else ""
    rows=[]
    for T in (450,900,1500,2100):
        q=torch.randn((T+1,T+1),device="cuda",dtype=torch.float64)
        qh=q.cpu().numpy(); edges=np.array([0,T]); allowed=np.ones(T,dtype=bool); opt=np.empty((0,0),dtype=np.int64)
        full_scores(qh,edges,0,T,allowed,opt,True,T); candidate_scores(q,0,T,True)
        for kind in ("numba","triton","torch"):
            times=[]
            for repeat in range(6):
                torch.cuda.synchronize(); start=perf_counter()
                if kind=="numba": out=full_scores(qh,edges,0,T,allowed,opt,True,T)
                elif kind=="triton": out=candidate_scores(q,0,T,True)
                else:
                    idx=torch.arange(1,T,device="cuda")
                    out=q[0,idx][:,None]+q[idx[:,None],idx[None,:]]+q[idx,T][None,:]-q[0,T]
                    out=torch.where(idx[:,None]<idx[None,:],out,-torch.inf)
                torch.cuda.synchronize(); times.append(perf_counter()-start)
            rows.append(dict(T=T,kernel=kind,median_s=float(np.median(times[1:])),times_s=times))
    Path(f"results/benchmarks/{prefix}kernels.json").write_text(json.dumps(rows,indent=2))
    x=np.random.default_rng(1).normal(size=(450,16)); c=GSBSConfig(20)
    for backend in ("torch-cuda","triton-cuda"):
        fit(x,c,backend)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            fit(x,c,backend)
        prof.export_chrome_trace(f"results/benchmarks/{prefix}{backend}_profile.json")
        Path(f"results/benchmarks/{prefix}{backend}_profile.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=30))


if __name__=="__main__": main()
