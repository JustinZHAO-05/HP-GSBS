"""Actual CUDA component timing, including metadata transfer and final sync.

This is not a full-fit benchmark or evidence of GPU-resident GSBS parity.
"""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from hpgsbs.core.interval_scores import IntervalScores
from hpgsbs.core.streaming_candidates import stream_q
from hpgsbs.backends.resident_candidates import candidate_summary
from threadpoolctl import threadpool_limits


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True); rows=[]
    with threadpool_limits(limits=1):
        for t,p in [(450,16),(1336,64),(2100,64)]:
            x=np.random.default_rng(20261016).normal(size=(t,p))
            q=IntervalScores(x).Q; device_q=torch.as_tensor(q,device="cuda")
            for cuts in ([],list(range(100,t,100))):
                edges=np.array([0,*cuts,t]); allowed=np.ones(t,dtype=bool); opt=np.empty((0,0),np.int64)
                for double in (False,True):
                    args=(edges,0,t,allowed,opt,double,t)
                    cpu=stream_q(q,*args); gpu=candidate_summary(device_q,q,*args)
                    assert cpu[0]==gpu[0]
                    np.testing.assert_allclose(cpu[1:3],gpu[1:3],atol=1e-12,rtol=1e-12)
                    times={}
                    for method in ("numba","triton-resident"):
                        samples=[]
                        for repeat in range(11):
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                            start=perf_counter()
                            if method=="numba": stream_q(q,*args)
                            else: candidate_summary(device_q,q,*args)
                            torch.cuda.synchronize()
                            samples.append(perf_counter()-start)
                        times[method]=dict(samples_s=samples,median_s=float(np.median(samples[1:])))
                    rows.append(dict(shape=[t,p],states=len(edges)-1,double=double,candidates=cpu[-1],
                        winner_equal=True,summary_score_error=abs(cpu[1]-gpu[1]),times=times,
                        gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved()))
                    out.write_text(json.dumps(rows,indent=2)); print(rows[-1],flush=True)


if __name__=="__main__": main()
