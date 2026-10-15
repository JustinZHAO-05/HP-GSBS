from dataclasses import replace
from time import perf_counter
import threading
import psutil
import numpy as np
from .api import fit


def benchmark(x,config,backend,repeats=3,y=None):
    if repeats<1: raise ValueError("repeats must be positive")
    gpu=backend in ("torch-cuda","triton-cuda")
    if gpu:
        from .backends.torch_cuda import require_cuda
        torch=require_cuda()
    c=replace(config,trace=False)
    trials=[]
    for iteration in range(repeats+1):
        done=threading.Event(); peak=[psutil.Process().memory_info().rss]
        def monitor():
            while not done.wait(.005): peak[0]=max(peak[0],psutil.Process().memory_info().rss)
        thread=threading.Thread(target=monitor,daemon=True); thread.start()
        if gpu: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=perf_counter()
        try:
            result=fit(x,c,backend,y=y)
            if gpu: torch.cuda.synchronize()
            elapsed=perf_counter()-start
        finally:
            done.set(); thread.join()
        trials.append(dict(iteration=iteration,wall_s=elapsed,cpu_peak_rss_bytes=peak[0],**result.metadata))
    warm=[r["wall_s"] for r in trials[1:]]
    return dict(backend=backend,precision=config.dtype,shape=list(x.shape),algorithm=config.algorithm(),
        first_call_s=trials[0]["wall_s"],steady_median_s=float(np.median(warm)),fits_per_minute=60/float(np.median(warm)),
        note="First call includes in-process JIT/device setup; disk cache may exist. Wall excludes output serialization. RSS sampled every 5 ms, process total.",trials=trials)
