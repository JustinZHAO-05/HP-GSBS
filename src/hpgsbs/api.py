import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from .config import GSBSConfig
from .core.interval_scores import validate_input,conditioning_reason


def fit(x, config, backend="cpu-fast", *, y=None, progress=None, cancel=None):
    if backend not in ("reference","cpu-fast","torch-cuda","triton-cuda"):
        raise ValueError(f"Unknown backend {backend}")
    if isinstance(config,dict): config=GSBSConfig(**config)
    if backend in ("torch-cuda","triton-cuda") and (config.q_strategy!="full" or config.td_backend!="prefix2d"):
        raise ValueError("CUDA requires full Q and prefix2d TD; GPU low-memory full fits are deferred")
    if backend!="cpu-fast" and (progress is not None or cancel is not None):
        raise ValueError("Progress/cancellation callbacks currently require cpu-fast")
    if config.candidate_backend=="triton-resident" and backend=="cpu-fast":
        raise ValueError("triton-resident candidates require a CUDA backend and optional Triton")
    x=validate_input(x,config.kmax)
    # Snapshot prevents caller mutation from invalidating per-fit Q/Z/cache.
    x=np.array(x,copy=True); x.flags.writeable=False
    if y is not None:
        y=validate_input(y)
        if len(y)!=len(x): raise ValueError("x and y must have same time axis")
        y=np.array(y,copy=True); y.flags.writeable=False
    if backend!="reference":
        from .core.budgets import ResourceBudgetExceeded
        T,P=x.shape; Py=P if y is None else y.shape[1]
        estimate=(10*(T+1)**2*8 if config.q_strategy=="full" else 0)
        estimate += 10*T*T*8 if config.td_backend=="prefix2d" else (4*T*Py+4*Py*Py)*8
        estimate += (8*T*P+config.candidate_chunk_size*16*P)*8
        estimate += (config.kmax+2)*T*8
        if estimate>config.max_workspace_bytes:
            raise ResourceBudgetExceeded("Precompute exceeds workspace budget",dict(estimated_bytes=estimate,budget=config.max_workspace_bytes))
    with threadpool_limits(limits=config.blas_threads):
        reason=conditioning_reason(x,config.dtype)
        if y is not None: reason=reason or conditioning_reason(y,config.dtype)
        if reason and backend!="reference":
            if config.td_backend == "feature-moments":
                raise FloatingPointError("Low-memory mode cannot certify ill-conditioned rows: "+reason+
                                         "; dense reference fallback is disabled")
            from .reference import fit_reference
            result=fit_reference(x,replace(config,dtype="float64"),y)
            result.metadata.update(requested_backend=backend,requested_precision=config.dtype,fallback_reason=reason)
            return result
        if backend=="reference":
            if config.dtype!="float64": raise ValueError("Reference uses original float64 arithmetic")
            from .reference import fit_reference
            return fit_reference(x,config,y)
        if backend=="cpu-fast":
            from numba import set_num_threads
            set_num_threads(config.numba_threads)
            from .backends.cpu_fast import fit_cpu
            return fit_cpu(x,config,y,progress=progress,cancel=cancel)
        if backend in ("torch-cuda","triton-cuda"):
            from .backends.torch_cuda import fit_gpu
            return fit_gpu(x,config,y,backend=backend)
        raise ValueError(f"Unknown backend {backend}")
