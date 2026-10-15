import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from .config import GSBSConfig
from .core.interval_scores import validate_input,conditioning_reason


def fit(x, config, backend="cpu-fast", *, y=None):
    if backend not in ("reference","cpu-fast","torch-cuda","triton-cuda"):
        raise ValueError(f"Unknown backend {backend}")
    if isinstance(config,dict): config=GSBSConfig(**config)
    x=validate_input(x,config.kmax)
    if y is not None:
        y=validate_input(y)
        if len(y)!=len(x): raise ValueError("x and y must have same time axis")
    with threadpool_limits(limits=config.blas_threads):
        reason=conditioning_reason(x,config.dtype)
        if y is not None: reason=reason or conditioning_reason(y,config.dtype)
        if reason and backend!="reference":
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
            return fit_cpu(x,config,y)
        if backend in ("torch-cuda","triton-cuda"):
            from .backends.torch_cuda import fit_gpu
            return fit_gpu(x,config,y,backend=backend)
        raise ValueError(f"Unknown backend {backend}")
