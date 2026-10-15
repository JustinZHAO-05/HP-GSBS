"""Batched CUDA precomputation with a shared, audited CPU search controller.

All arrays cross the device boundary once per precomputed batch, not per K.
This deliberately keeps branch-heavy sequential refinement on the CPU.
"""
from time import perf_counter
import numpy as np
from ..core.tdistance import TDistance
from .cpu_fast import run_search


def require_cuda():
    try: import torch
    except ImportError as e: raise RuntimeError("Install optional torch; use cpu-fast or the documented WSL environment") from e
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; use cpu-fast explicitly")
    return torch


class HostQ:
    def __init__(self,q): self.Q=q
    def __call__(self,a,b): return self.Q[a,b]


def prefix2_gpu(torch,x):
    return torch.nn.functional.pad(x.cumsum(-2).cumsum(-1),(1,0,1,0))


def precompute(xs, configs, ys=None, backend="torch-cuda"):
    torch=require_cuda()
    dtype=torch.float64 if configs[0].dtype=="float64" else torch.float32
    if any(c.q_strategy!="full" for c in configs):
        raise ValueError("CUDA currently requires q_strategy='full'; CPU supports lazy interval scores")
    torch.cuda.synchronize()
    start=perf_counter()
    old_precision=torch.get_float32_matmul_precision()
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        x=torch.as_tensor(np.stack(xs),device="cuda",dtype=dtype)
        y=x if ys is None else torch.as_tensor(np.stack(ys),device="cuda",dtype=dtype)
        torch.cuda.synchronize(); transfer_s=perf_counter()-start
        ts=perf_counter()
        B,T,P=x.shape
        centered=x-x.mean(-1,keepdim=True)
        z=centered/x.std(-1,correction=1,keepdim=True)
        zero=torch.zeros((B,1,P),device=x.device,dtype=dtype)
        A=torch.cat((zero,centered.cumsum(1)),1)
        C=torch.cat((zero,z.cumsum(1)),1)
        if backend=="triton-cuda":
            from .triton_cuda import interval_matrix
            q=interval_matrix(A,C,T,P)
        else:
            # Batched Gram formula, O(B T²) storage, no T x T x P tensor.
            aa=torch.bmm(A,A.transpose(1,2))
            ac=torch.bmm(A,C.transpose(1,2))
            da=torch.diagonal(aa,dim1=1,dim2=2)
            dc=torch.diagonal(ac,dim1=1,dim2=2)
            norm2=da[:,:,None]+da[:,None,:]-2*aa
            numerator=dc[:,:,None]+dc[:,None,:]-ac-ac.transpose(1,2)
            q=numerator/(T*np.sqrt(P-1)*torch.sqrt(norm2))
            # Short interval Gram subtraction can cancel: recompute all intervals
            # up to 16 samples by direct differences, no arbitrary epsilon.
            for gap in range(1,min(T,16)+1):
                a=torch.arange(T+1-gap,device=x.device); b=a+gap
                ad=A[:,b]-A[:,a]; cd=C[:,b]-C[:,a]
                q[:,a,b]=(ad*cd).sum(-1)/(T*np.sqrt(P-1)*torch.linalg.vector_norm(ad,dim=-1))
            q=torch.triu(q,diagonal=1)
            del aa,ac,norm2,numerator
        yz=(y-y.mean(-1,keepdim=True))/y.std(-1,correction=1,keepdim=True)
        yz=yz-yz.mean(-1,keepdim=True)
        r=torch.bmm(yz,yz.transpose(1,2))/(y.shape[-1]-1)
        mask=torch.triu(torch.ones((T,T),device=x.device,dtype=torch.bool),configs[0].dmin)
        r=torch.where(mask,r,0)
        pn=prefix2_gpu(torch,mask.to(torch.int64))
        ps=prefix2_gpu(torch,r)
        pq=prefix2_gpu(torch,r*r)
        torch.cuda.synchronize(); compute_s=perf_counter()-ts
        ts=perf_counter()
        qh=q.cpu().numpy(); nh=pn.cpu().numpy(); sh=ps.cpu().numpy(); qsh=pq.cpu().numpy()
        torch.cuda.synchronize(); transfer_s+=perf_counter()-ts
        stats=dict(precompute_s=perf_counter()-start,gpu_precompute_s=compute_s,transfer_s=transfer_s,
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return [(HostQ(qh[i]),TDistance(xs[i] if ys is None else ys[i],configs[i].dmin,(nh,sh[i],qsh[i]))) for i in range(B)],stats
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32=old_tf32


def iter_gpu(xs,configs,ys=None,backend="torch-cuda"):
    start=perf_counter()
    prepared,stats=precompute(xs,configs,ys,backend)
    for i,(x,c,(q,td)) in enumerate(zip(xs,configs,prepared)):
        result=run_search(x,c,q,td,backend=backend,timings=stats)
        result.model.y=None if ys is None else ys[i]
        result.metadata.update(elapsed_s=perf_counter()-start,batch_size=len(xs),batch_index=i,
            tdistance_input="x" if ys is None else "y",y_shape=None if ys is None else list(ys[i].shape),
            execution="batched CUDA precompute; independent sequential CPU controllers",
            gpu_peak_scope="process high-water since last reset; benchmark resets per timed call",
            precision_detail="FP64 throughout" if c.dtype=="float64" else "FP32 CUDA Q/correlation/prefixes; FP64 CPU controller and output patterns; FP64 exact fallback",
            tf32=False, **{k:v for k,v in stats.items() if k.startswith("gpu_peak")})
        yield result


def fit_gpu(x,config,y=None,backend="torch-cuda"):
    return next(iter_gpu([x],[config],None if y is None else [y],backend))
