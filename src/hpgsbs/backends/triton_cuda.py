"""Optional Triton kernels. This module is never imported on CPU-only paths."""
import torch
try:
    import triton
    import triton.language as tl
except ImportError as e:
    raise RuntimeError("Triton unavailable. Use torch-cuda/cpu-fast, or the documented WSL Linux environment") from e


@triton.jit
def _interval(A,B,Q,T:tl.constexpr,P:tl.constexpr,F:tl.constexpr,PAIRS:tl.constexpr):
    pair=tl.program_id(0)*PAIRS+tl.arange(0,PAIRS)
    batch=tl.program_id(1)
    n=T+1
    a=pair//n; b=pair%n
    f=tl.arange(0,F)[None,:]
    valid=(b[:,None]>a[:,None])&(f<P)&(pair[:,None]<n*n)
    da=tl.load(A+batch*n*P+b[:,None]*P+f,valid,0)-tl.load(A+batch*n*P+a[:,None]*P+f,valid,0)
    db=tl.load(B+batch*n*P+b[:,None]*P+f,valid,0)-tl.load(B+batch*n*P+a[:,None]*P+f,valid,0)
    numerator=tl.sum(da*db,1)
    norm=tl.sqrt(tl.sum(da*da,1))
    score=numerator/(T*tl.sqrt(tl.full((),P-1,tl.float64))*norm)
    tl.store(Q+batch*n*n+pair,tl.where(b>a,score,0.0),pair<n*n)


def interval_matrix(a,b,T,P):
    q=torch.empty((a.shape[0],T+1,T+1),device=a.device,dtype=a.dtype)
    pairs=16 if P<=256 else 4
    _interval[(triton.cdiv((T+1)**2,pairs),a.shape[0])](a,b,q,T,P,triton.next_power_of_2(P),pairs,num_warps=4,enable_fp_fusion=False)
    return q


@triton.jit
def _candidate(Q,OUT,A:tl.constexpr,B:tl.constexpr,N:tl.constexpr,DOUBLE:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    L=B-A-1
    i=A+1+(idx//L if DOUBLE else idx)
    j=A+1+idx%L
    valid=(idx<(L*L if DOUBLE else L))
    if DOUBLE: valid=valid&(j>i)
    v=tl.load(Q+A*N+i,valid,0)
    if DOUBLE:
        v+=tl.load(Q+i*N+j,valid,0)+tl.load(Q+j*N+B,valid,0)
    else: v+=tl.load(Q+i*N+B,valid,0)
    v-=tl.load(Q+A*N+B)
    tl.store(OUT+idx,tl.where(valid,v,-float('inf')),idx<(L*L if DOUBLE else L))


def candidate_scores(q,a,b,double=False):
    """Fused legal scoring; flat row-major order, torch.argmax chooses first tie.

    Experimental microbenchmark path. Production controller uses Numba because
    transferring a dense candidate matrix each round is not assumed beneficial.
    """
    size=(b-a-1)**(2 if double else 1)
    out=torch.empty(size,device=q.device,dtype=q.dtype)
    if size: _candidate[(triton.cdiv(size,256),)](q,out,a,b,q.shape[-1],double,256,enable_fp_fusion=False)
    return out
