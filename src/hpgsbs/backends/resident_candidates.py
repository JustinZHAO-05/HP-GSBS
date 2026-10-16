"""Experimental device-resident Q lookup and bounded stable reduction.

Explicit experimental option: fit retains its sequential CPU controller and
uses this component only for large unrestricted double-candidate domains.
Candidate coordinates are generated on device; only six winner diagnostics
return to the host. At most 128 program summaries, never a dense score array.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from ..core.streaming_candidates import gate_counts


@triton.jit
def _summary(Q,META,OUT,BASE,START,END,T,N:tl.constexpr,DOUBLE:tl.constexpr,
             RESTRICTED:tl.constexpr,B0,BLOCK:tl.constexpr,PROGRAMS:tl.constexpr):
    pid=tl.program_id(0)
    lane=tl.arange(0,BLOCK)
    n=END-START
    total=n*n if DOUBLE else n
    best=tl.full((),-1,tl.float64)
    runner=tl.full((),-1,tl.float64)
    winner=tl.full((),0,tl.int64)
    nanflat=tl.full((),9223372036854775807,tl.int64)
    changed=tl.full((),0,tl.int32)
    nonfinite=tl.full((),0,tl.int32)
    count=tl.full((),0,tl.int64)
    base=tl.load(BASE)
    for tile in range(pid*BLOCK,total,PROGRAMS*BLOCK):
        f=(tile+lane).to(tl.int64)
        i=START+(f//n if DOUBLE else f)
        j=START+f%n
        a=tl.load(META+(i-START)*5+0,f<total,0)
        b=tl.load(META+(i-START)*5+1,f<total,0)
        allowed=tl.load(META+(i-START)*5+2,f<total,0)
        valid=(f<total)&(i>a)&(i<b)&(allowed>0)
        if DOUBLE:
            valid=valid&(j>i)&(j<b)
            if RESTRICTED:
                first=tl.load(META+(i-START)*5+3,f<total,0)
                second=tl.load(META+(j-START)*5+4,f<total,0)
                valid=valid&(first+B0>=1)&(first+second>=2)
        parent=tl.load(Q+a*N+b,valid,0)
        left=tl.load(Q+a*N+i,valid,0)
        if DOUBLE:
            middle=tl.load(Q+i*N+j,valid,0)
            right=tl.load(Q+j*N+b,valid,0)
            v=(((base-parent)+left)+middle)+right
        else:
            right=tl.load(Q+i*N+b,valid,0)
            v=((base-parent)+left)+right
        v=v*T/n
        v=tl.where(valid,v,-1.)
        isnan=v!=v
        nanflat=tl.minimum(nanflat,tl.min(tl.where(isnan,f,9223372036854775807),0))
        changed=changed|tl.max((valid&(v!=-1)).to(tl.int32),0)
        nonfinite=nonfinite|tl.max((valid&(isnan|(tl.abs(v)==float('inf')))).to(tl.int32),0)
        count+=tl.sum(valid.to(tl.int64),0)
        numeric=tl.where(isnan,-float('inf'),v)
        top=tl.max(numeric,0)
        firstflat=tl.min(tl.where((numeric==top)&(f<total),f,9223372036854775807),0)
        secondtop=tl.max(tl.where(f==firstflat,-float('inf'),numeric),0)
        better=(top>best)|((top==best)&(firstflat<winner))
        runner=tl.where(better,tl.maximum(best,secondtop),tl.maximum(runner,top))
        winner=tl.where(better,firstflat,winner)
        best=tl.maximum(best,top)
    tl.store(OUT+pid*8+0,best)
    tl.store(OUT+pid*8+1,winner.to(tl.float64))
    tl.store(OUT+pid*8+2,runner)
    tl.store(OUT+pid*8+3,nanflat.to(tl.float64))
    tl.store(OUT+pid*8+4,changed.to(tl.float64))
    tl.store(OUT+pid*8+5,nonfinite.to(tl.float64))
    tl.store(OUT+pid*8+6,count.to(tl.float64))


def candidate_summary(q,q_host,edges,start,end,allowed,opt,double,T):
    """Same summary contract as stream_q; Q is already resident and FP64."""
    start,end,T=int(start),int(end),int(T)
    if q.dtype!=torch.float64 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("Require contiguous resident CUDA FP64 Q")
    n=end-start
    meta=np.zeros((n,5),dtype=np.int64)
    for a,b in zip(edges[:-1],edges[1:]):
        meta[a-start:b-start,:2]=(a,b)
    first,second=gate_counts(opt,end)
    meta[:,2]=allowed
    meta[:,3]=first[start:end]; meta[:,4]=second[start:end]
    base=0.
    for a,b in zip(edges[:-1],edges[1:]): base+=q_host[a,b]
    device_meta=torch.as_tensor(meta,device=q.device)
    device_base=torch.tensor(base,device=q.device,dtype=torch.float64)
    programs=min(128,triton.cdiv(n*n if double else n,256))
    out=torch.empty((programs,8),device=q.device,dtype=torch.float64)
    _summary[(programs,)](q,device_meta,out,device_base,start,end,T,q.shape[0],double,opt.shape[1]==2,
                          int(second[0]),256,programs,enable_fp_fusion=False)
    # Reduction stays on device. Earliest NaN wins independently of tile order.
    best=out[:,0].max()
    winning=torch.where(out[:,0]==best,out[:,1],float('inf')).min()
    nanflat=out[:,3].min()
    hasnan=nanflat<9e18
    runner=torch.maximum(out[:,2].max(),torch.where(out[:,1]!=winning,out[:,0],-float('inf')).max())
    result=torch.stack([torch.where(hasnan,nanflat,winning),
        torch.where(hasnan,float('nan'),best),torch.where(hasnan,float('nan'),runner),out[:,5].max(),out[:,4].max(),out[:,6].sum()])
    f,v,r,nonfinite,changed,count=result.cpu().numpy()
    return int(f),float(v),float(r),bool(nonfinite),bool(changed),int(count)
