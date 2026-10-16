"""Research-only full-fit incremental attempt and no-Q GPU component prototype.

Not an exported backend. Records failures and slower results; no auto dispatch.
"""
import argparse,json,hashlib
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from hpgsbs import fit,GSBSConfig,GSBSResult
from hpgsbs.validation import compare
from hpgsbs.core.interval_scores import IntervalScores
from hpgsbs.core.feature_moments import FeatureMomentsTDistance
from hpgsbs.core.incremental_tdistance import IncrementalTDistance
from hpgsbs.backends.cpu_fast import run_search


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);rows=[]
    def save(): (out/'research.json').write_text(json.dumps(rows,indent=2))
    with threadpool_limits(1):
        for name,x,k in [('small',np.random.default_rng(8).normal(size=(220,16)),30),
                         ('BBBD0',np.load(Path(a.fixtures)/'0.npy').astype(float),131)]:
            c=GSBSConfig(k,trace=True,dmin=5,q_strategy='streaming',td_backend='feature-moments')
            ref=fit(x,c)
            for iteration in range(4):
                for kind in (['direct','incremental'] if iteration%2==0 else ['incremental','direct']):
                    td=(FeatureMomentsTDistance if kind=='direct' else IncrementalTDistance)(x,5)
                    start=perf_counter();r=run_search(x,c,IntervalScores(x,'streaming'),td);wall=perf_counter()-start
                    report=compare(ref,r,level='finetune-move-trace')
                    rows.append(dict(kind='incremental-full-fit-attempt',workload=name,mode=kind,iteration=iteration,
                        warmup=iteration==0,wall_s=wall,report=report,commits=getattr(td,'commits',None),
                        final_cache_intervals=len(getattr(td,'moments',{}))))
                    save();print(name,kind,iteration,wall,report['passed'],flush=True)
                    if not report['passed']:raise AssertionError(report)
        # Near-constant pair distributions: high precision is a separate target,
        # computed from the exact FP64 U values, NOT an official trajectory certificate.
        for noise in (1.,1e-2,1e-4,1e-6,0.):
            x=np.tile(np.arange(8,dtype=float),(45,1))+np.random.default_rng(8).normal(size=(45,8))*noise
            td=FeatureMomentsTDistance(x,5,tile_size=7);b=np.zeros(45,bool);b[[15,30]]=True
            from hpgsbs.core.tdistance import TDistance
            native=TDistance(x,5).exact(b)
            u=td.u.astype(np.longdouble); labels=np.cumsum(b);w=[];v=[]
            for i in range(45):
                for j in range(i+5,45):
                    kind=labels[j]-labels[i]
                    if kind==0:w.append(u[i]@u[j]/7)
                    if kind==1:v.append(u[i]@u[j]/7)
            w,v=np.array(w),np.array(v)
            with np.errstate(all='ignore'): hp=(w.mean()-v.mean())/np.sqrt(w.var(ddof=1)/len(w)+v.var(ddof=1)/len(v))
            xl=x.astype(np.longdouble);xl-=xl.mean(1,keepdims=True)
            ul=xl/np.sqrt((xl*xl).sum(1,keepdims=True)/7);wl=[];al=[]
            for i in range(45):
                for j in range(i+5,45):
                    kind=labels[j]-labels[i]
                    if kind==0:wl.append(ul[i]@ul[j]/7)
                    if kind==1:al.append(ul[i]@ul[j]/7)
            wl,al=np.array(wl),np.array(al)
            with np.errstate(all='ignore'): raw_hp=(wl.mean()-al.mean())/np.sqrt(wl.var(ddof=1)/len(wl)+al.var(ddof=1)/len(al))
            try:value=td(b);status='accepted'
            except FloatingPointError:value=None;status='refused-uncertified'
            rows.append(dict(kind='numeric-targets',noise=noise,native=float(native),
                fp64_U_longdouble_pair_target=float(hp),moments=value,status=status,diagnostics=td.last_diagnostics));save()
            rows[-1].update(raw_FP64_X_longdouble_target=float(raw_hp),
                target_scope='higher precision approximation; not interval enclosure or native parity certificate',
                common_anchor_one_variances=[float((w-1).var(ddof=1)),float((v-1).var(ddof=1))]);save()
        import torch
        for T,P,C in [(1200,32,256),(5000,8,256),(5000,64,4096)]:
            x=np.random.default_rng(87).normal(size=(T,P));q=IntervalScores(x,'streaming')
            rng=np.random.default_rng(8);a0=rng.integers(0,T-2,size=C);b0=np.minimum(T,a0+rng.integers(1,T,size=C))
            A=torch.tensor(q.A,device='cuda');Z=torch.tensor(q.B,device='cuda')
            timings={'cpu':[],'gpu':[]};errors=[]
            for iteration in range(7):
                start=perf_counter();expected=q(a0,b0);timings['cpu'].append(perf_counter()-start)
                torch.cuda.synchronize();start=perf_counter()
                aa=torch.as_tensor(a0,device='cuda');bb=torch.as_tensor(b0,device='cuda')
                s=A[bb]-A[aa];s=s-s.mean(1,keepdim=True);z=Z[bb]-Z[aa]
                values=((s*z).sum(1)/(T*np.sqrt(P-1)*torch.linalg.vector_norm(s,dim=1))).cpu().numpy()
                torch.cuda.synchronize();timings['gpu'].append(perf_counter()-start)
                errors.append(float(np.max(np.abs(values-expected))))
            rows.append(dict(kind='no-Q-GPU-prefix-prototype',shape=[T,P],chunk=C,
                cpu_median_s=float(np.median(timings['cpu'][1:])),gpu_median_s=float(np.median(timings['gpu'][1:])),
                max_score_error=max(errors),prefix_upload_excluded=True,metadata_upload_and_output_transfer_included=True,
                full_fit_certified=False));save()
        for T,P in [(1200,32),(5000,64)]:
            x=np.random.default_rng(3).normal(size=(T,P));td=FeatureMomentsTDistance(x)
            b=np.zeros(T,bool);b[np.arange(100,T,100)]=True
            expected=td.stats(b);u=torch.tensor(td.u,device='cuda');edges=np.r_[0,np.flatnonzero(b),T]
            times=[]
            for iteration in range(5):
                torch.cuda.synchronize();start=perf_counter();w=torch.zeros(3,device='cuda',dtype=torch.float64);a=w.clone();previous=None
                for lo,hi in zip(edges[:-1],edges[1:]):
                    block=u[lo:hi];n=hi-lo;s=block.sum(0);M=block.T@block;d=(block*block).sum(1)
                    w+=torch.stack([torch.tensor(n*(n-1)/2,device='cuda'),(s@s-d.sum())/(2*(P-1)),((M*M).sum()-(d*d).sum())/(2*(P-1)**2)])
                    if previous is not None:
                        pn,ps,pM=previous;a+=torch.stack([torch.tensor(pn*n,device='cuda'),ps@s/(P-1),(pM*M).sum()/(P-1)**2])
                    previous=n,s,M
                actual=torch.stack([w,a]).cpu().numpy();torch.cuda.synchronize();times.append(perf_counter()-start)
            rows.append(dict(kind='GPU-moments-fixed-partition-prototype',shape=[T,P],states=len(edges)-1,
                gpu_median_s=float(np.median(times[1:])),max_stat_error=float(np.max(np.abs(actual-np.array(expected)))),
                dmin=1,full_fit_certified=False));save()


if __name__=='__main__':main()
