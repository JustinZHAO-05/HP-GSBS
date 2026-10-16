"""Interleaved, persistent-process comparison of actual source snapshots.

Each worker loads its own PYTHONPATH. Warm each workload/backend, then rotate
three revisions four times (first is additional warmup). No concurrent fits.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from time import perf_counter
import numpy as np


def worker(fixtures):
    from hpgsbs import fit,GSBSConfig
    from threadpoolctl import threadpool_limits
    import psutil
    for line in sys.stdin:
        job=json.loads(line); name,T,P,K=job['workload']; backend=job['backend']
        x=np.load(Path(fixtures)/'0.npy').astype(float) if name=='BBBD0' else np.random.default_rng(20261016).normal(size=(T,P))
        c=GSBSConfig(K,dmin=5 if name=='BBBD0' else 1)
        gpu=backend.endswith('cuda')
        if gpu:
            import torch
            torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with threadpool_limits(1):
            start=perf_counter(); r=fit(x,c,backend)
            if gpu:torch.cuda.synchronize()
            wall=perf_counter()-start
        payload=dict(wall_s=wall,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            bounds_sha256=hashlib.sha256(r.all_bounds.tobytes()).hexdigest(),nstates=int(r.nstates),
            tdists=r.tdists.tolist(),rss_bytes=psutil.Process().memory_info().rss,
            patterns=r.state_patterns.tolist(),strengths=r.strengths.tolist(),
            gpu_allocated=torch.cuda.max_memory_allocated() if gpu else None,
            gpu_reserved=torch.cuda.max_memory_reserved() if gpu else None)
        print(json.dumps(payload),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--output')
    p.add_argument('--snapshots',nargs=3);p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:return worker(a.fixtures)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True); rows=[]; procs=[]
    variants=['v0.1.0','accepted-9309874','final']
    try:
        for path in a.snapshots:
            env=dict(os.environ,PYTHONPATH=str(Path(path).resolve()/'src'))
            procs.append(subprocess.Popen([sys.executable,__file__,'--worker','--fixtures',a.fixtures],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env))
        hashes={v:{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (Path(root)/'src').rglob('*.py')} for v,root in zip(variants,a.snapshots)}
        (out/'sources.json').write_text(json.dumps(hashes,indent=2))
        for workload in [('Small',450,16,20),('Medium',900,32,30),('Large',1500,64,40),('XL',2100,64,50),('BBBD0',1336,64,131)]:
            for backend in ('cpu-fast','torch-cuda','triton-cuda'):
                reference=None
                for iteration in range(5):
                    for index in np.roll(np.arange(3),iteration%3):
                        proc=procs[index];proc.stdin.write(json.dumps(dict(workload=workload,backend=backend))+'\n');proc.stdin.flush()
                        line=proc.stdout.readline()
                        if not line:raise RuntimeError('Revision worker exited')
                        result=json.loads(line)
                        if reference is None:reference=result
                        result['passed']=bool(result['bounds_sha256']==reference['bounds_sha256'] and result['nstates']==reference['nstates'] and all(np.allclose(result[key],reference[key],atol=1e-8,rtol=1e-8,equal_nan=True) for key in ('tdists','patterns','strengths')))
                        result.update(workload=workload[0],backend=backend,revision=variants[index],iteration=iteration,warmup=iteration<2)
                        rows.append(result);(out/'trials.json').write_text(json.dumps(rows,separators=(',',':')))
                        print(workload[0],backend,variants[index],iteration,result['wall_s'],result['passed'],flush=True)
                        if not result['passed']:raise AssertionError(result)
    finally:
        for proc in procs:
            proc.stdin.close();proc.wait()


if __name__=='__main__':main()
