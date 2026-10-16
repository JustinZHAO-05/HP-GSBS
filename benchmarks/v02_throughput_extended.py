"""Array/manifest throughput and actual generator completion timestamps."""
import argparse,json,tempfile,threading
from dataclasses import replace,asdict
from pathlib import Path
from time import perf_counter
import numpy as np
import psutil
from hpgsbs import fit,fit_many,GSBSConfig
from hpgsbs.batch.scheduler import iter_many,run_manifest
from hpgsbs.batch.memory_planner import estimated_bytes
from hpgsbs.validation import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);rows=[]
    import torch
    real=np.load(Path(a.fixtures)/'0.npy').astype(float)
    def save(): (out/'throughput.json').write_text(json.dumps(rows,separators=(',',':')))
    for workload in ('Small','highK','variable'):
        count=64
        xs=([np.random.default_rng(200+i).normal(size=((450,16) if workload=='Small' else (450+20*(i%3),16+4*(i%2)))) for i in range(count)]
            if workload!='highK' else [real.copy() for _ in range(count)])
        base=GSBSConfig(131 if workload=='highK' else 20,dmin=5 if workload=='highK' else 1,mode='throughput')
        expected=[fit(x,base) for x in (xs if workload!='highK' else xs[:1])]
        jobs=[('cpu-fast',w,8 if workload=='highK' else 32) for w in (1,2,4,8)]
        jobs += [(backend,1,b) for backend in ('torch-cuda','triton-cuda') for b in (1,2,4,8,16,32,64)]
        for backend,workers,batch in jobs:
            gpu=backend.endswith('cuda');torch.cuda.empty_cache()
            if gpu and estimated_bytes(*max((x.shape for x in xs[:batch])),batch)>torch.cuda.mem_get_info()[0]*.55:
                rows.append(dict(workload=workload,backend=backend,batch=batch,status='skipped-memory-admission'));save();continue
            c=replace(base,n_jobs=workers)
            for iteration in range(4):
                done=threading.Event();peak=[0];parent=psutil.Process()
                def monitor():
                    while not done.wait(.01):
                        total=0
                        for proc in [parent,*parent.children(recursive=True)]:
                            try:total+=proc.memory_info().rss
                            except psutil.Error:pass
                        peak[0]=max(peak[0],total)
                thread=threading.Thread(target=monitor,daemon=True);thread.start()
                if gpu:torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                start=perf_counter();completions=[];passed=True
                try:
                    for index,result in iter_many(xs[:batch],[c]*batch,backend,batch_size=batch):
                        completions.append(perf_counter()-start)
                        passed &= compare(expected[0 if workload=='highK' else index],result,level='output-only')['passed']
                    if gpu:torch.cuda.synchronize()
                    wall=perf_counter()-start
                finally:done.set();thread.join()
                rows.append(dict(workload=workload,backend=backend,workers=workers,batch=batch,iteration=iteration,
                    warmup=iteration==0,wall_s=wall,amortized_wall_s=wall/batch,fits_per_minute=60*batch/wall,
                    completion_latency_s=completions,validation_in_timed_scope=True,peak_process_tree_rss_bytes=peak[0],
                    gpu_allocated=torch.cuda.max_memory_allocated() if gpu else None,
                    gpu_reserved=torch.cuda.max_memory_reserved() if gpu else None,all_outputs_passed=bool(passed)))
                save();print(workload,backend,workers,batch,iteration,wall,passed,flush=True)
                if not passed or len(completions)!=batch:raise AssertionError('batch outputs')
        # Disk files stay in private temporary storage and are removed on exit.
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);tasks=[]
            for i,x in enumerate(xs[:4]):
                path=folder/f'input{i}.npy';np.save(path,x)
                tasks.append(dict(id=f'task{i}',path=str(path),config=asdict(base)))
            for backend in ('cpu-fast','torch-cuda','triton-cuda'):
                start=perf_counter();receipt=run_manifest(tasks,folder/backend,backend,n_jobs=2,batch_size=2);wall=perf_counter()-start
                start=perf_counter();resume=run_manifest(tasks,folder/backend,backend,n_jobs=2,batch_size=2);resume_wall=perf_counter()-start
                rows.append(dict(workload=workload,backend=backend,kind='manifest',wall_s=wall,resume_wall_s=resume_wall,
                    statuses=[r['status'] for r in receipt],resume_statuses=[r['status'] for r in resume]));save()
                assert all(r['status']=='success' for r in receipt) and all(r['status']=='skipped' for r in resume)
            start=perf_counter();rr=fit_many(xs[:2],base,'cpu-fast');wall=perf_counter()-start
            assert all(compare(expected[0 if workload=='highK' else i],r,level='output-only')['passed'] for i,r in enumerate(rr))
            rows.append(dict(workload=workload,kind='fit_many-array-api',wall_s=wall,batch=2,all_outputs_passed=True));save()


if __name__=='__main__':main()
