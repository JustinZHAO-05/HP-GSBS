from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from threadpoolctl import threadpool_limits
from ..config import GSBSConfig
from ..result import GSBSResult
from ..core.interval_scores import validate_input,conditioning_reason
from .bucketing import buckets
from .memory_planner import automatic_batch_size


def _fit_worker(args):
    from ..api import fit
    x,c,backend=args
    return fit(x,c,backend)


def iter_many(recordings,configs,backend,batch_size="auto"):
    from ..api import fit
    if backend not in ("reference","cpu-fast","torch-cuda","triton-cuda"):
        raise ValueError(f"Unknown backend {backend}")
    if len(recordings)!=len(configs): raise ValueError("One config required per recording")
    if not recordings: return
    if backend in ("cpu-fast","reference"):
        xs=[validate_input(x,c.kmax) for x,c in zip(recordings,configs)]
        workers=max((c.n_jobs for c in configs),default=1)
        if workers==1:
            for i,(x,c) in enumerate(zip(xs,configs)): yield i,fit(x,c,backend)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(_fit_worker,(x,replace(c,n_jobs=1),backend)):i for i,(x,c) in enumerate(zip(xs,configs))}
                for future in as_completed(futures): yield futures[future],future.result()
        return
    from ..backends.torch_cuda import require_cuda,iter_gpu
    torch=require_cuda()
    xs=recordings  # retain file-backed arrays; materialize only the current batch
    for group in buckets(xs,configs):
        x=xs[group[0]]; c=configs[group[0]]
        if batch_size=="auto":
            size=1 if c.mode=="latency" else automatic_batch_size(*x.shape,torch.cuda.mem_get_info()[0],c.memory_fraction,8 if c.dtype=="float64" else 4)
        else: size=int(batch_size)
        if size<1: raise ValueError("batch_size must be positive")
        pos=0
        while pos<len(group):
            chunk=group[pos:pos+size]
            try:
                validated={i:validate_input(xs[i],configs[i].kmax) for i in chunk}
                safe=[]
                for i in chunk:
                    if conditioning_reason(validated[i],configs[i].dtype): yield i,fit(validated[i],configs[i],backend)
                    else: safe.append(i)
                with threadpool_limits(limits=c.blas_threads):
                    if safe:
                        outputs=iter_gpu([validated[i] for i in safe],[configs[i] for i in safe],backend=backend)
                        for i,result in zip(safe,outputs): yield i,result
                pos+=len(chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size==1: raise
                size=max(1,size//2)


def fit_many(recordings,config,backend="torch-cuda",batch_size="auto"):
    recordings=list(recordings)
    if isinstance(config,dict): config=GSBSConfig(**config)
    configs=[config]*len(recordings) if isinstance(config,GSBSConfig) else list(config)
    configs=[GSBSConfig(**c) if isinstance(c,dict) else c for c in configs]
    results=[None]*len(recordings)
    for i,result in iter_many(recordings,configs,backend,batch_size): results[i]=result
    return results


def _task_key(task,backend):
    path=Path(task["path"]).resolve(); stat=path.stat()
    payload=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,config=task["config"],backend=backend,
                 y=task.get("y"),times=task.get("times"))
    for name in ("y","times"):
        if task.get(name):
            st=Path(task[name]).stat(); payload[name]=(str(Path(task[name]).resolve()),st.st_size,st.st_mtime_ns)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def _run_task(task,output,backend,n_jobs):
    from ..api import fit
    identifier=task["id"]
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",identifier) or identifier in (".",".."):
            raise ValueError("Task id must be a safe filename")
        path=Path(output)/(identifier+".npz"); key=_task_key(task,backend)
        if path.exists():
            result=GSBSResult.load(path)
            if result.metadata.get("task_key")==key and result.metadata["status"]=="success":
                return dict(id=identifier,status="skipped",output=str(path))
            raise FileExistsError("Existing result belongs to different input/config; choose a new output directory")
        x=np.load(task["path"],allow_pickle=False)
        y=np.load(task["y"],allow_pickle=False) if task.get("y") else None
        c=GSBSConfig(**task["config"])
        result=fit(x,replace(c,n_jobs=1),backend,y=y)
        result.metadata.update(task_key=key,recording_id=identifier)
        result.save(path,times=np.load(task["times"]) if task.get("times") else None)
        return dict(id=identifier,status="success",output=str(path),elapsed_s=result.metadata["elapsed_s"])
    except Exception as e:
        return dict(id=identifier,status="failed",error=f"{type(e).__name__}: {e}")


def run_manifest(manifest,output,backend="cpu-fast",n_jobs=1,batch_size="auto"):
    tasks=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else list(manifest)
    ids=[t["id"] for t in tasks]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate recording IDs")
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    receipts=[]
    # Each line is flushed immediately. Result files are the recovery authority.
    def record(receipt):
        receipts.append(receipt)
        with (out/"manifest.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(receipt)+"\n"); f.flush()
    if backend in ("cpu-fast","reference"):
        if n_jobs==1:
            for t in tasks: record(_run_task(t,out,backend,1))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures=[pool.submit(_run_task,t,out,backend,1) for t in tasks]
                for f in as_completed(futures): record(f.result())
    else:
        pending=[]; xs=[]; configs=[]
        # Existence and input validation failures are isolated from the batch.
        for t in tasks:
            try:
                if not re.fullmatch(r"[A-Za-z0-9_.-]+",t["id"]) or t["id"] in (".",".."):
                    raise ValueError("Unsafe task id")
                key=_task_key(t,backend); path=out/(t["id"]+".npz")
                if path.exists():
                    r=GSBSResult.load(path)
                    if r.metadata.get("task_key")!=key: raise FileExistsError("Existing output differs")
                    record(dict(id=t["id"],status="skipped",output=str(path))); continue
                if t.get("y"):
                    record(_run_task(t,out,backend,1)); continue
                c=GSBSConfig(**t["config"])
                x=np.load(t["path"],allow_pickle=False,mmap_mode="r")
                if x.ndim!=2: raise ValueError("Expected time x features NPY")
                pending.append((t,key,path)); xs.append(x); configs.append(c)
            except Exception as e: record(dict(id=t["id"],status="failed",error=str(e)))
        done=set()
        try:
            for i,r in iter_many(xs,configs,backend,batch_size):
                t,key,path=pending[i]
                r.metadata.update(task_key=key,recording_id=t["id"])
                r.save(path,times=np.load(t["times"]) if t.get("times") else None)
                done.add(i); record(dict(id=t["id"],status="success",output=str(path),elapsed_s=r.metadata["elapsed_s"]))
        except Exception as e:
            # Independent retry isolates a single input/kernel failure; saved fits persist.
            for i,(t,_,_) in enumerate(pending):
                if i not in done: record(_run_task(t,out,backend,1))
    return receipts
