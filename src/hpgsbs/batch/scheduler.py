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
from .memory_planner import automatic_batch_size,estimated_bytes


def _fit_worker(args):
    from ..api import fit
    x,c,backend=args
    return fit(x,c,backend)


def iter_many(recordings,configs,backend,batch_size="auto"):
    from ..api import fit
    if backend not in ("reference","cpu-fast","torch-cuda","triton-cuda"):
        raise ValueError(f"Unknown backend {backend}")
    if len(recordings)!=len(configs): raise ValueError("One config required per recording")
    if backend.endswith("cuda") and any(c.q_strategy!="full" or c.td_backend!="prefix2d" for c in configs):
        raise ValueError("CUDA batches require full Q and prefix2d TD")
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
    xs=recordings  # retain file-backed arrays; snapshot only the current batch
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
                validated={i:np.array(validate_input(xs[i],configs[i].kmax),copy=True) for i in chunk}
                for i,xv in validated.items():
                    xv.flags.writeable=False
                    if estimated_bytes(*xv.shape)>configs[i].max_workspace_bytes:
                        from ..core.budgets import ResourceBudgetExceeded
                        raise ResourceBudgetExceeded("GPU precompute exceeds workspace budget",dict(estimated_bytes=estimated_bytes(*xv.shape)))
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


TASK_SIGNATURE_VERSION=1


def _execution_identity(task,backend):
    """Canonical requested policy; bump signature schema if its meaning changes."""
    from .. import __version__
    return dict(signature_version=TASK_SIGNATURE_VERSION,software_version=__version__,
                config=asdict(GSBSConfig(**task["config"])),backend=backend)


def _task_key(task,backend):
    path=Path(task["path"]).resolve(); stat=path.stat()
    payload=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,execution=_execution_identity(task,backend),
                 y=task.get("y"),times=task.get("times"))
    for name in ("y","times"):
        if task.get(name):
            st=Path(task[name]).stat(); payload[name]=(str(Path(task[name]).resolve()),st.st_size,st.st_mtime_ns)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def _validate_resume(path,key):
    """A scientific result is reusable as an execution cache only on success."""
    result=GSBSResult.load(path)
    if result.metadata.get("status")!="success":
        raise FileExistsError("Existing result is not successful; preserve it and choose a new output directory")
    if result.metadata.get("task_key")!=key:
        if "execution_identity" not in result.metadata:
            raise FileExistsError("Legacy resume signature cannot verify this software/config execution; "
                                  "load the existing result for analysis or choose a new output directory to rerun")
        raise FileExistsError("Existing result belongs to different input/config/software; choose a new output directory")


def _stamp_execution(result,task,backend,key):
    result.metadata.update(task_key=key,recording_id=task["id"],
                           execution_identity=_execution_identity(task,backend))


def _run_task(task,output,backend,n_jobs):
    from ..api import fit
    identifier=task["id"]
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",identifier) or identifier in (".",".."):
            raise ValueError("Task id must be a safe filename")
        path=Path(output)/(identifier+".npz"); key=_task_key(task,backend)
        if path.exists():
            _validate_resume(path,key)
            return dict(id=identifier,status="skipped",output=str(path))
        x=np.load(task["path"],allow_pickle=False)
        y=np.load(task["y"],allow_pickle=False) if task.get("y") else None
        c=GSBSConfig(**task["config"])
        result=fit(x,replace(c,n_jobs=1),backend,y=y)
        _stamp_execution(result,task,backend,key)
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
                    _validate_resume(path,key)
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
                _stamp_execution(r,t,backend,key)
                r.save(path,times=np.load(t["times"]) if t.get("times") else None)
                done.add(i); record(dict(id=t["id"],status="success",output=str(path),elapsed_s=r.metadata["elapsed_s"]))
        except Exception as e:
            # Independent retry isolates a single input/kernel failure; saved fits persist.
            for i,(t,_,_) in enumerate(pending):
                if i not in done: record(_run_task(t,out,backend,1))
    return receipts
