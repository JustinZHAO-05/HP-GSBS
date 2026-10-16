import json
import numpy as np
import pytest
from hpgsbs import fit,fit_many,GSBSConfig,GSBSResult
from hpgsbs.batch.scheduler import run_manifest


def test_roundtrip(tmp_path):
    x=np.random.default_rng(8).normal(size=(18,4))
    r=fit(x,GSBSConfig(7,trace=True)); p=tmp_path/"out.npz"
    r.save(p,times=np.arange(18)/5.12)
    with pytest.raises(FileExistsError): r.save(p)
    s=GSBSResult.load(p)
    np.testing.assert_array_equal(r.all_bounds,s.all_bounds)
    assert r.trace==s.trace


def test_y_roundtrip(tmp_path):
    rng=np.random.default_rng(122)
    x=rng.normal(size=(18,4)); y=rng.normal(size=(18,7))
    result=fit(x,GSBSConfig(6),y=y)
    path=tmp_path/"cv.npz"; result.save(path)
    restored=GSBSResult.load(path)
    np.testing.assert_array_equal(restored.y,y)
    assert restored.metadata["tdistance_input"]=="y"
    np.testing.assert_array_equal(restored.tdists,result.tdists)


def test_resume_and_failure(tmp_path):
    x=tmp_path/"x.npy"; np.save(x,np.random.default_rng(0).normal(size=(20,4)))
    tasks=[dict(id="ok",path=str(x),config=dict(kmax=5)),dict(id="bad",path=str(tmp_path/"missing.npy"),config=dict(kmax=5))]
    out=tmp_path/"out"
    a=run_manifest(tasks,out)
    assert [r["status"] for r in a]==["success","failed"]
    identity=GSBSResult.load(out/"ok.npz").metadata["execution_identity"]
    assert identity["signature_version"]==1
    assert identity["config"]["candidate_backend"]=="streaming"
    assert identity["backend"]=="cpu-fast"
    mtime=(out/"ok.npz").stat().st_mtime_ns
    b=run_manifest(tasks,out)
    assert b[0]["status"]=="skipped"
    assert (out/"ok.npz").stat().st_mtime_ns==mtime
    from dataclasses import asdict
    tasks[0]["config"]=asdict(GSBSConfig(5))
    assert run_manifest(tasks,out)[0]["status"]=="skipped"
    tasks[0]["config"]["kmax"]=6
    assert run_manifest(tasks,out)[0]["status"]=="failed"


def test_process_batch():
    xs=[np.random.default_rng(i).normal(size=(18+i,5)) for i in range(3)]
    c=GSBSConfig(7,n_jobs=2)
    many=fit_many(xs,c,"cpu-fast")
    for x,r in zip(xs,many): np.testing.assert_array_equal(r.all_bounds,fit(x,c).all_bounds)


def test_execution_signature_defaults_version_and_policy(tmp_path,monkeypatch):
    import hpgsbs
    from dataclasses import asdict
    from hpgsbs.batch.scheduler import _task_key
    path=tmp_path/"x.npy"; np.save(path,np.random.default_rng(5).normal(size=(18,4)))
    task=dict(id="rec",path=str(path),config=dict(kmax=5))
    key=_task_key(task,"cpu-fast")
    assert key==_task_key(dict(task,config=asdict(GSBSConfig(5))),"cpu-fast")
    assert key!=_task_key(dict(task,config=dict(kmax=5,candidate_backend="materialized")),"cpu-fast")
    assert key!=_task_key(task,"reference")
    monkeypatch.setattr(hpgsbs,"__version__","0.1.0")
    assert key!=_task_key(task,"cpu-fast")


@pytest.mark.parametrize("backend",["cpu-fast","reference","torch-cuda","triton-cuda"])
@pytest.mark.parametrize("cache",["legacy","old-version","failed","cancelled","missing-status","success"])
def test_manifest_resume_validation(tmp_path,monkeypatch,backend,cache):
    import hashlib
    from hpgsbs.batch import scheduler
    path=tmp_path/"x.npy"; x=np.random.default_rng(9).normal(size=(18,4)); np.save(path,x)
    task=dict(id="rec",path=str(path),config=dict(kmax=5))
    result=fit(x,GSBSConfig(5))
    scheduler._stamp_execution(result,task,backend,scheduler._task_key(task,backend))
    if cache=="legacy":
        # Exact pre-versioning (v0.1.0 and initial v0.2) signature structure.
        stat=path.stat()
        payload=dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                     config=task["config"],backend=backend,y=None,times=None)
        result.metadata["task_key"]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        del result.metadata["execution_identity"]
    elif cache=="old-version":
        with monkeypatch.context() as patch:
            import hpgsbs
            patch.setattr(hpgsbs,"__version__","0.1.0")
            scheduler._stamp_execution(result,task,backend,scheduler._task_key(task,backend))
    elif cache=="missing-status": del result.metadata["status"]
    elif cache!="success": result.metadata["status"]=cache
    out=tmp_path/"out"; target=out/"rec.npz"; result.save(target)
    before=target.read_bytes(); mtime=target.stat().st_mtime_ns
    rows=run_manifest([task],out,backend=backend)
    assert rows[0]["status"]==("skipped" if cache=="success" else "failed"),rows
    if cache=="legacy": assert "Legacy resume signature" in rows[0]["error"]
    elif cache=="old-version": assert "different input/config/software" in rows[0]["error"]
    elif cache!="success": assert "not successful" in rows[0]["error"]
    assert target.read_bytes()==before and target.stat().st_mtime_ns==mtime


@pytest.mark.parametrize("reverse",[False,True])
def test_gpu_auto_batch_mixed_policies(monkeypatch,reverse):
    from types import SimpleNamespace
    from hpgsbs.batch import scheduler
    import hpgsbs.backends.torch_cuda as gpu
    class FakeOOM(Exception): pass
    fake_torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:(10**9,10**9),
                               OutOfMemoryError=FakeOOM,empty_cache=lambda:None))
    plans=[]; batches=[]
    def plan(T,P,free,fraction,itemsize):
        plans.append(fraction)
        return 2 if fraction==.9 else 1
    def outputs(xs,cs,**kwargs):
        policies={(c.mode,c.memory_fraction) for c in cs}
        assert len(policies)==1
        batches.append((next(iter(policies)),len(xs)))
        yield from [object() for _ in xs]
    monkeypatch.setattr(gpu,"require_cuda",lambda:fake_torch)
    monkeypatch.setattr(gpu,"iter_gpu",outputs)
    monkeypatch.setattr(scheduler,"automatic_batch_size",plan)
    policies=[("throughput",.9),("latency",.1),("throughput",.1)]*2
    if reverse: policies.reverse()
    xs=[np.random.default_rng(i).normal(size=(18,4)) for i in range(len(policies))]
    cs=[GSBSConfig(5,mode=m,memory_fraction=f) for m,f in policies]
    rows=list(scheduler.iter_many(xs,cs,"torch-cuda"))
    assert sorted(i for i,_ in rows)==list(range(6))
    assert sorted(plans)==[.1,.9]  # latency never requests a throughput plan
    assert sorted(batches)==sorted([(("throughput",.9),2),(("latency",.1),1),
                                  (("latency",.1),1),(("throughput",.1),1),(("throughput",.1),1)])


def test_actual_interruption_resume(tmp_path):
    import subprocess,sys,time
    tasks=[]
    for i,T in enumerate((16,120,18)):
        path=tmp_path/f"x{i}.npy"; np.save(path,np.random.default_rng(i).normal(size=(T,8)))
        tasks.append(dict(id=f"rec{i}",path=str(path),config=dict(kmax=5)))
    manifest=tmp_path/"tasks.json"; manifest.write_text(json.dumps(tasks)); out=tmp_path/"out"
    proc=subprocess.Popen([sys.executable,"-m","hpgsbs","fit-batch",str(manifest),"--backend","reference","--output",str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+20
        while not (out/"rec0.npz").exists() and time.monotonic()<deadline:
            if proc.poll() is not None: break
            time.sleep(.01)
        assert (out/"rec0.npz").exists()
        first_time=(out/"rec0.npz").stat().st_mtime_ns
        proc.terminate(); proc.communicate(timeout=10)
        rows=run_manifest(tasks,out,backend="reference")
        assert rows[0]["status"]=="skipped"
        assert all(r["status"] in ("success","skipped") for r in rows)
        assert (out/"rec0.npz").stat().st_mtime_ns==first_time
    finally:
        if proc.poll() is None: proc.kill(); proc.communicate()
