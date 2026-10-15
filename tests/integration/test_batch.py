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
    mtime=(out/"ok.npz").stat().st_mtime_ns
    b=run_manifest(tasks,out)
    assert b[0]["status"]=="skipped"
    assert (out/"ok.npz").stat().st_mtime_ns==mtime
    tasks[0]["config"]["kmax"]=6
    assert run_manifest(tasks,out)[0]["status"]=="failed"


def test_process_batch():
    xs=[np.random.default_rng(i).normal(size=(18+i,5)) for i in range(3)]
    c=GSBSConfig(7,n_jobs=2)
    many=fit_many(xs,c,"cpu-fast")
    for x,r in zip(xs,many): np.testing.assert_array_equal(r.all_bounds,fit(x,c).all_bounds)


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
