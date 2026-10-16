import numpy as np
import pytest
import os
from hpgsbs import fit,GSBSConfig
from hpgsbs.validation import compare

torch=pytest.importorskip("torch")
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")]


@pytest.mark.parametrize("backend",["torch-cuda","triton-cuda"])
@pytest.mark.parametrize("seed",[1,7,41])
def test_single(backend,seed):
    if backend=="triton-cuda": pytest.importorskip("triton")
    x=np.random.default_rng(seed).normal(size=(40,8))
    c=GSBSConfig(12,blocksize=7,trace=True)
    report=compare(fit(x,c,"reference"),fit(x,c,backend))
    assert report["passed"],report


@pytest.mark.parametrize("backend",["torch-cuda","triton-cuda"])
def test_batch(backend):
    from hpgsbs.backends.torch_cuda import iter_gpu
    xs=[np.random.default_rng(i).normal(size=(32,8)) for i in range(3)]
    cs=[GSBSConfig(kmax=8+i,dmin=2,trace=True) for i in range(3)]
    for x,c,out in zip(xs,cs,iter_gpu(xs,cs,backend=backend)):
        assert compare(fit(x,c,"reference"),out)["passed"]


def test_triton_candidates():
    from hpgsbs.backends.triton_cuda import candidate_scores
    q=torch.randn((33,33),device="cuda",dtype=torch.float64)
    for double in (False,True):
        scores=candidate_scores(q,3,27,double).cpu().numpy()
        qh=q.cpu().numpy()
        if double:
            ref=np.array([qh[3,i]+qh[i,j]+qh[j,27]-qh[3,27] if j>i else -np.inf for i in range(4,27) for j in range(4,27)])
        else: ref=np.array([qh[3,i]+qh[i,27]-qh[3,27] for i in range(4,27)])
        np.testing.assert_allclose(scores,ref,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize("backend",["torch-cuda","triton-cuda"])
def test_interval_precompute(backend):
    from hpgsbs.backends.torch_cuda import precompute
    from hpgsbs.core.interval_scores import IntervalScores
    x=np.random.default_rng(51).normal(size=(64,32))
    pairs,_=precompute([x],[GSBSConfig(10)],backend=backend)
    q=IntervalScores(x).Q
    np.testing.assert_allclose(pairs[0][0].Q,q,atol=2e-12,rtol=2e-12)


@pytest.mark.parametrize("backend",["torch-cuda","triton-cuda"])
@pytest.mark.parametrize("fine",[0,2,-1])
def test_y_and_finetune(backend,fine):
    rng=np.random.default_rng(177)
    x=rng.normal(size=(30,5)); y=rng.normal(size=(30,9))
    c=GSBSConfig(9,dmin=3,finetune=fine,finetune_order=False,trace=True,blocksize=0)
    assert compare(fit(x,c,"reference",y=y),fit(x,c,backend,y=y))["passed"]


@pytest.mark.parametrize("backend",["cpu-fast","torch-cuda","triton-cuda"])
@pytest.mark.parametrize("index",range(4))
def test_full_eeg_regression(backend,index):
    from pathlib import Path
    from hpgsbs import GSBSResult
    p=Path(os.environ.get("HPGSBS_FIXTURES","results/fixtures"))/f"{index}_full_cpu.npz"
    if not p.exists(): pytest.skip("Private full EEG fixtures unavailable")
    ref=GSBSResult.load(p)
    c=GSBSConfig(**ref.metadata["algorithm"],trace=True)
    report=compare(ref,fit(ref.x,c,backend))
    assert report["passed"],report


def test_oom_halving(monkeypatch):
    from hpgsbs.batch.scheduler import iter_many
    import hpgsbs.backends.torch_cuda as gpu
    attempted=[]
    def fake(xs,cs,**kwargs):
        attempted.append(len(xs))
        if len(xs)>1: raise torch.cuda.OutOfMemoryError("injected test OOM")
        yield fit(xs[0],cs[0])
    monkeypatch.setattr(gpu,"iter_gpu",fake)
    xs=[np.random.default_rng(i).normal(size=(18,4)) for i in range(3)]
    results=list(iter_many(xs,[GSBSConfig(5)]*3,"torch-cuda",batch_size=3))
    assert attempted==[3,1,1,1]
    assert [i for i,r in results]==[0,1,2]


def test_variable_manifest_resume(tmp_path):
    from hpgsbs.batch.scheduler import run_manifest
    tasks=[]
    for i,(T,P) in enumerate(((20,5),(20,5),(25,8))):
        p=tmp_path/f"x{i}.npy"; np.save(p,np.random.default_rng(i).normal(size=(T,P)))
        tasks.append(dict(id=str(i),path=str(p),config=dict(kmax=6+i,mode="throughput")))
    out=tmp_path/"out"
    rows=run_manifest(tasks,out,backend="triton-cuda",batch_size=2)
    assert all(r["status"]=="success" for r in rows),rows
    assert all(r["status"]=="skipped" for r in run_manifest(tasks,out,backend="triton-cuda",batch_size=2))


@pytest.mark.parametrize("backend",["torch-cuda","triton-cuda"])
@pytest.mark.parametrize("reverse",[False,True])
def test_auto_batch_mixed_resource_policies(backend,reverse):
    from hpgsbs import fit_many
    xs=[np.random.default_rng(80+i).normal(size=(24,5)) for i in range(6)]
    policies=[("throughput",.9),("latency",.1),("throughput",.1)]*2
    cs=[GSBSConfig(7,mode=m,memory_fraction=f,trace=True) for m,f in policies]
    if reverse: xs.reverse(); cs.reverse()
    results=fit_many(xs,cs,backend,batch_size="auto")
    for x,c,result in zip(xs,cs,results):
        assert compare(fit(x,c,"reference"),result)["passed"]
        assert result.metadata["batch_size"]==(1 if c.mode=="latency" else 2)
