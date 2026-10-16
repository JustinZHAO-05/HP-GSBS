import numpy as np
import pytest
from hpgsbs.core.streaming_candidates import stream_q

torch=pytest.importorskip("torch")
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")]


@pytest.mark.parametrize("double",[False,True])
@pytest.mark.parametrize("opt",[None,[],[(2,5),(4,7)],[(2,5),(2,7)],[(0,0)]])
def test_device_reducer_gates_sentinel_and_nan(double,opt):
    pytest.importorskip("triton")
    from hpgsbs.backends.resident_candidates import candidate_summary
    for kind in ("random","tie","nan","inf","below-sentinel"):
        q=np.random.default_rng(73).normal(size=(34,34))
        if kind=="tie": q[:]=0
        if kind=="nan": q[2,5]=np.nan
        if kind=="inf": q[2,5]=np.inf
        if kind=="below-sentinel": q[:]=-100
        e=np.array([1,12,33]); allowed=np.ones(32,dtype=bool)
        so=np.empty((0,0),np.int64) if opt is None else np.asarray(opt,dtype=np.int64).reshape(-1,2)
        cpu=stream_q(q,e,1,33,allowed,so,double,33)
        gpu=candidate_summary(torch.as_tensor(q,device="cuda"),q,e,1,33,allowed,so,double,33)
        assert cpu[0]==gpu[0]
        np.testing.assert_allclose(cpu[1:3],gpu[1:3],atol=1e-12,rtol=1e-12,equal_nan=True)
        assert cpu[3:]==gpu[3:]


def test_multiple_grid_stride_tiles():
    pytest.importorskip("triton")
    from hpgsbs.backends.resident_candidates import candidate_summary
    q=np.random.default_rng(7).normal(size=(501,501))
    e=np.array([0,500]); allowed=np.ones(500,dtype=bool); opt=np.empty((0,0),np.int64)
    cpu=stream_q(q,e,0,500,allowed,opt,True,500)
    gpu=candidate_summary(torch.as_tensor(q,device="cuda"),q,e,np.int64(0),np.int64(500),allowed,opt,True,np.int64(500))
    assert cpu[0]==gpu[0]
    np.testing.assert_allclose(cpu[1:3],gpu[1:3],atol=1e-12,rtol=1e-12)
    assert cpu[3:]==gpu[3:]


@pytest.mark.parametrize('kind',['multiple-nan','infinity','near-tie','exact-tie'])
def test_cross_program_complete_summary(kind):
    pytest.importorskip('triton')
    from hpgsbs.backends.resident_candidates import candidate_summary
    q=np.zeros((701,701));e=np.array([0,700]);allowed=np.ones(700,bool);opt=np.empty((0,0),np.int64)
    if kind=='multiple-nan':q[2,30]=np.nan;q[500,600]=np.nan
    if kind=='infinity':q[20,300]=np.inf;q[500,600]=-np.inf
    if kind=='near-tie':q[2,300]=1;q[400,600]=1+1e-12
    cpu=stream_q(q,e,0,700,allowed,opt,True,700)
    gpu=candidate_summary(torch.tensor(q,device='cuda'),q,e,0,700,allowed,opt,True,700)
    np.testing.assert_equal(cpu,gpu)


def test_host_device_mismatch_and_invalid_metadata():
    pytest.importorskip('triton')
    from hpgsbs.backends.resident_candidates import candidate_summary
    q=np.zeros((11,11));device=torch.tensor(q,device='cuda');q[1,2]=1
    args=(np.array([0,10]),0,10,np.ones(10,bool),np.empty((0,0),np.int64),True,10)
    with pytest.raises(ValueError,match='Host/device'):candidate_summary(device,q,*args)
    with pytest.raises(ValueError,match='metadata'):
        candidate_summary(device,q,np.array([0,11]),0,11,np.ones(11,bool),args[4],True,11)


@pytest.mark.parametrize("backend",["torch-cuda","triton-cuda"])
def test_resident_complete_fit_trace(backend):
    pytest.importorskip("triton")
    from hpgsbs import fit,GSBSConfig
    from hpgsbs.validation import compare
    from dataclasses import replace
    x=np.random.default_rng(81).normal(size=(270,8))
    c=GSBSConfig(10,trace=True)
    ref=fit(x,replace(c,candidate_backend="materialized"))
    r=fit(x,replace(c,candidate_backend="triton-resident"),backend)
    assert r.metadata["candidate_gpu_calls"]>0
    report=compare(ref,r)
    assert report["passed"],report


@pytest.mark.parametrize("index",range(4))
def test_resident_full_eeg_trace(index):
    pytest.importorskip("triton")
    import os
    from pathlib import Path
    from hpgsbs import fit,GSBSConfig,GSBSResult
    from hpgsbs.validation import compare
    path=Path(os.environ.get("HPGSBS_FIXTURES","results/fixtures"))/f"{index}_full_cpu.npz"
    if not path.exists(): pytest.skip("Private full EEG fixtures unavailable")
    ref=GSBSResult.load(path)
    c=GSBSConfig(**ref.metadata["algorithm"],trace=True,candidate_backend="triton-resident")
    r=fit(ref.x,c,"triton-cuda")
    assert r.metadata["candidate_gpu_calls"]>0
    report=compare(ref,r)
    assert report["passed"],report
