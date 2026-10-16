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
    gpu=candidate_summary(torch.as_tensor(q,device="cuda"),q,e,0,500,allowed,opt,True,500)
    assert cpu[0]==gpu[0]
    np.testing.assert_allclose(cpu[1:3],gpu[1:3],atol=1e-12,rtol=1e-12)
    assert cpu[3:]==gpu[3:]
