from dataclasses import replace
import numpy as np
import pytest
from hpgsbs import fit, GSBSConfig


def assert_parity(ref,fast):
    np.testing.assert_array_equal(fast.all_bounds,ref.all_bounds)
    np.testing.assert_allclose(fast.tdists,ref.tdists,atol=1e-8,rtol=1e-8)
    assert fast.nstates==ref.nstates
    np.testing.assert_allclose(fast.state_patterns,ref.state_patterns,atol=1e-12)
    np.testing.assert_allclose(fast.strengths,ref.strengths,atol=1e-12)
    assert len(fast.trace)==len(ref.trace)
    for r,f in zip(ref.trace,fast.trace):
        for key in ("k","before","single","double","increment","pre_finetune","after"):
            assert r[key]==f[key], (key,r,f)
        for key in ("single_score","double_score","single_t","double_t"):
            if r[key] is not None: np.testing.assert_allclose(f[key],r[key],atol=1e-8,rtol=1e-8)


@pytest.mark.parametrize("finetune",[0,1,2,-1])
@pytest.mark.parametrize("statewise",[True,False])
@pytest.mark.parametrize("order",[True,False])
@pytest.mark.parametrize("block",[0,7,50])
def test_parameters(finetune,statewise,order,block):
    x=np.random.default_rng(91).normal(size=(24,8))
    c=GSBSConfig(10,finetune=finetune,statewise_detection=statewise,finetune_order=order,blocksize=block,trace=True)
    assert_parity(fit(x,c,"reference"),fit(x,c))


@pytest.mark.parametrize("seed",range(8))
@pytest.mark.parametrize("p",[2,64])
def test_y_dmin_and_lazy(seed,p):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(22,p)); y=rng.normal(size=(22,7))
    c=GSBSConfig(8,dmin=3,blocksize=5,trace=True,q_strategy="lazy")
    assert_parity(fit(x,c,"reference",y=y),fit(x,c,y=y))
