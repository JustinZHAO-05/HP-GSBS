import numpy as np
import pytest
from statesegmentation import GSBS
from hpgsbs.core.interval_scores import IntervalScores, interval_reference, zscore, validate_input
from hpgsbs.core.tdistance import TDistance


@pytest.mark.parametrize("p", [2, 8, 64])
@pytest.mark.parametrize("strategy", ["full", "lazy"])
def test_all_intervals(p, strategy):
    x = np.random.default_rng(42).normal(size=(20,p))
    q = IntervalScores(x, strategy)
    np.testing.assert_allclose(zscore(x), GSBS._zscore(x), atol=0, rtol=0)
    for a in range(len(x)):
        for b in range(a+1, len(x)+1):
            np.testing.assert_allclose(q(a,b), interval_reference(x,a,b), atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("dmin", [1,2,5])
def test_tdistance(dmin):
    x = np.random.default_rng(11).normal(size=(32,8))
    ind = np.triu(np.ones((32,32),bool), dmin)
    t = np.cov(GSBS._zscore(x))[ind]
    fast = TDistance(x,dmin)
    for edges in ([3], [1,2,5], [8,16,24], list(range(1,32))):
        delta = np.zeros(32,bool); delta[edges] = True
        np.testing.assert_allclose(fast(delta), GSBS._tdist(delta,t,ind), atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize("x", [np.ones((10,2)),np.ones((10,1)),np.full((10,3),np.nan),np.r_[np.zeros((1,3)),np.ones((9,3))]])
def test_undefined_inputs(x):
    with pytest.raises(ValueError): validate_input(x)
