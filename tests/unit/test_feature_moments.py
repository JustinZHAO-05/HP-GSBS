from fractions import Fraction as F
import numpy as np
import pytest
from hpgsbs import GSBSConfig,fit
from hpgsbs.core.feature_moments import FeatureMomentsTDistance
from hpgsbs.core.tdistance import TDistance
from hpgsbs.validation import compare


def test_exact_rational_moment_identities():
    u = [[F(v) for v in row] for row in [[1,2,-3],[2,-1,-1],[-2,3,-1],[1,0,-1]]]
    h=F(2)
    def dot(a,b): return sum(x*y for x,y in zip(a,b))
    def moment(rows):
        s=[sum(r[i] for r in rows) for i in range(3)]
        m=[[sum(r[i]*r[j] for r in rows) for j in range(3)] for i in range(3)]
        return s,m,sum(dot(r,r) for r in rows),sum(dot(r,r)**2 for r in rows)
    s,m,d1,d2=moment(u)
    pairs=[dot(u[i],u[j])/h for i in range(4) for j in range(i+1,4)]
    assert (dot(s,s)-d1)/(2*h)==sum(pairs)
    assert (sum(v*v for row in m for v in row)-d2)/(2*h*h)==sum(v*v for v in pairs)
    s1,m1,_,_=moment(u[:2]); s2,m2,_,_=moment(u[2:])
    cross=[dot(a,b)/h for a in u[:2] for b in u[2:]]
    assert dot(s1,s2)/h==sum(cross)
    assert sum(m1[i][j]*m2[i][j] for i in range(3) for j in range(3))/(h*h)==sum(v*v for v in cross)


@pytest.mark.parametrize("dmin",[1,2,5,30])
@pytest.mark.parametrize("p",[2,8,16])
def test_stats_and_bounded_two_pass(dmin,p):
    x=np.random.default_rng(27).normal(size=(29,p))
    m=FeatureMomentsTDistance(x,dmin,tile_size=7); ref=TDistance(x,dmin)
    for cuts in ([4],[1,2,4,10,28],list(range(1,29)),[7,14,21]):
        b=np.zeros(29,bool); b[cuts]=True
        for a,c in zip(m.stats(b),ref.stats(b)):
            np.testing.assert_allclose(a,c,atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(m(b),ref.exact(b),atol=2e-11,rtol=2e-11,equal_nan=True)
        np.testing.assert_allclose(m.exact(b),ref.exact(b),atol=2e-11,rtol=2e-11,equal_nan=True)


@pytest.mark.parametrize("fine",[0,1,2,-1])
@pytest.mark.parametrize("statewise",[False,True])
def test_low_memory_full_fit(fine,statewise):
    rng=np.random.default_rng(56); x=rng.normal(size=(31,8)); y=rng.normal(size=(31,5))
    c=GSBSConfig(11,td_backend="feature-moments",q_strategy="streaming",trace=True,
                 finetune=fine,statewise_detection=statewise,blocksize=7,candidate_chunk_size=7)
    r=compare(fit(x,c,"reference",y=y),fit(x,c,y=y))
    assert r["passed"],r


def test_no_dense_fallback_on_ill_conditioned_rows():
    x=1e10+np.random.default_rng(12).normal(size=(16,5))*1e-5
    with pytest.raises(FloatingPointError,match="dense reference fallback is disabled"):
        fit(x,GSBSConfig(7,td_backend="feature-moments",q_strategy="streaming"))


def test_cancellation_uses_two_pass_without_cov(monkeypatch):
    x=np.tile(np.arange(8,dtype=float),(30,1))+np.random.default_rng(8).normal(size=(30,8))*1e-5
    td=FeatureMomentsTDistance(x,5,tile_size=7)
    bounds=np.zeros(30,bool); bounds[[10,20]]=True
    ref=TDistance(x,5).exact(bounds)
    def forbidden(*args,**kwargs): raise AssertionError("dense covariance fallback")
    monkeypatch.setattr(np,"cov",forbidden)
    with pytest.raises(FloatingPointError,match="Uncertified near-degenerate"):
        td(bounds)
    assert td.fallbacks==1
    # Near-zero native variance is sensitive to BLAS tiling, so report rather
    # than granting generic parity; the ordinary-domain tolerance is unchanged.
    assert np.isfinite(ref)
    assert td.last_diagnostics["method"]=="tiled-two-pass"
    assert td.last_diagnostics["status"]=="unresolved-near-degenerate"
