from fractions import Fraction
import numpy as np
import pytest
from hpgsbs import fit,GSBSConfig
from hpgsbs.core.incremental_tdistance import IncrementalTDistance
from hpgsbs.validation import compare


def test_binary64_u_exact_pair_oracle_with_short_band():
    td=IncrementalTDistance(np.random.default_rng(66).normal(size=(11,3)),dmin=3)
    u=[[Fraction.from_float(v) for v in row] for row in td.u]
    for cuts in ([4],[2,4,7],[2,7],[2,9],[1,2,9]):
        b=np.zeros(11,bool); b[cuts]=True
        labels=np.cumsum(b)
        expected=[[Fraction(0) for _ in range(3)] for _ in range(2)]
        for i in range(11):
            for j in range(i+3,11):
                kind=labels[j]-labels[i]
                if kind not in (0,1): continue
                r=sum(a*c for a,c in zip(u[i],u[j]))/2
                expected[kind][0]+=1; expected[kind][1]+=r; expected[kind][2]+=r*r
        for actual,oracle in zip(td.stats(b),expected):
            np.testing.assert_allclose(actual,[float(v) for v in oracle],atol=1e-13,rtol=1e-13)


@pytest.mark.parametrize("kind",["repeated","offset","cancellation"])
def test_default_path_preserves_original_pathological_semantics(kind):
    rng=np.random.default_rng(12)
    if kind=="repeated": x=np.tile([1.,2.,3.,4.],(16,1))
    elif kind=="offset": x=1e10+rng.normal(size=(16,5))*1e-5
    else: x=np.tile([[1.,2.,3.],[-1.,-2.,-3.]],(8,1))
    c=GSBSConfig(7,trace=True,dmin=5)
    report=compare(fit(x,c,"reference"),fit(x,c))
    assert report["passed"],report
