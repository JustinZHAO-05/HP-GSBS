import numpy as np
import pytest
from statesegmentation import GSBS
from hpgsbs.core.interval_scores import IntervalScores
from hpgsbs.core.candidate_search import CandidateSearch


@pytest.mark.parametrize("stateopt",[None,[(2,5),(2,7)],[(2,5),(4,7)],[]])
def test_double_mask_actual_upstream_semantics(stateopt):
    x=np.random.default_rng(6).normal(size=(13,8)); deltas=np.zeros(13,bool); deltas[8]=True
    so=None if stateopt is None else np.asarray(stateopt).reshape(-1,2)
    r=GSBS._wdists_state(deltas,GSBS._states(deltas),x,GSBS._zscore(x),so)
    pos,score=CandidateSearch(x,IntervalScores(x)).best(deltas,stateopt=so,double=True)
    if r is None: assert pos is None
    else:
        assert pos==np.unravel_index(r.argmax(),r.shape)
        np.testing.assert_allclose(score,r.max(),atol=1e-12,rtol=1e-12)


def test_no_candidates():
    x=np.random.default_rng(4).normal(size=(8,3)); d=np.ones(8,bool); d[0]=False
    search=CandidateSearch(x,IntervalScores(x))
    assert search.best(d)==(0,-1)
    assert search.best(d,double=True)==(None,-1)


def test_vectorized_strengths():
    from hpgsbs.core.state_utils import boundary_strengths
    for p in (2,3,64):
        x=np.random.default_rng(3).normal(size=(50,p))
        g=GSBS(kmax=20,x=x); g._argmax=8
        g.all_bounds[8,[1,2,3,8,15,35,49]]=np.arange(1,8)
        np.testing.assert_allclose(boundary_strengths(x,g.bounds),g.strengths,atol=1e-14,rtol=1e-14)
