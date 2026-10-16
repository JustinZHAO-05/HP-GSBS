import numpy as np
import pytest
from hpgsbs import fit, GSBSConfig
from hpgsbs.core.candidate_search import CandidateSearch, full_scores
from hpgsbs.core.interval_scores import IntervalScores
from hpgsbs.core.streaming_candidates import stream_q, reduce_values, scored_chunks
from hpgsbs.validation import compare


@pytest.mark.parametrize("values", [[],[-1,-1],[-2,-3],[np.nan,4,np.nan],
                                    [2,2],[np.inf,np.inf],[-np.inf,-2],[-1,0]])
@pytest.mark.parametrize("chunk", [1,7,64])
def test_reducer_numpy_contract(values,chunk):
    dense = np.full(25,-1.0)
    flats = np.arange(1,len(values)+1,dtype=np.int64)
    dense[flats] = values
    summary = (0,-1.0,-1.0,False,False,0)
    for k in range(0,len(values),chunk):
        summary = reduce_values(flats[k:k+chunk],np.asarray(values[k:k+chunk]),summary)
    assert summary[0] == dense.argmax()
    assert summary[4] == (not np.all(dense == -1))
    np.testing.assert_equal(summary[1],dense.max())


@pytest.mark.parametrize("double",[False,True])
@pytest.mark.parametrize("opt",[None,[],[(2,5),(2,7)],[(2,5),(4,7)],[(0,0)]])
def test_stream_q_arbitrary_scores(double,opt):
    rng = np.random.default_rng(83)
    for seed in range(10):
        q = rng.normal(size=(18,18))*3
        if seed == 0: q[2,5] = np.nan
        if seed == 1: q[:] = 0
        if seed == 2: q[2,5] = np.inf
        edges = np.array([1,8,17])
        allowed = rng.random(16)>.2
        so = np.empty((0,0),np.int64) if opt is None else np.asarray(opt,dtype=np.int64).reshape(-1,2)
        dense = full_scores(q,edges,1,17,allowed,so,double,17)
        s = stream_q(q,edges,1,17,allowed,so,double,17)
        assert s[0] == dense.argmax()
        np.testing.assert_equal(s[1],dense.ravel()[s[0]])
        assert s[4] == (not np.all(dense==-1))


@pytest.mark.parametrize("chunk",[1,7,64])
@pytest.mark.parametrize("double",[False,True])
def test_prefix_chunks_scores(chunk,double):
    x = np.random.default_rng(62).normal(size=(31,9))
    q = IntervalScores(x,"lazy")
    full = IntervalScores(x)
    e = np.array([3,14,29]); allowed = np.ones(26,dtype=bool); so=np.empty((0,0),np.int64)
    dense = full_scores(full.Q,e,3,29,allowed,so,double,31).ravel()
    for rows,flats,values in scored_chunks(q,e,3,29,allowed,so,double,31,chunk):
        assert len(rows)<=chunk
        np.testing.assert_array_equal(values,dense[flats])


@pytest.mark.parametrize("fine",[1,2,-1])
@pytest.mark.parametrize("block",[0,7,50])
@pytest.mark.parametrize("strategy",["full","streaming"])
def test_complete_trace_and_every_refinement(fine,block,strategy):
    x = np.random.default_rng(133).normal(size=(26,7))
    c = GSBSConfig(10,trace=True,finetune=fine,blocksize=block,q_strategy=strategy,candidate_chunk_size=7)
    report = compare(fit(x,c,"reference"),fit(x,c))
    assert report["passed"], report


def test_streaming_does_not_call_materialized(monkeypatch):
    x=np.random.default_rng(63).normal(size=(50,8))
    search=CandidateSearch(x,IntervalScores(x))
    def forbidden(*args): raise AssertionError("materialized path called")
    monkeypatch.setattr(search,"_scores",forbidden)
    assert search.best(np.zeros(50,bool),double=True)[0] is not None


def test_cache_invalidates_internal_cuts_and_grouping():
    from statesegmentation import GSBS
    x=np.random.default_rng(47).normal(size=(26,8))
    search=CandidateSearch(x,IntervalScores(x))
    for cuts,block in [([6,12,20],0),([6,12,20],0),([6,13,20],0),
                       ([6,13,20],7),([6,20],7),([3,6,20],7)]:
        d=np.zeros(26,bool); d[cuts]=True
        expected=GSBS._wdists_blocks(d,GSBS._states(d),x,GSBS._zscore(x),True,block)
        one,two=search.blocks(d,True,block)
        assert one[0]==expected[0].argmax()
        assert two[0]==(None if expected[1] is None else np.unravel_index(expected[1].argmax(),expected[1].shape))
        assert len(search._local_cache)<=len(cuts)+1
    assert search.cache_hits>0


def test_streaming_prefix_respects_supplied_z():
    from hpgsbs.core.interval_scores import zscore
    x=np.random.default_rng(47).normal(size=(26,8)); z=zscore(x)*np.linspace(.5,2,26)[:,None]
    q=IntervalScores(x,"streaming",z=z)
    for a,b in [(0,26),(1,12),(17,18)]:
        expected=np.sum(zscore(x[a:b].mean(0,keepdims=True))*z[a:b])/(26*7)
        np.testing.assert_allclose(q(a,b),expected,atol=2e-14,rtol=2e-14)
