from dataclasses import replace
import numpy as np
import pytest
from statesegmentation import GSBS
from hpgsbs import fit,GSBSConfig
from hpgsbs.validation import compare
from hpgsbs.core.interval_scores import IntervalScores
from hpgsbs.core.candidate_search import CandidateSearch
from hpgsbs.core.streaming_candidates import candidate_count,stream_q,stream_window,stream_prefix,candidates
from hpgsbs.core.budgets import ResourceBudgetExceeded,SearchCancelled


def test_exact_minus_one_after_near_tie_returns_none():
    x=np.tile([-1.,0.,1.],(6,1))
    q=IntervalScores(x,z=-GSBS._zscore(x)); q.Q[:]=0
    search=CandidateSearch(x,q)
    assert search.best(np.zeros(6,bool),double=True)==(None,-1.)
    assert search.last_diagnostics['fallback_evaluated']==10


@pytest.mark.parametrize('cause',['work','workspace','candidates','cancel'])
def test_refusal_never_produces_fit(cause):
    x=np.tile([-1.,0.,1.],(30,1))
    c=GSBSConfig(8,q_strategy='streaming',td_backend='feature-moments')
    kwargs={}
    if cause=='work': c=replace(c,max_fallback_elements=1)
    if cause=='workspace': c=replace(c,max_workspace_bytes=1)
    if cause=='candidates': c=replace(c,max_candidates=1)
    if cause=='cancel': kwargs['cancel']=lambda:True
    with pytest.raises((ResourceBudgetExceeded,SearchCancelled)):
        fit(x,c,**kwargs)


@pytest.mark.parametrize('level',['full-search-trace','finetune-move-trace','numeric-stress'])
def test_missing_evidence_is_failure(level):
    x=np.random.default_rng(91).normal(size=(18,5))
    r=fit(x,GSBSConfig(7))
    report=compare(r,r,level=level)
    assert not report['passed'] and not report['trace_compared']
    assert compare(r,r,level='output-only')['passed']


@pytest.mark.parametrize('fine',[0,1,2,-1])
@pytest.mark.parametrize('order',[False,True])
def test_strict_every_move(fine,order):
    x=np.random.default_rng(91).normal(size=(23,5))
    c=GSBSConfig(12,trace=True,finetune=fine,finetune_order=order,blocksize=6)
    r=fit(x,c,'reference'); s=fit(x,c)
    report=compare(r,s,level='finetune-move-trace')
    assert report['passed'],report
    if fine:
        del s.trace[-1]['finetune_moves']
        assert not compare(r,s,level='finetune-move-trace')['passed']


@pytest.mark.parametrize('double',[False,True])
@pytest.mark.parametrize('opt',[None,[],[(2,4),(2,4)],[(2,4),(5,7)],[(0,0)]])
def test_independent_domain_summary(double,opt):
    rng=np.random.default_rng(83)
    edges=np.array([1,4,5,11]); allowed=np.ones(10,bool)
    so=np.empty((0,0),np.int64) if opt is None else np.array(opt,dtype=np.int64).reshape(-1,2)
    for kind in ['random','tie','nan','inf','below','minus-one']:
        q=rng.normal(size=(13,13))
        if kind=='tie': q[:]=0
        if kind=='nan': q[2,3]=np.nan; q[7,8]=np.nan
        if kind=='inf': q[2,3]=np.inf
        if kind=='below': q[:]=-10
        if kind=='minus-one': q[:]=-1/((len(edges)-1)+(2 if double else 1)); q*=10/12
        dense=np.full((10,10) if double else (10,),-1.)
        legal=0
        base=sum(q[a,b] for a,b in zip(edges[:-1],edges[1:]))
        for a,b in zip(edges[:-1],edges[1:]):
            for i in range(a+1,b):
                if not allowed[i-1]:continue
                if double and opt is not None and np.sum(so==[i,0])<1:continue
                for j in (range(i+1,b) if double else [i]):
                    if double and opt is not None and np.sum(so==[i,j])<2:continue
                    value=(base-q[a,b]+q[a,i]+q[i,j]+q[j,b])*12/10 if double else (base-q[a,b]+q[a,i]+q[i,b])*12/10
                    dense[(i-1,j-1) if double else i-1]=value; legal+=1
        args=(edges,1,11,allowed,so,double,12)
        s=stream_q(q,*args)
        assert s[0]==dense.argmax()
        np.testing.assert_equal(s[1],dense.max())
        np.testing.assert_equal(s[2],np.nan if np.isnan(dense).any() else np.sort(dense.ravel())[-2])
        assert s[4]==(not np.all(dense==-1))
        assert s[5]==legal==candidate_count(edges,1,allowed,so,double)
        class Q:
            def __call__(self,a,b):return q[a,b]
        for chunk in (1,7,64,256):
            ss=stream_prefix(Q(),*args,chunk)
            np.testing.assert_equal(s,ss)


def test_window_nan_summary_and_domain():
    q=np.zeros((10,10)); q[1,3]=np.nan
    e=np.array([0,1,6,9])
    s=stream_window(q,e,2,8,9)
    allowed=np.array([False,False,True,True,True,True,True,True,False])
    t=stream_q(q,e,0,9,allowed,np.empty((0,0),np.int64),False,9)
    np.testing.assert_equal(s,t)


def test_fit_input_snapshot_protects_cache():
    x=np.random.default_rng(4).normal(size=(31,6)); original=x.copy()
    c=GSBSConfig(10,q_strategy='streaming',trace=True)
    def progress(_): x[:]=100
    r=fit(x,c,progress=progress)
    assert compare(fit(original,c),r,level='finetune-move-trace')['passed']


def test_transactional_incremental_full_fit():
    from hpgsbs.core.incremental_tdistance import IncrementalTDistance
    from hpgsbs.backends.cpu_fast import run_search
    from threadpoolctl import threadpool_limits
    x=np.random.default_rng(31).normal(size=(51,7))
    for fine in (0,1,2,-1):
        c=GSBSConfig(25,finetune=fine,trace=True,blocksize=9)
        td=IncrementalTDistance(x,5)
        td.rebuild_interval=2
        c=replace(c,dmin=5)
        with threadpool_limits(1): r=run_search(x,c,IntervalScores(x),td)
        assert compare(fit(x,c,'reference'),r,level='finetune-move-trace')['passed']
        snapshot=td.moments,td.regions,td.totals,td.updates
        td(np.zeros(len(x),bool))
        assert all(a is b for a,b in zip(snapshot[:3],(td.moments,td.regions,td.totals)))
