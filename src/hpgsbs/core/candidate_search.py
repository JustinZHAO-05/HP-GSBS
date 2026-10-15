import numpy as np
from numba import njit
from statesegmentation import GSBS


@njit(cache=True)
def full_scores(q, edges, start, end, allowed, opt, double, T):
    n = end-start
    out = np.full((n,n) if double else (n,1), -1.0)
    base = 0.0
    for s in range(len(edges)-1):
        base += q[edges[s],edges[s+1]]
    for s in range(len(edges)-1):
        a,b = edges[s],edges[s+1]
        for i in range(a+1,b):
            if not allowed[i-start]: continue
            if double:
                first = 1
                if opt.shape[0] >= 0 and opt.shape[1] == 2:
                    first = 0
                    for row in range(opt.shape[0]):
                        first += (opt[row,0] == i) + (opt[row,1] == 0)
                if first == 0: continue
                for j in range(i+1,b):
                    matches = 2
                    if opt.shape[1] == 2:
                        matches = 0
                        for row in range(opt.shape[0]):
                            matches += (opt[row,0] == i) + (opt[row,1] == j)
                    if matches >= 2:
                        out[i-start,j-start] = (base-q[a,b]+q[a,i]+q[i,j]+q[j,b])*T/n
            else:
                out[i-start,0] = (base-q[a,b]+q[a,i]+q[i,b])*T/n
    return out


class CandidateSearch:
    def __init__(self, x, scores, tolerance=1e-11):
        self.x, self.q, self.tolerance = x, scores, tolerance
        self.z = GSBS._zscore(x)
        self.fallbacks = 0

    def _scores(self, edges, start, end, allowed, opt, double):
        if self.q.Q is not None:
            return full_scores(self.q.Q, edges, start, end, allowed, opt, double, len(self.x))
        n = end-start
        out = np.full((n,n) if double else (n,1), -1.0)
        base = sum(self.q(a,b) for a,b in zip(edges[:-1],edges[1:]))
        for a,b in zip(edges[:-1],edges[1:]):
            for i in range(a+1,b):
                if not allowed[i-start]: continue
                if double:
                    if opt.shape[1] == 2 and np.sum(opt == [i,0]) < 1: continue
                    js = np.arange(i+1,b)
                    if opt.shape[1] == 2:
                        js = np.array([j for j in js if np.sum(opt == [i,j]) >= 2], dtype=int)
                    out[i-start,js-start] = (base-self.q(a,b)+self.q(a,i)+self.q(i,js)+self.q(js,b))*len(self.x)/n
                else:
                    out[i-start,0] = (base-self.q(a,b)+self.q(a,i)+self.q(i,b))*len(self.x)/n
        return out

    def best(self, deltas, start=0, end=None, *, boundopt=None, stateopt=None, double=False):
        if end is None: end = len(deltas)
        edges = np.r_[start, np.flatnonzero(deltas)[(np.flatnonzero(deltas)>start)&(np.flatnonzero(deltas)<end)], end]
        allowed = np.ones(end-start, dtype=np.bool_)
        if boundopt is not None:
            allowed[:] = False
            for i in boundopt:
                if start <= i < end: allowed[i-start] = True
        opt = np.empty((0,0),dtype=np.int64) if stateopt is None else np.asarray(stateopt,dtype=np.int64).reshape(-1,2)
        scores = self._scores(edges, start, end, allowed, opt, double)
        if np.all(scores == -1): return (None if double else start), -1.0
        near = np.flatnonzero(scores.ravel() >= np.nanmax(scores)-self.tolerance)
        # Re-evaluate ambiguous candidates in original summation order.
        if len(near)>1 or not np.isfinite(scores).all():
            self.fallbacks += 1
            xt,zt = self.x[start:end],self.z[start:end]
            dt = deltas[start:end]
            states = GSBS._states(dt)
            if not np.isfinite(scores).all():
                if double:
                    st = None if stateopt is None else np.asarray(stateopt)-start
                    scores = GSBS._wdists_state(dt,states,xt,zt,st)
                    if scores is None: return None,-1.0
                else:
                    bo = None if boundopt is None else np.asarray(boundopt)-start
                    scores = GSBS._wdists(dt,states,xt,zt,bo)[:,None]
            else:
                xmeans = np.zeros_like(xt, dtype=float)
                for a,b in zip(edges[:-1],edges[1:]):
                    xmeans[a-start:b-start] = xt[a-start:b-start].mean(0)
                exact = np.full(scores.shape,-np.inf)
                for flat in near:
                    i,j = np.unravel_index(flat,scores.shape)
                    si = np.nonzero(states[i]==states)[0]
                    means = xmeans.copy()
                    a,b = si[0],si[-1]+1
                    means[a:i] = xt[a:i].mean(0)
                    if double:
                        means[i:j] = xt[i:j].mean(0); means[j:b] = xt[j:b].mean(0)
                    else: means[i:b] = xt[i:b].mean(0)
                    exact[i,j] = xt.shape[1]*(GSBS._zscore(means)*zt).mean()/(xt.shape[1]-1)
                scores = exact
        idx = np.unravel_index(scores.argmax(),scores.shape)
        return ((idx[0]+start,idx[1]+start) if double else idx[0]+start), float(scores[idx])

    def blocks(self, deltas, statewise, blocksize):
        edges = np.r_[0,np.flatnonzero(deltas),len(deltas)]
        bo,so = None,None
        if len(edges)>2:
            bo,so = [],[]
            start = 0
            for end in edges[1:]:
                if end-start > blocksize or end==len(deltas):
                    if statewise:
                        pair,_ = self.best(deltas,start,end,double=True)
                        if pair is not None: so.append(pair)
                    one,_ = self.best(deltas,start,end)
                    if one>0: bo.append(one)
                    start=end
        one = self.best(deltas,boundopt=bo)
        two = self.best(deltas,stateopt=so,double=True) if statewise else (None,None)
        return one,two
