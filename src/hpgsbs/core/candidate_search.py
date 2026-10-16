import numpy as np
from numba import njit
from statesegmentation import GSBS
from .streaming_candidates import stream_q, stream_prefix, scored_chunks, stream_window, candidate_count
from .budgets import ResourceBudgetExceeded, SearchCancelled


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
    def __init__(self, x, scores, tolerance=1e-11, backend="streaming", chunk_size=256,
                 *, max_candidates=50_000_000, max_fallback_elements=50_000_000,
                 max_workspace_bytes=2_147_483_648, resident_min_candidates=20_000,
                 progress=None, cancel=None):
        self.x, self.q, self.tolerance = x, scores, tolerance
        self.z = scores.z if getattr(scores,"supplied_z",False) else GSBS._zscore(x)
        self.fallbacks = 0
        self.fallback_reasons = {}
        self.backend, self.chunk_size = backend, chunk_size
        self.last_diagnostics = {}
        self._local_cache = {}
        self.cache_hits = 0
        self.gpu_calls = 0
        self.max_candidates, self.max_fallback_elements = max_candidates,max_fallback_elements
        self.max_workspace_bytes,self.resident_min_candidates = max_workspace_bytes,resident_min_candidates
        self.progress,self.cancel = progress,cancel

    def checkpoint(self,count):
        if self.cancel is not None and self.cancel():
            raise SearchCancelled("Candidate search cancelled; no successful fit produced")
        if self.progress is not None: self.progress(dict(self.last_diagnostics,evaluated_candidates=count))

    def summary(self,args):
        edges,start,end,allowed,opt,double,T=args
        if (self.backend=="triton-resident" and double and opt.shape[1]!=2
                and self.last_diagnostics.get("estimated_candidates",0)>=self.resident_min_candidates):
            from ..backends.resident_candidates import candidate_summary
            self.gpu_calls += 1
            # HostQ owns matching immutable copies from one precompute.
            return candidate_summary(self.q.device_Q,self.q.Q,*args,validate_host=False)
        return (stream_q(self.q.Q,*args) if self.q.Q is not None else
                stream_prefix(self.q,*args,self.chunk_size,self.checkpoint if self.progress is not None or self.cancel is not None else None))

    def window(self, bounds, lo, hi):
        if self.backend in ("streaming","triton-resident") and self.q.Q is not None:
            edges = np.concatenate(([0], np.flatnonzero(bounds), [len(bounds)]))
            flat, best, runner, nonfinite, changed, _ = stream_window(self.q.Q, edges, lo, hi, len(bounds))
            if not nonfinite and (not changed or best-runner > self.tolerance):
                return int(flat), float(best)
        return self.best(bounds>0, boundopt=np.arange(lo,hi))

    def best(self, deltas, start=0, end=None, *, boundopt=None, stateopt=None, double=False):
        if end is None: end = len(deltas)
        cuts = np.flatnonzero(deltas)
        edges = np.concatenate(([start], cuts[(cuts>start)&(cuts<end)], [end]))
        allowed = np.ones(end-start, dtype=np.bool_)
        if boundopt is not None:
            allowed[:] = False
            opts = np.asarray(boundopt, dtype=np.int64)
            opts = opts[(opts>=start)&(opts<end)]
            allowed[opts-start] = True
        opt = np.empty((0,0),dtype=np.int64) if stateopt is None else np.asarray(stateopt,dtype=np.int64).reshape(-1,2)
        args = (edges, start, end, allowed, opt, double, len(self.x))
        estimated = int(candidate_count(edges,start,allowed,opt,double))
        self.last_diagnostics = dict(estimated_candidates=estimated,evaluated_candidates=0,
            start=int(start),end=int(end),double=bool(double),strategy="streaming")
        self.checkpoint(0)
        if estimated > self.max_candidates:
            raise ResourceBudgetExceeded("Candidate domain exceeds max_candidates",self.last_diagnostics)
        workspace = self.chunk_size*(16*self.x.shape[1]*8+64)
        if self.q.Q is None and workspace > self.max_workspace_bytes:
            raise ResourceBudgetExceeded("Prefix chunk exceeds workspace budget",self.last_diagnostics)
        if self.backend == "materialized":
            if (end-start)**(2 if double else 1)*8>self.max_workspace_bytes:
                raise ResourceBudgetExceeded("Materialized candidates exceed workspace budget",self.last_diagnostics)
            return self._best_materialized(deltas,start,end,boundopt=boundopt,stateopt=stateopt,double=double)
        summary = self.summary(args)
        flat, score, runner, nonfinite, changed, count = summary
        self.last_diagnostics.update(candidates=int(count), gap=float(score-runner), nonfinite=bool(nonfinite),evaluated_candidates=int(count))
        self.checkpoint(int(count))
        if not changed: return (None if double else start), -1.0
        if nonfinite or score-runner <= self.tolerance:
            self.fallbacks += 1
            reason = "nonfinite interval score" if nonfinite else "near-tied candidates"
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason,0)+1
            elements=(end-start)*self.x.shape[1]
            self.last_diagnostics.update(fallback_reason=reason,fallback_strategy="bounded-original-expression",
                fallback_evaluated=0,first_problematic_candidate=int(flat),
                estimated_fallback_elements=int(estimated*elements),fallback_elements=0)
            if elements*8*8 > self.max_workspace_bytes:
                raise ResourceBudgetExceeded("Original-expression fallback exceeds workspace budget",self.last_diagnostics)
            # Revisit candidates without retaining a matrix or near-tie ledger.
            # Reproduce the original means/zscore expression, including NaNs.
            xt, zt = self.x[start:end], self.z[start:end]
            means = np.empty_like(xt, dtype=float)
            for a,b in zip(edges[:-1],edges[1:]):
                means[a-start:b-start] = xt[a-start:b-start].mean(0)
            best, winner, any_changed = -1.0, 0, False
            for rows, flats, values in scored_chunks(self.q, *args, self.chunk_size):
                for row, f, value in zip(rows, flats, values):
                    if not nonfinite and value < score-self.tolerance: continue
                    self.checkpoint(int(count))
                    work=(self.last_diagnostics["fallback_evaluated"]+1)*elements
                    if work>self.max_fallback_elements:
                        self.last_diagnostics["refused_candidate"]=int(f)
                        raise ResourceBudgetExceeded("Original-expression fallback exceeds work budget",self.last_diagnostics)
                    self.last_diagnostics["fallback_evaluated"] += 1
                    self.last_diagnostics["fallback_elements"] = int(work)
                    a,i,j,b = row-start
                    candidate = means.copy()
                    candidate[a:i] = xt[a:i].mean(0)
                    if double:
                        candidate[i:j] = xt[i:j].mean(0)
                        candidate[j:b] = xt[j:b].mean(0)
                    else: candidate[i:b] = xt[i:b].mean(0)
                    v = xt.shape[1]*(GSBS._zscore(candidate)*zt).mean()/(xt.shape[1]-1)
                    any_changed |= v != -1
                    if not np.isnan(best) and (np.isnan(v) or v>best):
                        best, winner = float(v), int(f)
            if not any_changed and self.last_diagnostics["fallback_evaluated"]==count:
                return (None if double else start), -1.0
            flat, score = winner, best
        pos = (int(flat)//(end-start)+start, int(flat)%(end-start)+start) if double else int(flat)+start
        return pos, float(score)

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

    def _best_materialized(self, deltas, start=0, end=None, *, boundopt=None, stateopt=None, double=False):
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
            reason = "nonfinite interval score" if not np.isfinite(scores).all() else "near-tied candidates"
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason,0)+1
            work=(self.last_diagnostics["estimated_candidates"] if not np.isfinite(scores).all() else len(near))*(end-start)*self.x.shape[1]
            if work>self.max_fallback_elements:
                raise ResourceBudgetExceeded("Materialized exact fallback exceeds work budget",dict(self.last_diagnostics,estimated_fallback_elements=work))
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
                if np.all(exact.ravel()[near]==-1):
                    return (None if double else start),-1.0
                scores = exact
        idx = np.unravel_index(scores.argmax(),scores.shape)
        return ((idx[0]+start,idx[1]+start) if double else idx[0]+start), float(scores[idx])

    def blocks(self, deltas, statewise, blocksize):
        edges = np.r_[0,np.flatnonzero(deltas),len(deltas)]
        bo,so = None,None
        active_cache = {}
        if len(edges)>2:
            bo,so = [],[]
            start = 0
            for end in edges[1:]:
                if end-start > blocksize or end==len(deltas):
                    # Exact block endpoints AND internal cuts define the cache.
                    # Only currently active blocks survive; no global winner is
                    # reused. Input/Q/precision are fixed for this search object.
                    key=(start,int(end),tuple(edges[(edges>start)&(edges<end)]),bool(statewise))
                    if self.backend != "materialized" and key in self._local_cache:
                        one,pair=self._local_cache[key]
                        self.cache_hits += 1
                    else:
                        pair=self.best(deltas,start,end,double=True)[0] if statewise else None
                        one=self.best(deltas,start,end)[0]
                    active_cache[key]=(one,pair)
                    if pair is not None: so.append(pair)
                    if one>0: bo.append(one)
                    start=end
        self._local_cache=active_cache
        one = self.best(deltas,boundopt=bo)
        two = self.best(deltas,stateopt=so,double=True) if statewise else (None,None)
        return one,two
