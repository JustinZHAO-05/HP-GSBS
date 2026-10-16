"""Experimental active-region TD statistics. Not a production fit backend.

Partition operations are represented by their resulting ordered cuts, so split,
double split, deletion, insertion and arbitrarily distant moves use the same
invalidation rule. Only unchanged intervals and unchanged adjacent pairs reuse
statistics. No monotone-split complexity bound is claimed for this algorithm.
"""
import numpy as np
from .feature_moments import FeatureMomentsTDistance


class IncrementalTDistance(FeatureMomentsTDistance):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.moments = {}
        self.regions = {}
        self.totals = (np.zeros(3),np.zeros(3))
        self.updates = 0
        self.commits = 0
        self.rebuild_interval = 32

    def corrected_region(self,key):
        kind,a,b,c,d = key
        value = (self.within(self.moments[a,b]) if kind==0 else
                 self.cross(self.moments[a,b],self.moments[c,d]))
        for gap in range(1,min(self.dmin,d-a)):
            lo,hi = (a,b-gap) if kind==0 else (max(a,c-gap),min(b,d-gap))
            if hi<=lo: continue
            r=self.shortbands.get(gap)
            if r is None:
                u,v = self.u[lo:hi],self.u[lo+gap:hi+gap]
                r=np.einsum("ij,ij->i",u,v)/self.h
            else:r=r[lo:hi]
            value -= [len(r),r.sum(),r@r]
        return value

    def stats(self,bounds):
        cuts=np.flatnonzero(bounds)
        e=np.r_[0,cuts[cuts>0],len(bounds)]
        intervals=list(zip(e[:-1],e[1:]))
        self.moments={s:self.moments[s] if s in self.moments else self.moment(*s) for s in intervals}
        keys=[(0,a,b,a,b) for a,b in intervals]
        keys += [(1,*left,*right) for left,right in zip(intervals[:-1],intervals[1:])]
        keyset=set(keys)
        new={}
        totals=[s.copy() for s in self.totals]
        for key,value in self.regions.items():
            if key not in keyset: totals[key[0]] -= value
        for key in keys:
            if key in self.regions: new[key]=self.regions[key]
            else:
                value=self.corrected_region(key)
                new[key]=value
                totals[key[0]] += value
        self.regions,self.totals=new,tuple(totals)
        self.updates += 1
        return tuple(s.copy() for s in self.totals)

    def __call__(self,bounds):
        """Pure proposal: an unaccepted single/double cannot pollute active state."""
        snapshot=self.moments,self.regions,self.totals,self.updates
        try:
            return super().__call__(bounds)
        finally:
            self.moments,self.regions,self.totals,self.updates=snapshot

    def commit(self,bounds):
        """Commit the partition after sequential finetuning; bounded active cache.

        Rebase from current regions every 32 accepted partitions. No history of
        discarded proposals is retained; proposal rollback also runs on errors.
        """
        self.commits+=1
        if self.commits % self.rebuild_interval==0:
            self.moments,self.regions={},{}
            self.totals=(np.zeros(3),np.zeros(3))
        return self.stats(bounds)
