"""FP64 feature moments TD with bounded two-pass pair fallback.

Expanding ||sum u||² removes diagonal dot products. Expanding
||sum uu' ||F² removes squared diagonal dot products. Cross-state Frobenius
products count each Cartesian pair once. No N x P x P prefix is retained.
"""
import numpy as np
from .interval_scores import zscore


class FeatureMomentsTDistance:
    def __init__(self, x, dmin=1, tile_size=128):
        self.x, self.dmin, self.tile_size = x, dmin, tile_size
        z = zscore(x)
        self.u = z-z.mean(1,keepdims=True)
        self.h = x.shape[1]-1
        self.fallbacks = 0
        self.fallback_reasons = {}
        self.last_diagnostics = {}

    def moment(self,a,b):
        u = self.u[a:b]
        norms = np.einsum("ij,ij->i",u,u)
        return b-a, u.sum(0), u.T@u, norms.sum(), norms@norms

    def within(self,m):
        n,s,M,d1,d2 = m
        return np.array([n*(n-1)//2,(s@s-d1)/(2*self.h),
                         (np.sum(M*M)-d2)/(2*self.h**2)],dtype=float)

    def cross(self,m,n):
        return np.array([m[0]*n[0],m[1]@n[1]/self.h,
                         np.sum(m[2]*n[2])/self.h**2],dtype=float)

    def stats(self,bounds):
        edges = np.r_[0,np.flatnonzero(bounds)[np.flatnonzero(bounds)>0],len(bounds)]
        w,a = np.zeros(3),np.zeros(3)
        previous = None
        for lo,hi in zip(edges[:-1],edges[1:]):
            current = self.moment(lo,hi)
            w += self.within(current)
            if previous is not None: a += self.cross(previous,current)
            previous = current
        labels = np.cumsum(np.asarray(bounds)>0)
        for gap in range(1,min(self.dmin,len(bounds))):
            # O(NP) temporary, independent of dmin, N² and candidate count.
            r = np.einsum("ij,ij->i",self.u[:-gap],self.u[gap:])/self.h
            diff = labels[gap:]-labels[:-gap]
            for target,kind in ((w,0),(a,1)):
                vals = r[diff==kind]
                target -= [len(vals),vals.sum(),vals@vals]
        return w,a

    def pair_tiles(self,bounds):
        """Yield only W/A eligible pairs; tile storage <= tile_size²."""
        edges = np.r_[0,np.flatnonzero(bounds)[np.flatnonzero(bounds)>0],len(bounds)]
        tile = self.tile_size
        for k,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            regions = [(0,a,b)]
            if k+2 < len(edges): regions.append((1,b,edges[k+2]))
            for kind,c,d in regions:
                for i in range(a,b,tile):
                    ii = np.arange(i,min(i+tile,b))
                    for j in range(max(c,i+self.dmin) if kind==0 else c,d,tile):
                        jj = np.arange(j,min(j+tile,d))
                        r = self.u[ii]@self.u[jj].T
                        r *= 1/self.h  # np.cov normalization
                        yield kind,r[(jj[None,:]-ii[:,None])>=self.dmin]

    def exact(self,bounds,reason="explicit two-pass request"):
        self.fallbacks += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason,0)+1
        counts = np.zeros(2,dtype=np.int64)
        sums = np.zeros(2)
        for kind,values in self.pair_tiles(bounds):
            counts[kind] += len(values)
            sums[kind] += values.sum()
        if counts[0]<2: return 0.0
        if counts[1]<2: return float("nan")
        means = sums/counts
        m2 = np.zeros(2)
        for kind,values in self.pair_tiles(bounds):
            delta = values-means[kind]
            m2[kind] += delta@delta
        with np.errstate(divide="ignore",invalid="ignore"):
            value = float((means[0]-means[1])/np.sqrt(np.sum(m2/(counts-1)/counts)))
        self.last_diagnostics = dict(method="tiled-two-pass",reason=reason,
            status="finite" if np.isfinite(value) else "degenerate-pair-statistic",
            counts=counts.tolist(),m2=m2.tolist())
        if not np.isfinite(means).all() or not np.isfinite(m2).all():
            raise FloatingPointError("Unresolved nonfinite tiled TD statistics; no dense fallback allowed")
        return value

    def __call__(self,bounds):
        w,a = self.stats(bounds)
        if w[0]<2: return 0.0
        if a[0]<2: return float("nan")
        variances = np.array([(s[2]-s[1]**2/s[0])/(s[0]-1) for s in (w,a)])
        # Engineering risk detector, not an error-bound certificate. Never clamp.
        if not np.isfinite(variances).all() or np.any(variances<=1e-12):
            return self.exact(bounds,"moment variance cancellation risk")
        self.last_diagnostics = dict(method="feature-moments",status="finite")
        return float((w[1]/w[0]-a[1]/a[0])/np.sqrt(variances[0]/w[0]+variances[1]/a[0]))
