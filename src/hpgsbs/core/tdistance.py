import numpy as np
from scipy.stats import ttest_ind
from .interval_scores import zscore


def prefix2(x):
    return np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def rectangle(p, a, b, c, d):
    return p[b, d] - p[a, d] - p[b, c] + p[a, c]


class TDistance:
    def __init__(self, x, dmin=1, prefixes=None):
        self.x, self.dmin = x, dmin
        self.fallbacks = 0
        if prefixes is None:
            z = zscore(x)
            # np.cov subtracts residual row means; retain that detail.
            z = z - z.mean(1, keepdims=True)
            r = z @ z.T / (x.shape[1] - 1)
            mask = np.triu(np.ones(r.shape, dtype=bool), dmin)
            r = np.where(mask, r, 0)
            self.prefixes = (prefix2(mask.astype(np.int64)), prefix2(r), prefix2(r*r))
        else:
            self.prefixes = prefixes

    def stats(self, bounds):
        e = np.r_[0, np.flatnonzero(bounds)[np.flatnonzero(bounds) > 0], len(bounds)]
        within, between = np.zeros(3), np.zeros(3)
        for i, (a, b) in enumerate(zip(e[:-1], e[1:])):
            within += [rectangle(p, a, b, a, b) for p in self.prefixes]
            if i + 2 < len(e):
                between += [rectangle(p, a, b, b, e[i+2]) for p in self.prefixes]
        return within, between

    def exact(self, bounds):
        self.fallbacks += 1
        z = zscore(self.x)
        r = np.cov(z)
        states = np.cumsum(np.asarray(bounds) > 0)
        diff = states[None, :] - states[:, None]
        mask = np.triu(np.ones(r.shape, bool), self.dmin)
        a, b = r[mask & (diff == 0)], r[mask & (diff == 1)]
        return 0.0 if len(a) < 2 else float(ttest_ind(a, b, equal_var=False).statistic)

    def __call__(self, bounds):
        w, b = self.stats(bounds)
        if w[0] < 2:
            return 0.0
        if b[0] < 2:
            return float("nan")
        variances = np.array([(s[2] - s[1]**2/s[0])/(s[0]-1) for s in (w,b)])
        # Cancellation at near-zero variance requires the original two-pass test.
        if np.any(variances <= 1e-12):
            return self.exact(bounds)
        return float((w[1]/w[0] - b[1]/b[0]) / np.sqrt(variances[0]/w[0] + variances[1]/b[0]))
