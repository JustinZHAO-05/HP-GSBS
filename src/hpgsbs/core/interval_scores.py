"""Interval contributions, without any T x T x P intermediate."""
import numpy as np


def validate_input(x, kmax=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("GSBS requires time x features with T >= 3 and P >= 2")
    if kmax is not None and kmax >= len(x):
        raise ValueError("Require kmax < T, reserving room for double insertion")
    if not np.isfinite(x).all():
        raise ValueError("NaN/Inf input is not defined by GSBS spatial correlation")
    sd = x.std(axis=1, ddof=1)
    if np.any(sd == 0) or not np.isfinite(sd).all():
        raise ValueError("Spatially constant/zero-filled row: GSBS z-score undefined; no imputation performed")
    return x


def zscore(x):
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True, ddof=1)


def conditioning_reason(x, dtype="float64"):
    """Near-constant detector, never a denominator regularizer.

    Uses the eps**0.75 scale also used for near-constant Pearson inputs.
    An unsafe input is evaluated by the original implementation, not 'repaired'.
    """
    means=np.mean(x,axis=1,keepdims=True)
    norm=np.linalg.norm(x-means,axis=1)
    if np.any(norm < np.finfo(dtype).eps**0.75*np.abs(means[:,0])):
        return "spatial pattern ill-conditioned relative to row offset"
    return None


class IntervalScores:
    def __init__(self, x, strategy="full", tile_size=128, dtype="float64", z=None):
        self.x = np.asarray(x, dtype=dtype)
        self.supplied_z = z is not None
        self.z = zscore(self.x) if z is None else np.asarray(z,dtype=dtype)
        if self.z.shape != self.x.shape: raise ValueError("Z must have the same shape as X")
        self.T, self.P = x.shape
        centered = self.x - self.x.mean(1, keepdims=True)
        self.A = np.vstack((np.zeros((1, self.P)), (self.x if strategy=="streaming" else centered).cumsum(0))).astype(dtype)
        self.B = np.vstack((np.zeros((1, self.P)), self.z.cumsum(0))).astype(dtype)
        self.Q = None
        self.strategy = strategy
        if strategy == "full":
            # Direct prefix differences avoid cancellation in Gram distance identities.
            # At most tile_size x P temporary memory; Q is the only quadratic array.
            q = np.zeros((self.T + 1, self.T + 1), dtype=dtype)
            for a in range(self.T):
                for start in range(a + 1, self.T + 1, tile_size):
                    b = np.arange(start, min(self.T + 1, start + tile_size))
                    q[a, b] = self._direct(a, b)
            self.Q = q

    def _direct(self, a, b):
        da, db = self.A[b] - self.A[a], self.B[b] - self.B[a]
        if self.strategy == "streaming": da = da-da.mean(axis=-1,keepdims=True)
        norm = np.linalg.norm(da, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sum(da * db, axis=-1) / (self.T * np.sqrt(self.P - 1) * norm)
        # No epsilon: an exactly flat interval has undefined spatial correlation.
        return out

    def __call__(self, a, b):
        return self.Q[a, b] if self.Q is not None else self._direct(a, b)


def interval_reference(x, a, b):
    pattern = x[a:b].mean(0, keepdims=True)
    return np.sum(zscore(pattern) * zscore(x[a:b])) / (len(x) * (x.shape[1] - 1))
