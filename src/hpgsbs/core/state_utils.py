import numpy as np
from scipy.stats import pearsonr


def boundary_strengths(x,bounds):
    """Same contiguous state means and SciPy Pearson formula, batched over pairs."""
    delta=np.asarray(bounds)>0
    edges=np.r_[0,np.flatnonzero(delta),len(x)]
    patterns=np.asarray([x[a:b].mean(0) for a,b in zip(edges[:-1],edges[1:])])
    strengths=np.zeros(len(x))
    if len(patterns)>1:
        strengths[delta]=1-pearsonr(patterns[1:],patterns[:-1],axis=1).statistic
    return strengths
