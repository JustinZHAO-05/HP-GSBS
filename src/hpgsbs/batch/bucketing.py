from collections import defaultdict


def buckets(recordings,configs):
    """Exact shapes avoid padding and feature-mask ambiguity."""
    groups=defaultdict(list)
    for i,(x,c) in enumerate(zip(recordings,configs)):
        groups[(tuple(x.shape),c.dtype,c.dmin,c.q_strategy,c.td_backend,
                c.candidate_backend,c.blas_threads,c.numba_threads,c.max_workspace_bytes)].append(i)
    return list(groups.values())
