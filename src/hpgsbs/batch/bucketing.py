from collections import defaultdict


def buckets(recordings,configs):
    """Group exact shapes and compatible compute/resource scheduling policies."""
    groups=defaultdict(list)
    for i,(x,c) in enumerate(zip(recordings,configs)):
        groups[(tuple(x.shape),c.dtype,c.dmin,c.q_strategy,c.td_backend,
                c.candidate_backend,c.blas_threads,c.numba_threads,c.max_workspace_bytes,
                c.mode,c.memory_fraction)].append(i)
    return list(groups.values())
