from collections import defaultdict


def buckets(recordings,configs):
    """Exact shapes avoid padding and feature-mask ambiguity."""
    groups=defaultdict(list)
    for i,(x,c) in enumerate(zip(recordings,configs)):
        groups[(tuple(x.shape),c.dtype,c.dmin,c.q_strategy)].append(i)
    return list(groups.values())
