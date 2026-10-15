def estimated_bytes(T,P,batch=1,itemsize=8):
    """Conservative simultaneous peak: Q, Grams, R, prefixes, temporaries, allocator.

    Includes 16 quadratic arrays and linear inputs, plus 256 MiB workspace.
    Actual allocated/reserved peak must still be measured. Not an OOM guarantee.
    """
    return int(batch*(16*(T+1)**2*itemsize+8*T*P*itemsize)+256*1024**2)


def automatic_batch_size(T,P,free_bytes,fraction=.55,itemsize=8):
    budget=max(0,free_bytes*fraction-256*1024**2)
    per=estimated_bytes(T,P,1,itemsize)-256*1024**2
    return max(1,int(budget//per))
