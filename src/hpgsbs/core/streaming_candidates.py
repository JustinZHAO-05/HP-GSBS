"""Bounded candidate enumeration and C-order, -1-sentinel reduction.

The unscored flat zero is always present. NaN beats every number; its first
occurrence wins. All-minus-one is distinct from a nonempty candidate domain.
No fastmath: score expression order matches full_scores.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def gate_counts(opt, end):
    first = np.zeros(end, np.int64)
    second = np.zeros(end, np.int64)
    if opt.shape[1] == 2:
        for r in range(len(opt)):
            a, b = opt[r]
            if 0 <= a < end: first[a] += 1
            if 0 <= b < end: second[b] += 1
    return first, second


@njit(cache=True)
def candidate_count(edges, start, allowed, opt, double):
    """Exact legal-domain size, O(T + K + gate rows), without enumeration."""
    first, second = gate_counts(opt, edges[-1])
    restricted = opt.shape[1] == 2
    count = 0
    for s in range(len(edges)-1):
        a,b = edges[s],edges[s+1]
        n1,n2 = 0,0
        for i in range(b-1,a,-1):
            if allowed[i-start]:
                if not double: count += 1
                elif not restricted: count += b-i-1
                elif first[i]+second[0]>=1:
                    count += b-i-1 if first[i]>=2 else (n1 if first[i]==1 else n2)
            n1 += second[i]>=1
            n2 += second[i]>=2
    return count


@njit(cache=False)
def candidates(edges, start, allowed, opt, double):
    """Yield only legal coordinates, globally ordered; no candidate ledger."""
    first, second = gate_counts(opt, edges[-1])
    restricted = opt.shape[1] == 2
    for s in range(len(edges)-1):
        a, b = edges[s], edges[s+1]
        for i in range(a+1, b):
            if not allowed[i-start]: continue
            if double:
                if restricted and first[i]+second[0] < 1: continue
                for j in range(i+1, b):
                    if not restricted or first[i]+second[j] >= 2:
                        yield a, i, j, b
            else:
                yield a, i, i, b


@njit(cache=True)
def reduce_values(flats, values, summary):
    """Merge an ordered chunk into (flat, best, runner, nonfinite, changed, n)."""
    flat, best, runner, nonfinite, changed, count = summary
    for k in range(len(values)):
        v = values[k]
        count += 1
        nonfinite = nonfinite or not np.isfinite(v)
        changed = changed or v != -1
        if not np.isnan(best):
            if np.isnan(v) or v > best:
                runner, best, flat = best, v, flats[k]
            elif v > runner:
                runner = v
    return flat, best, np.nan if np.isnan(best) else runner, nonfinite, changed, count


@njit(cache=True)
def stream_q(q, edges, start, end, allowed, opt, double, T):
    n = end-start
    base = 0.0
    for s in range(len(edges)-1):
        base += q[edges[s], edges[s+1]]
    flat, best, runner = 0, -1.0, -1.0
    nonfinite, changed, count = False, False, 0
    first, second = gate_counts(opt,end)
    restricted = opt.shape[1] == 2
    # Fuse enumeration here: cached compiled callers of a cached Numba
    # generator can lose its lowering registration in a fresh process. Keep
    # the separately testable generator for bounded Python chunk consumers.
    for s in range(len(edges)-1):
        a,b = edges[s],edges[s+1]
        for i in range(a+1,b):
            if not allowed[i-start]: continue
            if double and restricted and first[i]+second[0]<1: continue
            for j in range(i+1 if double else i,b if double else i+1):
                if double and restricted and first[i]+second[j]<2: continue
                if double:
                    v = (base-q[a,b]+q[a,i]+q[i,j]+q[j,b])*T/n
                else:
                    v = (base-q[a,b]+q[a,i]+q[i,b])*T/n
                count += 1
                nonfinite = nonfinite or not np.isfinite(v)
                changed = changed or v != -1
                f = (i-start)*n+j-start if double else i-start
                if not np.isnan(best):
                    if np.isnan(v) or v > best:
                        runner, best, flat = best, v, f
                    elif v > runner:
                        runner = v
    return flat, best, np.nan if np.isnan(best) else runner, nonfinite, changed, count


def candidate_chunks(edges, start, allowed, opt, double, chunk_size):
    buffer = np.empty((chunk_size, 4), dtype=np.int64)
    count = 0
    for row in candidates(edges, start, allowed, opt, double):
        buffer[count] = row
        count += 1
        if count == chunk_size:
            yield buffer
            count = 0
    if count:
        yield buffer[:count]


def scored_chunks(q, edges, start, end, allowed, opt, double, T, chunk_size):
    """Vectorized Q or prefix endpoint scoring, with at most C x P workspace."""
    base = sum(q(a,b) for a,b in zip(edges[:-1],edges[1:]))
    for rows in candidate_chunks(edges, start, allowed, opt, double, chunk_size):
        a,i,j,b = rows.T
        if double:
            values = (base-q(a,b)+q(a,i)+q(i,j)+q(j,b))*T/(end-start)
            flats = (i-start)*(end-start)+j-start
        else:
            values = (base-q(a,b)+q(a,i)+q(i,b))*T/(end-start)
            flats = i-start
        yield rows, flats, values


def stream_prefix(q, edges, start, end, allowed, opt, double, T, chunk_size, progress=None):
    summary = (0, -1.0, -1.0, False, False, 0)
    for _, flats, values in scored_chunks(q, edges, start, end, allowed, opt, double, T, chunk_size):
        summary = reduce_values(flats, values, summary)
        if progress is not None: progress(int(summary[-1]))
    return summary


@njit(cache=True)
def stream_window(q, edges, lo, hi, T):
    """Single-cut refinement after deletion, including distant target states."""
    base = 0.0
    for s in range(len(edges)-1): base += q[edges[s],edges[s+1]]
    flat, best, runner = 0, -1.0, -1.0
    nonfinite, changed, count = False, False, 0
    for s in range(len(edges)-1):
        a,b = edges[s],edges[s+1]
        for i in range(max(a+1,lo),min(b,hi)):
            v = (base-q[a,b]+q[a,i]+q[i,b])*T/T
            count += 1
            changed = changed or v != -1
            nonfinite = nonfinite or not np.isfinite(v)
            if not np.isnan(best):
                if np.isnan(v) or v > best:
                    runner, best, flat = best, v, i
                elif v > runner: runner = v
    return flat, best, np.nan if np.isnan(best) else runner, nonfinite, changed, count
