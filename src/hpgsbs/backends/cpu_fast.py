from time import perf_counter
from dataclasses import asdict
import numpy as np
from statesegmentation import GSBS
from ..core.interval_scores import IntervalScores
from ..core.tdistance import TDistance
from ..core.candidate_search import CandidateSearch
from ..core.finetuning import finetune
from ..reference import provenance
from ..result import GSBSResult


def run_search(x, config, q, td, *, backend="cpu-fast", timings=None, progress=None, cancel=None):
    """Sequential controller follows GSBS.fit a75d2f1; MIT attribution in NOTICE."""
    model = GSBS(x=x, **config.algorithm())
    search = CandidateSearch(x,q,config.tie_tolerance,config.candidate_backend,config.candidate_chunk_size,
        max_candidates=config.max_candidates,max_fallback_elements=config.max_fallback_elements,
        max_workspace_bytes=config.max_workspace_bytes,resident_min_candidates=config.resident_min_candidates,
        progress=progress,cancel=cancel)
    trace=[]
    times = dict(precompute_s=0.0,candidate_s=0.0,tdistance_s=0.0,finetune_s=0.0)
    if timings: times.update(timings)
    k=2
    start=perf_counter()
    while k < config.kmax+1:
        before=model._bounds.tolist() if config.trace else None
        ts=perf_counter()
        (one,os),(two,ds)=search.blocks(model._deltas,config.statewise_detection,config.blocksize)
        times["candidate_s"] += perf_counter()-ts
        ts=perf_counter()
        d1=model._deltas.copy(); d1[one]=True
        t1=td(d1); t2=None
        increment=1
        if two is not None:
            d2=model._deltas.copy(); d2[list(two)]=True
            t2=td(d2)
            if np.isfinite(t1) and np.isfinite(t2) and abs(t1-t2)<config.tie_tolerance*max(1,abs(t1),abs(t2)):
                t1,t2=td.exact(d1),td.exact(d2)
            if t2>t1: increment=2
        model._deltas=d2 if increment==2 else d1
        if increment==2: model._bounds[list(two)]=k+1
        else: model._bounds[one]=k
        model.all_bounds[k:k+increment]=model._bounds
        times["tdistance_s"] += perf_counter()-ts
        pre=model._bounds.tolist() if config.trace else None
        ts=perf_counter()
        fine_trace=[] if config.trace else None
        if config.finetune != 0 and k>2:
            model._bounds=finetune(model,search,fine_trace)
            model._deltas=model._bounds>0
            model.all_bounds[k:k+increment]=model._bounds
        times["finetune_s"] += perf_counter()-ts
        if hasattr(td,"commit"): td.commit(model._deltas)
        ts=perf_counter()
        if increment>1:
            model._tdists[k]=model._tdists[k-1]
            model._tdists[k+1]=td(model._deltas)
        else: model._tdists[k]=td(model._deltas)
        times["tdistance_s"] += perf_counter()-ts
        if config.trace:
            trace.append(dict(k=k,before=before,single=int(one),double=list(map(int,two)) if two else None,
                single_score=os,double_score=ds,single_t=t1,double_t=t2,increment=increment,pre_finetune=pre,
                finetune_moves=fine_trace,after=model._bounds.tolist()))
        k+=increment
    model._argmax=model._tdists.argmax()
    return GSBSResult(model,dict(backend=backend, precision=config.dtype,input_shape=list(x.shape),
        status="success",maturity=config.maturity, algorithm=config.algorithm(),compute_config=asdict(config),upstream=provenance(), timings=times,
        search_s=perf_counter()-start, candidate_fallbacks=search.fallbacks,tdistance_fallbacks=td.fallbacks,
        candidate_fallback_reasons=search.fallback_reasons,
        candidate_gpu_calls=search.gpu_calls,
        precision_detail="FP64 throughout" if config.dtype=="float64" else "FP32 interval table; FP64 controller, output means and CPU t-distance",
        candidate_backend=config.candidate_backend,q_strategy=config.q_strategy,local_cache_hits=search.cache_hits,
        td_backend=config.td_backend,tdistance_fallback_reasons=getattr(td,"fallback_reasons",{}),
        td_numerical_scope="experimental validated domain; not a rigorous error certificate" if config.td_backend=="feature-moments" else "v0.1 prefix2d guards",
        compatibility="fp64-validated-domain" if config.maturity=="stable" else "experimental-validated-domain"),trace)


def fit_cpu(x,config,y=None,*,progress=None,cancel=None):
    start=perf_counter()
    q=IntervalScores(x,config.q_strategy,config.tile_size,config.dtype)
    if config.td_backend == "feature-moments":
        from ..core.feature_moments import FeatureMomentsTDistance
        td=FeatureMomentsTDistance(x if y is None else y,config.dmin,config.tile_size,
            max_pair_evaluations=config.max_pair_evaluations)
    else: td=TDistance(x if y is None else y, config.dmin)
    result=run_search(x,config,q,td,timings={"precompute_s":perf_counter()-start},progress=progress,cancel=cancel)
    result.model.y=y
    result.metadata.update(tdistance_input="x" if y is None else "y",y_shape=None if y is None else list(y.shape))
    result.metadata["elapsed_s"]=perf_counter()-start
    return result
