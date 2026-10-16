"""Isolate candidate streaming, refinement window and cache in full BBBD0 fits.

Temporary substitutions affect this process only and are always restored;
official statesegmentation is never patched.
"""
import argparse,json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
from hpgsbs import fit,GSBSConfig
from hpgsbs.core.candidate_search import CandidateSearch
from hpgsbs.validation import compare


def main():
    p=argparse.ArgumentParser(); p.add_argument("--fixtures",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    x=np.load(Path(a.fixtures)/"0.npy").astype(float); c=GSBSConfig(131,dmin=5)
    expected=fit(x,replace(c,candidate_backend="materialized")); rows=[]
    old_blocks,old_window=CandidateSearch.blocks,CandidateSearch.window
    def no_cache(self,*args):
        self._local_cache.clear()
        return old_blocks(self,*args)
    def no_window(self,bounds,lo,hi):
        return self.best(bounds>0,boundopt=np.arange(lo,hi))
    modes=[("materialized",False,False),("stream-only",False,False),
           ("stream-window",False,True),("stream-window-cache",True,True)]
    for iteration in range(4):
        for mode,cache,window in (modes if iteration%2==0 else list(reversed(modes))):
            config=replace(c,candidate_backend="materialized" if mode=="materialized" else "streaming")
            try:
                CandidateSearch.blocks=old_blocks if cache else no_cache
                CandidateSearch.window=old_window if window else no_window
                start=perf_counter(); r=fit(x,config); elapsed=perf_counter()-start
            finally:
                CandidateSearch.blocks, CandidateSearch.window=old_blocks,old_window
            report=compare(expected,r)
            rows.append(dict(mode=mode,iteration=iteration,warmup=iteration==0,wall_s=elapsed,
                             report=report,metadata=r.metadata))
            out.write_text(json.dumps(rows,indent=2)); print(mode,iteration,elapsed,report["passed"],flush=True)
            if not report["passed"]: raise AssertionError(report)


if __name__=="__main__": main()
