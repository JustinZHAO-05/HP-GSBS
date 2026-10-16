"""Long first-state double split, all candidates audited in bounded chunks."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from hpgsbs.core.interval_scores import IntervalScores
from hpgsbs.core.streaming_candidates import stream_q,scored_chunks,reduce_values


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    rows=[]
    with threadpool_limits(limits=1):
        for t,p in [(1200,32),(5000,8)]:
            x=np.random.default_rng(20261016).normal(size=(t,p))
            full=IntervalScores(x); prefix=IntervalScores(x,"streaming")
            args=(np.array([0,t]),0,t,np.ones(t,dtype=bool),np.empty((0,0),np.int64),True,t)
            stream_q(full.Q,*args)
            start=perf_counter(); expected=stream_q(full.Q,*args); qtime=perf_counter()-start
            start=perf_counter(); summary=(0,-1.,-1.,False,False,0); error=0.; flushes=0
            for block,flats,values in scored_chunks(prefix,*args,256):
                a,i,j,b=block.T
                oracle=(full(0,t)-full(a,b)+full(a,i)+full(i,j)+full(j,b))*t/t
                error=max(error,float(np.max(np.abs(values-oracle))))
                summary=reduce_values(flats,values,summary); flushes+=1
            elapsed=perf_counter()-start
            assert summary[0]==expected[0] and summary[-1]==(t-1)*(t-2)//2
            rows.append(dict(shape=[t,p],double=True,states=1,candidates=summary[-1],flushes=flushes,
                max_chunk=256,max_candidate_error=error,winner_equal=True,q_stream_wall_s=qtime,
                prefix_stream_with_per_candidate_oracle_wall_s=elapsed,
                scope="component only; audit overhead included; not a full-fit speedup"))
            out.write_text(json.dumps(rows,indent=2)); print(rows[-1],flush=True)


if __name__=="__main__": main()
