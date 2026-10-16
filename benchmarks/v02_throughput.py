"""Small EEG batch throughput, including pools, transfers, sync and controllers."""
import argparse,json,threading
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
import psutil
from hpgsbs import fit,fit_many,GSBSConfig
from hpgsbs.validation import compare
from hpgsbs.batch.memory_planner import estimated_bytes


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True)
    p.add_argument("--modes",nargs="+",default=["streaming"])
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    import torch
    rows=[]
    xs=[np.random.default_rng(200+i).normal(size=(450,16)) for i in range(64)]
    expected=[fit(x,GSBSConfig(20)) for x in xs]
    jobs=[("cpu-fast",w,32) for w in (1,2,4,8)]
    jobs += [(backend,1,b) for backend in ("torch-cuda","triton-cuda") for b in (1,2,4,8,16,32,64)]
    for backend,workers,batch in jobs:
        gpu=backend.endswith("cuda")
        if gpu and estimated_bytes(450,16,batch)>torch.cuda.mem_get_info()[0]*.55:
            rows.append(dict(backend=backend,batch=batch,status="skipped-memory-admission"))
            out.write_text(json.dumps(rows,indent=2)); continue
        for mode in a.modes:
            c=GSBSConfig(20,n_jobs=workers,mode="throughput",candidate_backend=mode)
            for iteration in range(4):
                done=threading.Event(); parent=psutil.Process(); peak=[0]
                def monitor():
                    while not done.is_set():
                        total=0
                        for proc in [parent,*parent.children(recursive=True)]:
                            try: total+=proc.memory_info().rss
                            except psutil.Error: pass
                        peak[0]=max(peak[0],total)
                        done.wait(.01)
                thread=threading.Thread(target=monitor,daemon=True); thread.start()
                if gpu: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=perf_counter()
                try:
                    results=fit_many(xs[:batch],c,backend,batch_size=batch)
                    if gpu: torch.cuda.synchronize()
                    wall=perf_counter()-start
                finally: done.set(); thread.join()
                reports=[compare(e,r) for e,r in zip(expected,results)]
                passed=len(results)==batch and all(r["passed"] for r in reports)
                row=dict(backend=backend,workers=workers,batch=batch,mode=mode,iteration=iteration,
                    warmup=iteration==0,wall_s=wall,fits_per_minute=60*batch/wall,
                    average_latency_s=wall/batch,peak_process_tree_rss_bytes=peak[0],all_outputs_passed=passed,
                    gpu_allocated=torch.cuda.max_memory_allocated() if gpu else None,
                    gpu_reserved=torch.cuda.max_memory_reserved() if gpu else None,
                    actual_gpu_precompute_batch_sizes=sorted(set(r.metadata.get("batch_size",1) for r in results)) if gpu else None,
                    precompute_transfer_s=results[0].metadata["timings"].get("transfer_s") if gpu else None,
                    gpu_precompute_wall_s=results[0].metadata["timings"].get("gpu_precompute_s") if gpu else None,
                    controller_sum_s=sum(r.metadata["search_s"] for r in results))
                rows.append(row); out.write_text(json.dumps(rows,indent=2))
                print(backend,workers,batch,mode,iteration,wall,passed,flush=True)
                if not passed: raise AssertionError(reports)


if __name__=="__main__": main()
