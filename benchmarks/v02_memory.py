"""Fresh-process long-N A/B at a memory-admitted dense size; no dense 50k run."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import psutil
from hpgsbs import fit,GSBSConfig
from hpgsbs.benchmark import benchmark
from hpgsbs.batch.memory_planner import estimated_bytes


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output"); p.add_argument("--child",choices=["materialized","low-memory"])
    a=p.parse_args()
    if a.child:
        x=np.random.default_rng(20261016).normal(size=(5000,8))
        extra=(dict(candidate_backend="materialized") if a.child=="materialized" else
               dict(q_strategy="streaming",td_backend="feature-moments"))
        c=GSBSConfig(16,statewise_detection=False,finetune=0,blocksize=0,**extra)
        measured=benchmark(x,c,"cpu-fast",repeats=3); r=fit(x,c)
        print(json.dumps(dict(mode=a.child,all_bounds_sha256=hashlib.sha256(r.all_bounds.tobytes()).hexdigest(),
            tdists=r.tdists.tolist(),nstates=int(r.nstates),**measured)))
        return
    out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    rows=[]
    for mode in ("materialized","low-memory"):
        if mode=="materialized" and estimated_bytes(5000,8)>.4*psutil.virtual_memory().available:
            rows.append(dict(mode=mode,status="skipped-conservative-memory-admission")); continue
        process=subprocess.run([sys.executable,__file__,"--child",mode],capture_output=True,text=True,check=True)
        rows.append(json.loads(process.stdout)); out.write_text(json.dumps(rows,indent=2))
        print(mode,rows[-1]["steady_median_s"],flush=True)
    if all("tdists" in r for r in rows):
        assert rows[0]["all_bounds_sha256"]==rows[1]["all_bounds_sha256"]
        np.testing.assert_allclose(rows[0]["tdists"],rows[1]["tdists"],atol=1e-8,rtol=1e-8)


if __name__=="__main__": main()
