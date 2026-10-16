"""Private cache regressions, public long-N components and bounded fits."""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from hpgsbs import fit,GSBSConfig,GSBSResult
from hpgsbs.validation import compare
from hpgsbs.benchmark import benchmark
from hpgsbs.core.feature_moments import FeatureMomentsTDistance


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures")
    p.add_argument("--output",required=True)
    p.add_argument("--long",action="store_true")
    p.add_argument("--backends",nargs="+",default=["cpu-fast"])
    args=p.parse_args(); out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    rows=[]
    def save(): (out/"validation.json").write_text(json.dumps(rows,indent=2))
    if not args.long:
        fixtures=Path(args.fixtures)
        for index in range(4):
            full=GSBSResult.load(fixtures/f"{index}_full_cpu.npz")
            cached=GSBSResult.load(fixtures/("0_full_reference.npz" if index==0 else f"{index}_full_cpu.npz"))
            config=GSBSConfig(**full.metadata["algorithm"],trace=True)
            modes=[("full",config)]
            modes.append(("low-memory",replace(config,q_strategy="streaming",td_backend="feature-moments")))
            for name,c in modes:
                for backend in args.backends:
                    if name=="low-memory" and backend!="cpu-fast": continue
                    print("START",index,name,backend,flush=True)
                    start=perf_counter(); candidate=fit(full.x,c,backend)
                    report=compare(cached,candidate,level="full-search-trace")
                    rows.append(dict(record=index,mode=name,backend=backend,wall_s=perf_counter()-start,
                        source="official-full" if index==0 else "v0.1-full-validated-against-official-history",
                        input_sha256=hashlib.sha256(full.x.tobytes()).hexdigest(),config=asdict(c),
                        report=report,metadata=candidate.metadata))
                    save(); print(index,name,backend,report,flush=True)
                    if not report["passed"]: raise AssertionError(report)
            dev=GSBSResult.load(fixtures/f"{index}_dev96_reference.npz")
            c=GSBSConfig(**dev.metadata["algorithm"],trace=True)
            # Fresh unchanged official code also records every refinement move.
            official=fit(dev.x,c,"reference")
            for name,config in [("full",c),("low-memory",replace(c,q_strategy="streaming",td_backend="feature-moments"))]:
                report=compare(official,fit(dev.x,config),level="finetune-move-trace")
                rows.append(dict(record=index,mode=name,scope="fresh-official-dev96-every-move",report=report))
                save()
                if not report["passed"]: raise AssertionError(report)
    else:
        for t,p in [(5000,8),(10000,16),(20000,32),(50000,64)]:
            print("LONG",t,p,flush=True)
            x=np.random.default_rng(20261016).normal(size=(t,p))
            c=GSBSConfig(16,statewise_detection=False,finetune=0,blocksize=0,
                q_strategy="streaming",td_backend="feature-moments")
            measured=benchmark(x,c,"cpu-fast",repeats=3)
            r=fit(x,c)
            # Audit both selected and final partitions; no full native long fit.
            # Deliberate fixed-partition oracle budget: at T<=50k two passes
            # contain at most T*(T-1) < 2.5e9 W/A pairs. No candidate sweep or
            # dense array is admitted by this validation-only override.
            td=FeatureMomentsTDistance(x,1,tile_size=128,max_pair_evaluations=3_000_000_000)
            checks=[]
            with threadpool_limits(limits=1):
                for k in sorted(set([int(r.nstates),16])):
                    b=r.all_bounds[k]>0
                    start=perf_counter(); oracle=td.exact(b)
                    moments=td(b)
                    checks.append(dict(k=k,moments=moments,tiled=oracle,error=abs(moments-oracle),
                        tiled_wall_s=perf_counter()-start,passed=bool(np.isclose(moments,oracle,atol=1e-8,rtol=1e-8))))
            rows.append(dict(scope="restricted-full-fit-and-fixed-partition-oracle",
                full_official_trace_certified=False,checks=checks,**measured))
            save()
            if not all(c["passed"] for c in checks): raise AssertionError(checks)
            print(t,p,measured["steady_median_s"],max(v["cpu_peak_rss_bytes"] for v in measured["trials"]),checks,flush=True)


if __name__=="__main__": main()
