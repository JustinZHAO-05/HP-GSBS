"""Sequential same-process/same-environment end-to-end comparison, synchronized CUDA."""
import json
from pathlib import Path
import numpy as np
from hpgsbs import GSBSConfig,fit
from hpgsbs.benchmark import benchmark
from hpgsbs.validation import compare


def main():
    rows=[]
    workloads=[("Small",450,16,20),("Medium",900,32,30),("Large",1500,64,40),("XL",2100,64,50),("BBBD0",1336,64,131)]
    for name,T,P,k in workloads:
        x=np.load("results/fixtures/0.npy").astype(float) if name=="BBBD0" else np.random.default_rng(20261016).normal(size=(T,P))
        c=GSBSConfig(k,dmin=5 if name=="BBBD0" else 1)
        baseline=fit(x,c)
        backends=("reference","cpu-fast","torch-cuda","triton-cuda") if name=="Small" else ("cpu-fast","torch-cuda","triton-cuda")
        for backend in backends:
            print("BENCH",name,backend,flush=True)
            measured=benchmark(x,c,backend,repeats=3)
            validation=compare(baseline,fit(x,c,backend))
            rows.append(dict(workload=name,validation_against_cpu=validation,**measured))
            Path("results/benchmarks/delivery_latency.json").write_text(json.dumps(rows,indent=2))
            print(name,backend,measured["steady_median_s"],validation["passed"],flush=True)


if __name__=="__main__": main()
