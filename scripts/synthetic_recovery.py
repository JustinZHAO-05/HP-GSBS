import json
from pathlib import Path
import numpy as np
from hpgsbs import fit,GSBSConfig
from hpgsbs.validation import compare


def main():
    rows=[]
    for seed in (12,19):
        for noise in (.05,2.0):
            rng=np.random.default_rng(seed); means=rng.normal(size=(4,16))
            x=np.repeat(means,20,axis=0)+rng.normal(size=(80,16))*noise
            c=GSBSConfig(12,trace=True)
            ref=fit(x,c,"reference"); fast=fit(x,c)
            truth=[20,40,60]; predicted=np.flatnonzero(fast.deltas).tolist()
            rows.append(dict(seed=seed,noise_sd=noise,truth=truth,predicted=predicted,
                exact_recall=len(set(truth)&set(predicted))/3,
                exact_precision=len(set(truth)&set(predicted))/max(1,len(predicted)),
                optimal_k=int(fast.nstates),reference_parity=compare(ref,fast)))
    Path("results/benchmarks/synthetic_recovery.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__": main()
