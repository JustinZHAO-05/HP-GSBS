"""Profile production hybrid CUDA on BBBD0; no private input is exported."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from hpgsbs import fit,GSBSConfig


def main():
    p=argparse.ArgumentParser(); p.add_argument("--fixtures",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    x=np.load(Path(a.fixtures)/"0.npy").astype(float); c=GSBSConfig(131,dmin=5)
    for backend in ("torch-cuda","triton-cuda"):
        fit(x,c,backend)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            r=fit(x,c,backend)
            torch.cuda.synchronize()
        (out/(backend+".txt")).write_text(prof.key_averages().table(sort_by="cuda_time_total",row_limit=40))
        prof.export_chrome_trace(str(out/(backend+".json")))
        (out/(backend+"_metadata.json")).write_text(json.dumps(r.metadata,indent=2))


if __name__=="__main__": main()
