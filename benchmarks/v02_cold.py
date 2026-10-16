"""Fresh subprocess and isolated Numba/Triton cache, including imports and fit."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
from time import perf_counter


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    rows=[]
    code="""
import json,sys
from time import perf_counter
start=perf_counter()
import numpy as np
from hpgsbs import fit,GSBSConfig
x=np.random.default_rng(20261016).normal(size=(450,16))
r=fit(x,GSBSConfig(20,candidate_backend=sys.argv[2]),sys.argv[1])
print(json.dumps(dict(import_and_fit_s=perf_counter()-start,metadata=r.metadata)))
"""
    for backend in ("cpu-fast","torch-cuda","triton-cuda"):
        for mode in ("materialized","streaming"):
            with tempfile.TemporaryDirectory(prefix="hpgsbs-v02-cold-") as tmp:
                env=os.environ.copy(); env["NUMBA_CACHE_DIR"]=str(Path(tmp)/"numba"); env["TRITON_CACHE_DIR"]=str(Path(tmp)/"triton")
                start=perf_counter()
                proc=subprocess.run([sys.executable,"-c",code,backend,mode],env=env,capture_output=True,text=True,check=True)
                rows.append(dict(backend=backend,mode=mode,process_wall_s=perf_counter()-start,
                    repetitions=1,**json.loads(proc.stdout)))
                out.write_text(json.dumps(rows,indent=2)); print(backend,mode,rows[-1]["process_wall_s"],flush=True)


if __name__=="__main__": main()
