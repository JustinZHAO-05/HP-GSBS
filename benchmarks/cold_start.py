"""Fresh subprocess and fresh JIT cache, unlike first-call timings in a warm process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter


def main():
    rows=[]
    code="""
import json
from time import perf_counter
start=perf_counter()
import numpy as np
from hpgsbs import fit,GSBSConfig
x=np.random.default_rng(20261016).normal(size=(450,16))
import sys
r=fit(x,GSBSConfig(20),sys.argv[1])
print(json.dumps(dict(import_and_fit_s=perf_counter()-start,metadata=r.metadata)))
"""
    for backend in ("cpu-fast","torch-cuda","triton-cuda"):
        with tempfile.TemporaryDirectory(prefix="hpgsbs-cold-") as tmp:
            env=os.environ.copy(); env["NUMBA_CACHE_DIR"]=str(Path(tmp)/"numba"); env["TRITON_CACHE_DIR"]=str(Path(tmp)/"triton")
            start=perf_counter(); proc=subprocess.run([sys.executable,"-c",code,backend],env=env,capture_output=True,text=True,check=True)
            rows.append(dict(backend=backend,process_wall_s=perf_counter()-start,**json.loads(proc.stdout)))
    Path("results/benchmarks/cold_start.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__": main()
