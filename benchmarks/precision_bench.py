import argparse
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from hpgsbs import fit,GSBSConfig,GSBSResult
from hpgsbs.validation import compare


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); a=p.parse_args()
    rows=[]
    for item in json.loads(Path("results/fixtures/manifest.json").read_text()):
        ref=GSBSResult.load(f"results/fixtures/{item['id']}_full_cpu.npz")
        c=GSBSConfig(**ref.metadata["algorithm"],trace=True)
        for backend in ("torch-cuda","triton-cuda"):
            for dtype in ("float64","float32"):
                result=fit(ref.x,replace(c,dtype=dtype),backend)
                row=dict(id=item["id"],backend=backend,dtype=dtype,**compare(ref,result),metadata=result.metadata)
                rows.append(row); print({k:v for k,v in row.items() if k!="metadata"},flush=True)
                Path(a.output).write_text(json.dumps(rows,indent=2))
                result.save(f"results/fixtures/{item['id']}_{backend}_{dtype}.npz")


if __name__=="__main__": main()
