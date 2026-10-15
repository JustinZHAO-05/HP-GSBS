import json
from pathlib import Path
from hpgsbs import GSBSResult
from hpgsbs.validation import compare


def main():
    rows=[]
    for i in range(4):
        ref=GSBSResult.load(f"results/fixtures/{i}_full_cpu.npz")
        for backend in ("torch-cuda","triton-cuda"):
            for dtype in ("float64","float32"):
                p=Path(f"results/fixtures/{i}_{backend}_{dtype}.npz")
                if not p.exists(): continue
                rows.append(dict(id=i,backend=backend,dtype=dtype,**compare(ref,GSBSResult.load(p))))
    p=Path("results/fixtures/0_full_reference.npz")
    if p.exists():
        ref=GSBSResult.load(p)
        for backend,name in (("cpu-fast","0_full_cpu"),("torch-cuda","0_torch-cuda_float64"),("triton-cuda","0_triton-cuda_float64")):
            rows.append(dict(id=0,baseline="fresh official trace",backend=backend,dtype="float64",**compare(ref,GSBSResult.load(f"results/fixtures/{name}.npz"))))
    Path("results/benchmarks/saved_validation.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=="__main__": main()
