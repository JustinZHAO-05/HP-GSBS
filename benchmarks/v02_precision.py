"""FP32 is an extra experiment; report divergences without calling them parity."""
import argparse,json
from pathlib import Path
from hpgsbs import fit,GSBSConfig,GSBSResult
from hpgsbs.validation import compare


def main():
    p=argparse.ArgumentParser(); p.add_argument("--fixtures",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    rows=[]
    for index in range(4):
        ref=GSBSResult.load(Path(a.fixtures)/f"{index}_full_cpu.npz")
        for backend in ("cpu-fast","torch-cuda","triton-cuda"):
            r=fit(ref.x,GSBSConfig(**ref.metadata["algorithm"],dtype="float32",trace=True),backend)
            rows.append(dict(index=index,backend=backend,precision="float32",experimental=True,
                report=compare(ref,r),metadata=r.metadata))
            out.write_text(json.dumps(rows,indent=2)); print(index,backend,rows[-1]["report"],flush=True)


if __name__=="__main__": main()
