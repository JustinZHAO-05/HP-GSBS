"""Run remaining measurements sequentially; separate outputs never overwritten."""
import argparse,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--fixtures",required=True); p.add_argument("--output",required=True)
    p.add_argument("--only",nargs="+")
    p.add_argument("--revision",default="working tree; inspect environment source_hashes")
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    jobs=[("throughput",["benchmarks/v02_throughput.py","--output",str(out/"throughput.json"),"--modes","materialized","streaming"]),
          ("cold",["benchmarks/v02_cold.py","--output",str(out/"cold.json")]),
          ("profile",["benchmarks/v02_profile.py","--fixtures",a.fixtures,"--output",str(out/"profile")]),
          ("resident-fit",["benchmarks/v02.py","--revision",a.revision,"--fixtures",a.fixtures,
            "--output",str(out/"resident_fit"),"--backends","torch-cuda","triton-cuda","--candidate-backend","triton-resident"]),
          ("memory",["benchmarks/v02_memory.py","--output",str(out/"memory.json")]),
          ("precision",["benchmarks/v02_precision.py","--fixtures",a.fixtures,"--output",str(out/"precision.json")]),
          ("components",["benchmarks/v02_components.py","--output",str(out/"components.json")]),
          ("ablation",["benchmarks/v02_ablation.py","--fixtures",a.fixtures,"--output",str(out/"ablation.json")])]
    for name,command in jobs:
        if a.only and name not in a.only: continue
        print("START",name,flush=True)
        with (out/(name+".log")).open("x") as f:
            subprocess.run([sys.executable,*command],stdout=f,stderr=subprocess.STDOUT,check=True)
        print("DONE",name,flush=True)


if __name__=="__main__": main()
