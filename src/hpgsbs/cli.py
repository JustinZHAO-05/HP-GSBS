import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from . import fit,GSBSConfig,GSBSResult


def main():
    p=argparse.ArgumentParser(prog="hpgsbs")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor")
    for name in ("fit","validate","bench"):
        s=sub.add_parser(name); s.add_argument("input"); s.add_argument("--config"); s.add_argument("--kmax",type=int)
        s.add_argument("--backend",default="cpu-fast"); s.add_argument("--dtype",choices=["float64","float32"])
        s.add_argument("--output"); s.add_argument("--y"); s.add_argument("--times"); s.add_argument("--trace",action="store_true")
        if name=="bench": s.add_argument("--repeats",type=int,default=3)
    s=sub.add_parser("compare"); s.add_argument("reference"); s.add_argument("candidate")
    s=sub.add_parser("fit-batch"); s.add_argument("manifest"); s.add_argument("--backend",default="cpu-fast")
    s.add_argument("--batch-size",default="auto"); s.add_argument("--workers",type=int,default=1); s.add_argument("--output",required=True)
    s.add_argument("--dtype",choices=["float64","float32"])
    a=p.parse_args()
    if a.command=="doctor":
        from .doctor import doctor
        print(json.dumps(doctor(),indent=2,ensure_ascii=False)); return
    if a.command=="compare":
        from .validation import compare
        report=compare(GSBSResult.load(a.reference),GSBSResult.load(a.candidate))
        print(json.dumps(report,indent=2)); raise SystemExit(0 if report["passed"] else 1)
    if a.command=="fit-batch":
        from .batch.scheduler import run_manifest
        tasks=json.loads(Path(a.manifest).read_text())
        if a.dtype:
            tasks=[{**t,"config":{**t["config"],"dtype":a.dtype}} for t in tasks]
        report=run_manifest(tasks,a.output,backend=a.backend,n_jobs=a.workers,batch_size=a.batch_size)
        print(json.dumps(report,indent=2)); raise SystemExit(1 if any(r["status"]=="failed" for r in report) else 0)
    kwargs=json.loads(Path(a.config).read_text()) if a.config else {}
    if a.kmax is not None: kwargs["kmax"]=a.kmax
    if a.dtype: kwargs["dtype"]=a.dtype
    if "kmax" not in kwargs: p.error("--kmax or a config containing kmax is required")
    kwargs["trace"]=a.trace or a.command=="validate"
    c=GSBSConfig(**kwargs); x=np.load(a.input,allow_pickle=False); y=np.load(a.y,allow_pickle=False) if a.y else None
    if a.command=="bench":
        from .benchmark import benchmark
        report=benchmark(x,c,a.backend,repeats=a.repeats,y=y)
    else:
        result=fit(x,c,a.backend,y=y)
        if a.command=="fit":
            if not a.output: p.error("fit requires --output")
            result.save(a.output,times=np.load(a.times) if a.times else None)
            print(json.dumps(result.metadata,indent=2)); return
        from .validation import compare
        ref=fit(x,replace(c,dtype="float64"),"reference",y=y)
        report=compare(ref,result)
    text=json.dumps(report,indent=2)
    if a.output:
        out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("x") as f: f.write(text)
    print(text)
    if a.command=="validate" and not report["passed"]: raise SystemExit(1)
