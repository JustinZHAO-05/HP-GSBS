"""Sequential final-source measurements; preserves all earlier/failed runs."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--fixtures",required=True); p.add_argument("--output",required=True)
    p.add_argument("--revision",required=True)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    provenance=dict(revision=a.revision,
        source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path("src").rglob("*.py")},
        protocol="all measurement jobs sequential; fresh output directory; no private inputs exported",jobs=[])
    jobs=[("paired",["benchmarks/v02_paired.py","--fixtures",a.fixtures,"--output",str(out/"paired")]),
          ("resident-component",["benchmarks/v02_resident.py","--output",str(out/"resident_component.json")]),
          ("suite",["benchmarks/v02_suite.py","--fixtures",a.fixtures,"--output",str(out/"suite"),"--revision",a.revision]),
          ("long",["benchmarks/v02_validate.py","--long","--output",str(out/"long")])]
    for name,command in jobs:
        print("START",name,flush=True)
        with (out/(name+".log")).open("x") as f:
            proc=subprocess.run([sys.executable,*command],stdout=f,stderr=subprocess.STDOUT)
        provenance["jobs"].append(dict(name=name,returncode=proc.returncode))
        (out/"provenance.json").write_text(json.dumps(provenance,indent=2))
        if proc.returncode: raise RuntimeError(f"{name} failed; inspect retained log")
        print("DONE",name,flush=True)


if __name__=="__main__": main()
