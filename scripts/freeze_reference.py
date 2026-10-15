"""Freeze new reference fits in this workspace, never in the research tree."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from hpgsbs import fit, GSBSConfig


def main():
    p=argparse.ArgumentParser(); p.add_argument("--full",action="store_true"); p.add_argument("--index",type=int)
    args=p.parse_args()
    manifest=json.loads(Path("results/fixtures/manifest.json").read_text())
    rows=[]
    for item in manifest:
        if args.index is not None and int(item["id"])!=args.index: continue
        spec=item["source_spec"]
        x=np.load(item["path"])
        if not args.full: x=x[:96]
        config=GSBSConfig(spec["kmax"] if args.full else 12,dmin=spec["dmin"],trace=True,**spec["parameters"])
        label=item["id"]+("_full" if args.full else "_dev96")
        path=Path("results/fixtures")/(label+"_reference.npz")
        if path.exists(): continue
        print("START",label,x.shape,config,flush=True)
        result=fit(x,config,"reference")
        result.metadata.update(source_spec=spec,development_crop=not args.full)
        result.save(path)
        rows.append(dict(id=label,shape=list(x.shape),**result.metadata))
        print("DONE",label,result.metadata["elapsed_s"],flush=True)
        Path("results/fixtures/"+label+"_receipt.json").write_text(json.dumps(rows[-1],indent=2))


if __name__=="__main__": main()
