"""Supplemental resident profiler. Fits are warm; profiling is not latency data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from hpgsbs import fit,GSBSConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).read_bytes();(out/'resident_reproduce.py').write_bytes(code)
    x=np.load(Path(a.fixtures)/'0.npy').astype(float)
    c=GSBSConfig(131,dmin=5,candidate_backend='triton-resident')
    for backend in ('torch-cuda','triton-cuda'):
        prefix=out/('resident-'+backend)
        if prefix.with_suffix('.json').exists():raise FileExistsError(prefix)
        fit(x,c,backend)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            result=fit(x,c,backend);torch.cuda.synchronize()
        table=prof.key_averages()
        prefix.with_suffix('.txt').write_text(table.table(sort_by='self_cpu_time_total',row_limit=60))
        prof.export_chrome_trace(str(prefix.with_suffix('.json')))
        records=[dict(key=e.key,count=e.count,cpu_time_total_us=e.cpu_time_total,
            device_time_total_us=getattr(e,'device_time_total',0)) for e in table]
        metadata=dict(metadata=result.metadata,events=records,profiling_is_not_latency_measurement=True,
            reproduction_source_sha256=hashlib.sha256(code).hexdigest())
        Path(str(prefix)+'_metadata.json').write_text(json.dumps(metadata,indent=2))
        print(backend,result.metadata['candidate_gpu_calls'],flush=True)


if __name__=='__main__':main()
