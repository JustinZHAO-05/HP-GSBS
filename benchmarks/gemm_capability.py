import json
from pathlib import Path
from time import perf_counter
import torch


def main():
    n=2048; rows=[]
    old=torch.get_float32_matmul_precision()
    for label,dtype,precision in (("fp64",torch.float64,"highest"),("fp32",torch.float32,"highest"),("tf32_experimental",torch.float32,"high")):
        torch.set_float32_matmul_precision(precision)
        a=torch.randn((n,n),device="cuda",dtype=dtype); b=torch.randn_like(a)
        for _ in range(3): c=a@b
        ts=[]
        for _ in range(5):
            torch.cuda.synchronize(); start=perf_counter(); c=a@b; torch.cuda.synchronize(); ts.append(perf_counter()-start)
        import numpy as np
        rows.append(dict(mode=label,n=n,median_s=float(np.median(ts)),measured_tflops=2*n**3/float(np.median(ts))/1e12,tf32=precision=="high"))
    torch.set_float32_matmul_precision(old)
    Path("results/benchmarks/gemm_capability.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__": main()
