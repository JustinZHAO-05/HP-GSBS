import sys,json
from pathlib import Path
import numpy as np
from hpgsbs import GSBSConfig
from hpgsbs.backends.torch_cuda import precompute


def main():
    rows=[]
    for T in (450,1500):
        x=np.random.default_rng(2).normal(size=(T,64)); c=GSBSConfig(20)
        for backend in ("torch-cuda","triton-cuda"):
            for i in range(4):
                _,stats=precompute([x],[c],backend=backend)
                rows.append(dict(T=T,backend=backend,iteration=i,**stats))
    Path(sys.argv[1]).write_text(json.dumps(rows,indent=2))


if __name__=="__main__": main()
