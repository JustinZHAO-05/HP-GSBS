import argparse
from pathlib import Path
import json
import numpy as np
from hpgsbs import GSBSConfig
from hpgsbs.benchmark import benchmark


def main():
    p=argparse.ArgumentParser(); p.add_argument("--backend",default="cpu-fast"); p.add_argument("--T",type=int,default=450)
    p.add_argument("--P",type=int,default=16); p.add_argument("--kmax",type=int,default=20)
    p.add_argument("--repeats",type=int,default=3); p.add_argument("--output",required=True)
    a=p.parse_args(); x=np.random.default_rng(20261016).normal(size=(a.T,a.P))
    result=benchmark(x,GSBSConfig(a.kmax),a.backend,a.repeats)
    Path(a.output).write_text(json.dumps(result,indent=2)); print(result["steady_median_s"])


if __name__=="__main__": main()
