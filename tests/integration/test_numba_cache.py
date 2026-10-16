"""Regression for generator lowering lost across fresh processes/cache loads."""
import os
from pathlib import Path
import subprocess
import sys


def test_lazy_generator_then_fresh_fullq_signatures(tmp_path):
    env=os.environ.copy(); env["NUMBA_CACHE_DIR"]=str(tmp_path/"cache")
    code="""
import sys,numpy as np
from hpgsbs import fit,GSBSConfig
x=np.random.default_rng(56).normal(size=(18,5))
r=fit(x,GSBSConfig(7,q_strategy=sys.argv[1],dtype=sys.argv[2]))
assert r.metadata['status']=='success'
"""
    for strategy,dtype in [("lazy","float64"),("full","float64"),("full","float32")]:
        proc=subprocess.run([sys.executable,"-c",code,strategy,dtype],capture_output=True,text=True,env=env,timeout=60)
        assert proc.returncode==0,proc.stderr
