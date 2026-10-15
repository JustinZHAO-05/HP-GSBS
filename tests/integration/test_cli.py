import json
import subprocess
import sys
import numpy as np


def test_cli_commands(tmp_path):
    x=tmp_path/"x.npy"; np.save(x,np.random.default_rng(19).normal(size=(12,4)))
    r=tmp_path/"result.npz"
    def run(*args):
        p=subprocess.run([sys.executable,"-m","hpgsbs",*map(str,args)],capture_output=True,text=True)
        assert p.returncode==0,p.stderr
        return json.loads(p.stdout)
    assert run("fit",x,"--kmax",5,"--output",r)["status"]=="success"
    assert run("compare",r,r)["passed"]
    assert run("validate",x,"--kmax",5)["passed"]
    assert run("bench",x,"--kmax",5,"--repeats",1)["steady_median_s"]>0
    m=tmp_path/"manifest.json"; m.write_text(json.dumps([dict(id="rec1",path=str(x),config=dict(kmax=5))]))
    assert run("fit-batch",m,"--output",tmp_path/"batch")[0]["status"]=="success"
