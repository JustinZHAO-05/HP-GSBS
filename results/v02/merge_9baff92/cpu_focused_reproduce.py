"""Focused persistent-process A/B; outputs checked after each timed fit."""
import os,sys,subprocess,json,hashlib
from pathlib import Path
import numpy as np
base=Path('results/v02/merge_9baff92');rows=[];processes=[]
try:
    for folder in ('tmp/baselines/accepted','.'):
        env=dict(os.environ,PYTHONPATH=str(Path(folder).resolve()/'src'))
        processes.append(subprocess.Popen([sys.executable,'benchmarks/v02_revision_compare.py','--worker','--fixtures',os.environ['HPGSBS_FIXTURES']],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True))
    reference=None
    for iteration in range(9):
        for index in ([0,1] if iteration%2==0 else [1,0]):
            p=processes[index];p.stdin.write(json.dumps(dict(workload=['BBBD0',1336,64,131],backend='cpu-fast'))+'\n');p.stdin.flush()
            result=json.loads(p.stdout.readline())
            if reference is None:reference=result
            passed=result['bounds_sha256']==reference['bounds_sha256'] and result['nstates']==reference['nstates'] and all(np.allclose(result[k],reference[k],atol=1e-8,rtol=1e-8,equal_nan=True) for k in ('tdists','patterns','strengths'))
            assert passed
            rows.append(dict(revision='accepted-9309874' if index==0 else 'final',iteration=iteration,warmup=iteration<2,wall_s=result['wall_s'],passed=bool(passed),input_sha256=result['input_sha256']))
            print(rows[-1],flush=True)
finally:
    for p in processes:p.stdin.close();p.wait()
(base/'cpu_focused_recheck.json').write_text(json.dumps(rows,indent=2))
(base/'cpu_focused_reproduce.py').write_bytes(Path(__file__).read_bytes())
