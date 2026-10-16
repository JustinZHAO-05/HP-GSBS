"""Additional rare-parameter traces and exact throughput input identities."""
import json,hashlib,os
from pathlib import Path
import numpy as np
from hpgsbs import fit,GSBSConfig,GSBSResult
from hpgsbs.validation import compare
base=Path('results/v02/merge_9baff92');rows=[]
assert not (base/'supplemental_trace_checks.json').exists(),'Preserve previous evidence'
(base/'supplemental_reproduce.py').write_bytes(Path(__file__).read_bytes())
patterns=np.array([[1.,-1,0,0],[0,0,1.,-1],[-1.,1,0,0],[0,0,-1.,1]])
for kind,x in [('equal-nonzero-strength',np.repeat(patterns,8,axis=0)),
               ('repeated-cancellation',np.tile(np.array([[1.,2,3],[-1.,-2,-3]]),(12,1)))]:
    for fine in (0,1,2,-1):
        for order in (False,True):
            c=GSBSConfig(11,trace=True,finetune=fine,finetune_order=order,blocksize=5,dmin=5)
            report=compare(fit(x,c,'reference'),fit(x,c),level='finetune-move-trace')
            rows.append(dict(kind=kind,fine=fine,order=order,report=report))
            assert report['passed'],rows[-1]
(base/'supplemental_trace_checks.json').write_text(json.dumps(rows,indent=2))
inputs={}
for workload in ('Small','highK','variable'):
    values=[]
    for i in range(64):
        if workload=='highK':x=np.load(Path(os.environ.get('HPGSBS_FIXTURES','results/fixtures'))/'0.npy').astype(float)
        else:
            shape=(450,16) if workload=='Small' else (450+20*(i%3),16+4*(i%2))
            x=np.random.default_rng(200+i).normal(size=shape)
        values.append(dict(index=i,shape=list(x.shape),input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    inputs[workload]=dict(inputs=values,kmax=131 if workload=='highK' else 20,
        dmin=5 if workload=='highK' else 1,statewise=True,finetune=1,blocksize=50,
        precision='float64',highK_is_repeated_same_EEG=workload=='highK')
(base/'throughput_inputs.json').write_text(json.dumps(inputs,indent=2))
identities=[]
for t,p in [(450,16),(900,32),(1500,64),(2100,64),(1336,64),(5000,8),(10000,16),(20000,32),(50000,64)]:
    x=np.random.default_rng(20261016).normal(size=(t,p))
    identities.append(dict(kind='synthetic-seed-20261016',shape=[t,p],input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
fixtures=Path(os.environ.get('HPGSBS_FIXTURES','results/fixtures'))
for index in range(4):
    for scope in ('full_cpu','dev96_reference'):
        r=GSBSResult.load(fixtures/f'{index}_{scope}.npz')
        identities.append(dict(kind=scope,record=index,shape=list(r.x.shape),input_sha256=hashlib.sha256(r.x.tobytes()).hexdigest(),algorithm=r.metadata['algorithm']))
(base/'workload_identities.json').write_text(json.dumps(identities,indent=2))
print('16 additional strict per-move comparisons passed; 192 throughput input identities recorded')
