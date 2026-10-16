"""Run with a non-editable, installed release package from outside the repository."""
import argparse,hashlib,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import hpgsbs
from hpgsbs import fit,GSBSConfig,GSBSResult,fit_many
from hpgsbs.validation import compare
from hpgsbs.batch.scheduler import run_manifest,_task_key,_stamp_execution

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--fixtures');p.add_argument('--legacy-root');a=p.parse_args()
assert hpgsbs.__version__=='0.2.0'
assert 'site-packages' in hpgsbs.__file__,hpgsbs.__file__
if not a.gpu:assert importlib.util.find_spec('torch') is None and importlib.util.find_spec('triton') is None
x=np.random.default_rng(20261016).normal(size=(36,7));c=GSBSConfig(8,trace=True)
ref=fit(x,c,'reference');results=[]
for backend in (['cpu-fast','torch-cuda','triton-cuda'] if a.gpu else ['cpu-fast']):
    r=fit(x,c,backend);report=compare(ref,r,level='finetune-move-trace');assert report['passed'],report
    batch=fit_many([x,x.copy()],c,backend,batch_size=2)
    assert all(compare(ref,b,level='finetune-move-trace')['passed'] for b in batch)
    results.append(dict(backend=backend,strict_move_parity=True,batch_parity=True))
with tempfile.TemporaryDirectory() as scratch:
    scratch=Path(scratch);out=scratch/'fit.npz';r.save(out);loaded=GSBSResult.load(out)
    assert compare(r,loaded,level='output-only')['passed']
    input_path=scratch/'x.npy';np.save(input_path,x)
    tasks=[dict(id='rec',path=str(input_path),config=dict(kmax=8,trace=True))]
    for backend in (['cpu-fast','torch-cuda','triton-cuda'] if a.gpu else ['cpu-fast']):
        dest=scratch/backend
        assert run_manifest(tasks,dest,backend)[0]['status']=='success'
        assert run_manifest(tasks,dest,backend)[0]['status']=='skipped'
        task=tasks[0]; saved=GSBSResult.load(dest/'rec.npz')
        assert saved.metadata['execution_identity']['software_version']=='0.2.0'
        for status in ('failed','cancelled',None):
            d=scratch/(backend+'-'+str(status)); d.mkdir()
            if status is None:saved.metadata.pop('status',None)
            else:saved.metadata['status']=status
            saved.save(d/'rec.npz')
            assert run_manifest(tasks,d,backend)[0]['status']=='failed'
legacy={}
if a.legacy_root:
    base=Path(a.legacy_root);tasks=json.loads((base/'legacy_task.json').read_text())
    for name in ('legacy_cache','dev_cache'):
        path=base/name/'record.npz';before=hashlib.sha256(path.read_bytes()).hexdigest()
        old=GSBSResult.load(path);assert compare(old,fit(old.x,c),level='output-only')['passed']
        receipt=run_manifest(tasks,base/name)[0]
        assert receipt['status']=='failed',receipt
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        legacy[name]=dict(load_and_output_parity=True,resume_refused=True,bytes_preserved=True)
real=[]
if a.fixtures:
    for index in range(4):
        frozen=GSBSResult.load(Path(a.fixtures)/f'{index}_full_cpu.npz');cfg=GSBSConfig(**frozen.metadata['algorithm'],trace=True)
        for backend in (['cpu-fast','torch-cuda','triton-cuda'] if a.gpu else ['cpu-fast']):
            report=compare(frozen,fit(frozen.x,cfg,backend));assert report['passed'],report
            real.append(dict(index=index,backend=backend,passed=True,scope='full frozen output/path; not newly generated official moves'))
doctor=subprocess.run([sys.executable,'-m','hpgsbs','doctor'],check=True,capture_output=True,text=True)
report=dict(version=hpgsbs.__version__,installed_package=True,cpu_only_without_torch=not a.gpu,python=sys.version.split()[0],checks=results,save_load=True,manifest_resume=True,non_success_resume_refused=True,legacy=legacy,real=real,doctor_ok=True)
Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
