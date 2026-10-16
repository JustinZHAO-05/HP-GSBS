"""Sequential acceptance runner; failures persist and stop dependent work."""
import argparse,json,subprocess,sys
from pathlib import Path
from time import perf_counter


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--output',required=True)
    p.add_argument('--revision',required=True);a=p.parse_args();out=Path(a.output)
    assert (out/'source.json').exists(),'Initialize source manifest first'
    from v02_evidence import sources
    expected=json.loads((out/'source.json').read_text())
    assert expected['revision']==a.revision and expected['source_hashes']==sources()
    root=str(Path('.').resolve());snapshots=[str(Path('tmp/baselines/v01').resolve()),str(Path('tmp/baselines/accepted').resolve()),root]
    jobs=[('tests',['-m','pytest','tests','-q','--tb=short',f'--junitxml={out}/tests_wsl.xml']),
        ('real',['benchmarks/v02_validate.py','--fixtures',a.fixtures,'--output',str(out/'real'),'--backends','cpu-fast','torch-cuda','triton-cuda']),
        ('revisions',['benchmarks/v02_revision_compare.py','--fixtures',a.fixtures,'--output',str(out/'revisions'),'--snapshots',*snapshots]),
        ('resident_pair',['benchmarks/v02_paired.py','--fixtures',a.fixtures,'--output',str(out/'resident_pair'),'--backends','torch-cuda','triton-cuda','--modes','streaming','triton-resident']),
        ('long',['benchmarks/v02_validate.py','--long','--output',str(out/'long')]),
        ('research',['benchmarks/v02_research.py','--fixtures',a.fixtures,'--output',str(out/'research')]),
        ('resident_component',['benchmarks/v02_resident.py','--output',str(out/'resident_component.json')]),
        ('throughput',['benchmarks/v02_throughput_extended.py','--fixtures',a.fixtures,'--output',str(out/'throughput')]),
        ('cold',['benchmarks/v02_cold.py','--output',str(out/'cold.json')]),
        ('memory',['benchmarks/v02_memory.py','--output',str(out/'memory.json')]),
        ('precision',['benchmarks/v02_precision.py','--fixtures',a.fixtures,'--output',str(out/'precision.json')]),
        ('ablation',['benchmarks/v02_ablation.py','--fixtures',a.fixtures,'--output',str(out/'ablation.json')]),
        ('components',['benchmarks/v02_components.py','--output',str(out/'components.json')]),
        ('profile',['benchmarks/v02_profile.py','--fixtures',a.fixtures,'--output',str(out/'profile')])]
    receipts=[]
    for name,args in jobs:
        assert expected['source_hashes']==sources(),'Source changed during acceptance'
        start=perf_counter();print('START',name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            process=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
        receipts.append(dict(job=name,returncode=process.returncode,wall_s=perf_counter()-start))
        (out/'jobs.json').write_text(json.dumps(receipts,indent=2));print('END',receipts[-1],flush=True)
        if process.returncode:raise RuntimeError('Failed acceptance job: '+name)


if __name__=='__main__':main()
