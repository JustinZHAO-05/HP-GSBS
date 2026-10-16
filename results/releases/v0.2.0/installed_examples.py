import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',default='cpu-fast');p.add_argument('--example',required=True);a=p.parse_args()
out=Path(a.output).resolve();assert not out.exists(),out
subprocess.run([sys.executable,a.example,'--backend',a.backend,'--output',str(out)],check=True)
task=out/'tasks.json';task.write_text(json.dumps([dict(id='synthetic',path=str(out/'input.npy'),config=dict(kmax=12,trace=True))]))
command=[sys.executable,'-m','hpgsbs','fit-batch',str(task),'--output',str(out/'batch'),'--backend',a.backend]
subprocess.run(command,check=True);subprocess.run(command,check=True)
rows=[json.loads(s) for s in (out/'batch'/'manifest.jsonl').read_text().splitlines()]
assert [r['status'] for r in rows]==['success','skipped'],rows
print('Installed quickstart, CLI fit-batch and CLI resume passed')
