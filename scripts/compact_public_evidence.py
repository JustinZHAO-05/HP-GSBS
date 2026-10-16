"""Compact already archived public copies, then audit newly public text.

Retains all scalar timing/parity/error results. Dense output patterns stay in
the private archive, represented publicly by hashes and comparison errors.
Must run AFTER benchmarks/v02_evidence.py has archived and sanitized originals.
"""
import argparse,hashlib,json,math,re,subprocess
from pathlib import Path


def digest(data):return hashlib.sha256(data).hexdigest()


def flatten(values):
    for value in values:
        if isinstance(value,list):yield from flatten(value)
        else:yield value


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--private-archive',required=True)
    a=p.parse_args();base=Path(a.directory).resolve();archive=Path(a.private_archive).resolve()
    m=json.loads((base/'manifest.json').read_text());changes=[];archived={}
    def path(name):
        target=(base/name).resolve()
        if not target.is_relative_to(base):raise ValueError('Unsafe evidence path')
        return target
    for name,item in m['files'].items():
        assert digest(path(name).read_bytes())==item['sha256'],name
        assert digest((archive/name).read_bytes())==item['original_sha256'],name
    name='revisions/trials.json';rows=json.loads(path(name).read_text());references={}
    for row in rows:
        key=row['workload'],row['backend']
        reference=references.setdefault(key,{k:row[k] for k in ('tdists','patterns','strengths')})
        assert row['passed']
        for field in ('tdists','patterns','strengths'):
            values=row.pop(field)
            row[field+'_sha256']=digest(json.dumps(values,separators=(',',':')).encode())
            errors=[abs(x-y) for x,y in zip(flatten(values),flatten(reference[field])) if math.isfinite(x) and math.isfinite(y)]
            row[field+'_max_abs_error']=max(errors,default=0.)
    path(name).write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8');changes.append(name)
    for name in list(m['files']):
        if name.startswith('profile/') and name.endswith('.json') and not name.endswith('_metadata.json'):
            archived[name]=m['files'].pop(name);path(name).unlink()
    for name in m['files']:
        f=path(name)
        if f.suffix=='.json':
            f.write_text(json.dumps(json.loads(f.read_text()),separators=(',',':')),encoding='utf-8')
        m['files'][name]['sha256']=digest(f.read_bytes())
    receipt=dict(transformations=changes,archived_chrome_traces=archived,
        unchanged='all scalar numerical parity/timing values',
        removed_public_arrays='tdists/patterns/strengths repeated per revision trial; original hashes verified in private archive',
        public_replacements='per-array hashes and maximum absolute comparison errors',
        private_archive_directory_name=archive.name)
    f=path('public_compaction.json');f.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    m['files']['public_compaction.json']=dict(sha256=digest(f.read_bytes()),sanitized=True)
    m['archived_files']=archived
    own=Path(__file__).resolve();relative=own.relative_to(Path.cwd().resolve()).as_posix()
    m['source_hashes'][relative]=digest(own.read_bytes())
    (base/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    tracked=subprocess.check_output(['git','diff','--name-only','origin/main'],text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
    patterns={
        'machine_path':r'(?i)(?:(?<![\w])[A-Z]:[\\/]|/(?:home|mnt|tmp)/)[^\s"<>]+',
        'private_identity':r'(?i)(?:wxid_[a-z0-9_]+|Users[\\/]+[0-9]{4,})',
        'credential':r'(?:gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{25,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)'}
    findings=[];count=0
    for name in sorted(set(tracked+untracked)):
        f=Path(name)
        if not f.is_file():continue
        if f.suffix.lower() in ('.npy','.npz','.env'):findings.append(dict(file=name,kind='private-binary'));continue
        try:text=f.read_text(encoding='utf-8')
        except UnicodeDecodeError:continue
        count+=1
        for kind,pattern in patterns.items():
            if re.search(pattern,text):findings.append(dict(file=name,kind=kind))
    report=dict(scanned_new_or_changed_text_files=count,findings=findings,
        scope='current public tree relative to origin/main; historical Git not rewritten',
        raw_arrays_in_acceptance=bool(list(base.rglob('*.npy'))+list(base.rglob('*.npz'))))
    print(json.dumps(report,indent=2))
    if findings or report['raw_arrays_in_acceptance']:raise SystemExit(1)


if __name__=='__main__':main()
