"""Manifest public copies after archiving originals; never alter numbers/history."""
import argparse,hashlib,json,re,shutil,platform
from importlib.metadata import version
from pathlib import Path


def sources():
    files=[f for base in ('src','tests','benchmarks') for f in Path(base).rglob('*.py')]
    files += [Path('pyproject.toml'),Path('CITATION.cff')]
    return {f.as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def public_text(text):
    text=re.sub(r'(?<![\w])[A-Za-z]:[\\/][^\s"<>]*','[local-path]',text)
    text=re.sub(r'/(?:home|mnt|tmp|usr)/[^\s"<>]*','[system-path]',text)
    text=re.sub(r'hostname="[^"]*"','hostname="redacted"',text)
    return text


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--private-archive',required=True);p.add_argument('--initial',action='store_true')
    a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True)
    if a.initial:
        if (out/'source.json').exists():raise FileExistsError(out/'source.json')
        (out/'source.json').write_text(json.dumps(dict(revision=a.revision,source_hashes=sources(),
            versions={name:version(name) for name in ('numpy','scipy','numba','statesegmentation','pytest')},
            python=platform.python_version(),platform=platform.system(),
            protocol='FP64; TF32 disabled; BLAS/Numba 1; CUDA synchronized; warm interleaved trials; no simultaneous benchmarks'),indent=2));return
    source=json.loads((out/'source.json').read_text());assert source['source_hashes']==sources(),'Source changed since acceptance began'
    archive=Path(a.private_archive)
    if archive.exists():raise FileExistsError(archive)
    archive.mkdir(parents=True);records={}
    for f in out.rglob('*'):
        if not f.is_file() or f.name=='manifest.json':continue
        relative=f.relative_to(out);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,target);raw=f.read_bytes();text=raw.decode('utf-8');safe=public_text(text)
        f.write_text(safe,encoding='utf-8')
        records[relative.as_posix()]=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            original_sha256=hashlib.sha256(raw).hexdigest(),sanitized=safe!=text)
    manifest=dict(source_revision=a.revision,source_hashes=source['source_hashes'],files=records,
        old_evidence_untouched=True,raw_arrays_published=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
