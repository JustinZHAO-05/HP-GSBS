"""Verify explicitly selected evidence; never reinterpret historical acceptance."""
import argparse,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
    manifest=Path(a.manifest);base=manifest.parent;m=json.loads(manifest.read_text())
    for name,digest in m['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    for name,item in m['files'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==item['sha256'],name
    checks={}
    for platform in ('windows','wsl'):
        suites=ET.parse(base/f'tests_{platform}.xml').getroot().iter('testsuite')
        counts={k:0 for k in ('tests','failures','errors','skipped')}
        for suite in suites:
            for k in counts:counts[k]+=int(suite.attrib.get(k,0))
        assert counts['tests']>0 and counts['failures']==counts['errors']==0
        checks[platform]=counts
    real=json.loads((base/'real/validation.json').read_text())
    assert len(real)>=16 and all(r['report']['passed'] for r in real)
    assert all(r['report']['moves_compared'] for r in real if r.get('scope')=='fresh-official-dev96-every-move')
    pairs=json.loads((base/'revisions/trials.json').read_text())
    assert len(pairs)==225 and all(r['passed'] for r in pairs)
    long=json.loads((base/'long/validation.json').read_text())
    assert len(long)==4 and all(r['checks'] and all(c['passed'] for c in r['checks']) for r in long)
    throughput_path=base/'throughput_fit_only/throughput.json'
    fit_only=throughput_path.exists()
    if not fit_only:throughput_path=base/'throughput/throughput.json'
    throughput=json.loads(throughput_path.read_text())
    assert all(r['all_outputs_passed'] for r in throughput if 'all_outputs_passed' in r)
    assert all(all(s=='success' for s in r['statuses']) and all(s=='skipped' for s in r['resume_statuses']) for r in throughput if r.get('kind')=='manifest')
    if fit_only:
        extra=json.loads((base/'final_checks.json').read_text())
        assert extra['passed'] and len(extra['throughput_coverage'])==54
        assert all(not r['validation_in_timed_scope'] for r in throughput if 'iteration' in r)
    assert not list(base.rglob('*.npy')) and not list(base.rglob('*.npz'))
    print(json.dumps(dict(passed=True,source_revision=m['source_revision'],files=len(m['files']),
        tests=checks,real_comparisons=len(real),revision_comparisons=len(pairs),long_shapes=len(long),
        throughput_scope='fit-only' if fit_only else 'validation-inclusive',
        throughput_record_rows=len(throughput),throughput_trials=sum('iteration' in r for r in throughput),
        throughput_memory_skips=sum(r.get('status')=='skipped-memory-admission' for r in throughput),
        long_full_official_trace_certified=False),indent=2))


if __name__=='__main__':main()
