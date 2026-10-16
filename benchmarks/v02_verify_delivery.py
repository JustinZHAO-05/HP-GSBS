"""Verify explicitly selected evidence; never reinterpret historical acceptance."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--source-revision',help='Verify source hashes against this immutable Git commit, not the current workspace')
    a=p.parse_args()
    manifest=Path(a.manifest);base=manifest.parent;m=json.loads(manifest.read_text())
    revision=None
    if a.source_revision:
        revision=subprocess.check_output(['git','rev-parse','--verify',a.source_revision+'^{commit}'],text=True).strip()
    for name,digest in m['source_hashes'].items():
        raw=subprocess.check_output(['git','show',revision+':'+name]) if revision else Path(name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest,name
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
    if m.get('scope')=='scheduler-followup':
        baseline=Path(m['baseline_manifest']['path'])
        assert hashlib.sha256(baseline.read_bytes()).hexdigest()==m['baseline_manifest']['sha256']
        previous=json.loads(baseline.read_text())
        for name,item in previous['files'].items():
            assert hashlib.sha256((baseline.parent/name).read_bytes()).hexdigest()==item['sha256'],name
        changed=sorted(name for name,digest in previous['source_hashes'].items()
                       if name.startswith('src/') and m['source_hashes'][name]!=digest)
        assert changed==['src/hpgsbs/batch/bucketing.py','src/hpgsbs/batch/scheduler.py'],changed
        required={'test_manifest_resume_validation':24,'test_execution_signature_defaults_version_and_policy':1,
                  'test_gpu_auto_batch_mixed_policies':2}
        for platform in ('windows','wsl'):
            cases=list(ET.parse(base/f'tests_{platform}.xml').getroot().iter('testcase'))
            expected=dict(required)
            if platform=='wsl': expected['test_auto_batch_mixed_resource_policies']=4
            for name,count in expected.items():
                passed=[c for c in cases if c.attrib['name'].split('[')[0]==name
                        and all(c.find(tag) is None for tag in ('failure','error','skipped'))]
                assert len(passed)==count,(platform,name,len(passed))
        assert not list(base.rglob('*.npy')) and not list(base.rglob('*.npz'))
        print(json.dumps(dict(passed=True,scope=m['scope'],source_revision=m['source_revision'],verified_source_commit=revision,
                              files=len(m['files']),tests=checks,changed_library_sources=changed,
                              historical_performance_rerun=False),indent=2))
        return
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
    print(json.dumps(dict(passed=True,source_revision=m['source_revision'],verified_source_commit=revision,files=len(m['files']),
        tests=checks,real_comparisons=len(real),revision_comparisons=len(pairs),long_shapes=len(long),
        throughput_scope='fit-only' if fit_only else 'validation-inclusive',
        throughput_record_rows=len(throughput),throughput_trials=sum('iteration' in r for r in throughput),
        throughput_memory_skips=sum(r.get('status')=='skipped-memory-admission' for r in throughput),
        long_full_official_trace_certified=False),indent=2))


if __name__=='__main__':main()
