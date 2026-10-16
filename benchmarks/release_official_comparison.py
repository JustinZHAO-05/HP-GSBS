"""Remeasure the installed v0.2.0 wheel against retained official timings.

This is a cross-session comparison, not a fresh paired official benchmark.
Large official results and EEG inputs are private read-only local fixtures.
No arrays, private paths or participant data are written to the public receipt.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import hpgsbs
from hpgsbs import GSBSConfig, GSBSResult, fit
from hpgsbs.doctor import doctor
from hpgsbs.validation import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--official-receipts', required=True)
    p.add_argument('--official-environment', required=True)
    p.add_argument('--official-results', required=True)
    p.add_argument('--eeg-input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert hpgsbs.__version__ == '0.2.0' and 'site-packages' in hpgsbs.__file__
    import torch
    assert torch.cuda.is_available()
    source = Path(a.official_receipts).read_bytes()
    official = {r['workload']: r for r in json.loads(source) if r['backend'] == 'reference'}
    old_env = json.loads(Path(a.official_environment).read_text())['environment']
    env = doctor(); env.pop('executable', None)
    for key in ('os', 'python', 'cpu_model', 'packages', 'cuda_runtime', 'gpu'):
        assert env[key] == old_env[key], (key, env[key], old_env[key])
    run = dict(version=hpgsbs.__version__, release_commit='b86b409169aa173d4a361d8cbcc633add1bfacf6',
               release_wheel_sha256='43f1819b7d8c31adea736345210550b0a5aae4a247e77e363c6650b1b5db749b',
               started_at=datetime.now(timezone.utc).isoformat(), environment=env,
               official_receipt_sha256=hashlib.sha256(source).hexdigest(),
               protocol='Historical official times / newly measured release-wheel times; cross-session, not paired. FP64; TF32 off; 1 BLAS/Numba thread; trace=False; CUDA synchronized; validation and saving excluded. Synthetic: one warmup and three timed fits. BBBD0: one timed fit per backend, no workload-specific warmup or variability estimate; process already initialized.',
               trials=[], workloads=[])
    def save(): out.write_text(json.dumps(run, indent=2)+'\n', encoding='utf-8')
    save()
    for name, T, P, K in [('Small',450,16,20),('Medium',900,32,30),('Large',1500,64,40),('XL',2100,64,50),('BBBD0',1336,64,131)]:
        x = np.load(a.eeg_input, allow_pickle=False).astype(np.float64) if name == 'BBBD0' else np.random.default_rng(20261016).normal(size=(T,P))
        c = GSBSConfig(K, dmin=5 if name == 'BBBD0' else 1)
        old = official[name]; fingerprint = hashlib.sha256(x.tobytes()).hexdigest()
        assert x.shape == (T,P) and c.algorithm() == old['algorithm']
        if name == 'Small':
            # The earliest Small receipt has no input hash. Preserve that limit:
            # reconstruct its documented seed/shape and obtain a fresh official
            # result for validation only, outside the plotted timing protocol.
            print('Fresh official Small validation (not plotted timing)', flush=True)
            reference = fit(x,c,'reference')
            identity = 'reconstructed from original delivery_bench.py seed=20261016 and shape; historical Small receipt has no input hash'
        else:
            assert fingerprint == old['input_sha256']
            reference = GSBSResult.load(Path(a.official_results)/f'completed_{name}_reference.npz')
            assert np.array_equal(reference.x,x)
            assert reference.metadata['algorithm'] == c.algorithm()
            identity = 'historical receipt input SHA256 and saved official input verified'
        run['workloads'].append(dict(name=name,shape=[T,P],algorithm=c.algorithm(),compute_config=asdict(c),input_sha256=fingerprint,input_identity=identity))
        for iteration in range(1 if name == 'BBBD0' else 4):
            for backend in ('cpu-fast','torch-cuda','triton-cuda'):
                gpu=backend.endswith('cuda')
                if gpu: torch.cuda.synchronize()
                start=perf_counter();result=fit(x,c,backend)
                if gpu: torch.cuda.synchronize()
                elapsed=perf_counter()-start
                validation=compare(reference,result)
                assert validation['passed'],(name,backend,validation)
                run['trials'].append(dict(workload=name,backend=backend,iteration=iteration,
                    warmup=name!='BBBD0' and iteration==0,wall_s=elapsed,
                    input_sha256=fingerprint,validation=validation))
                save(); print(name,backend,iteration,f'{elapsed:.6f}s','parity=True',flush=True)
    run['completed_at']=datetime.now(timezone.utc).isoformat();save()


if __name__ == '__main__': main()
