"""Plot retained official GSBS timings versus measured v0.2.0 wheel timings.

No fitting. --check verifies the derived data and detailed timing table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/releases/v0.2.0/official_comparison.json'
OLD=ROOT/'results/benchmarks/delivery_latency.json'
ASSETS=ROOT/'docs/assets'
NAMES=['Small','Medium','Large','XL','BBBD0']
BACKENDS=['reference','cpu-fast','torch-cuda','triton-cuda']


def derive():
    raw=json.loads(RAW.read_text()); old=json.loads(OLD.read_text())
    assert raw['completed_at'] and raw['version']=='0.2.0' and len(raw['trials'])==51
    assert raw['official_receipt_sha256']==hashlib.sha256(OLD.read_bytes()).hexdigest()
    assert all(t['validation']['passed'] for t in raw['trials'])
    cells=[]
    for name in NAMES:
        cfg=next(w for w in raw['workloads'] if w['name']==name)
        ref=next(r for r in old if r['workload']==name and r['backend']=='reference')
        assert ref['algorithm']==cfg['algorithm'] and ref['shape']==cfg['shape']
        if name!='Small': assert ref['input_sha256']==cfg['input_sha256']
        ref_trials=ref['trials'] if name=='BBBD0' else ref['trials'][1:]
        values=[t['wall_s'] for t in ref_trials]
        ref_value=float(np.median(values))
        assert ref_value==ref.get('single_fit_s',ref.get('steady_median_s'))
        for backend in BACKENDS:
            trials=ref_trials if backend=='reference' else [t for t in raw['trials'] if t['workload']==name and t['backend']==backend and not t['warmup']]
            times=[t['wall_s'] for t in trials]
            assert len(times)==(1 if name=='BBBD0' else 3)
            value=float(np.median(times))
            cells.append(dict(workload=name,backend=backend,shape=cfg['shape'],kmax=cfg['algorithm']['kmax'],seconds=value,min_s=min(times),max_s=max(times),timed_fits=len(times),speedup=ref_value/value,source='historical official baseline' if backend=='reference' else 'published v0.2.0 wheel remeasurement'))
    report=dict(comparison='historical official / newly measured v0.2.0; cross-session, not paired',release_commit=raw['release_commit'],wheel_sha256=raw['release_wheel_sha256'],official_receipt_sha256=raw['official_receipt_sha256'],release_receipt_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),cells=cells)
    rows=['| Workload · T × P · Kmax | Official GSBS (s) | v0.2 CPU (s; speedup) | v0.2 PyTorch (s; speedup) | v0.2 Triton (s; speedup) |','|---|---:|---:|---:|---:|']
    for name in NAMES:
        group=[c for c in cells if c['workload']==name];ref=group[0];T,P=ref['shape']
        rows.append(f"| {name} · {T}×{P} · {ref['kmax']} | {ref['seconds']:,.3f} | "+' | '.join(f"{c['seconds']:.3f}; **{c['speedup']:,.1f}×**" for c in group[1:])+' |')
    return report,'\n'.join(rows)+'\n'


def plot(report):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from _performance_plot import SERIES, save, setup

    fig, ax = setup("hpgsbs-official-v020")
    ax.set_xscale("log")
    ax.set_xlim(.04, 13000)
    ax.xaxis.set_major_locator(FixedLocator([.1, 1, 10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Complete fit (seconds, log scale)", labelpad=8)
    for backend, offset in zip(BACKENDS, (-.30, -.10, .10, .30)):
        cells = [cell for cell in report["cells"] if cell["backend"] == backend]
        y = np.arange(len(cells)) + offset
        label, color, marker = SERIES[backend]
        ax.hlines(y, [cell["min_s"] for cell in cells],
                  [cell["max_s"] for cell in cells], color=color, linewidth=1.7)
        ax.scatter([cell["seconds"] for cell in cells], y, label=label,
                   color=color, marker=marker, s=34, zorder=3)
    save(fig, ax, ASSETS / "performance-official-v0.2.0", len(BACKENDS))


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    report,table=derive();serialized=json.dumps(report,indent=2)+'\n'
    json_path=ASSETS/'performance-official-v0.2.0.json';table_path=ROOT/'docs/v020_official_table.md'
    if a.check:
        assert json_path.read_text(encoding='utf-8')==serialized
        assert table_path.read_text(encoding='utf-8')==table
    else:
        json_path.write_text(serialized,encoding='utf-8',newline='\n');table_path.write_text(table,encoding='utf-8',newline='\n');plot(report)
    print('20 official/release timing cells match the measured receipts')


if __name__=='__main__':main()
