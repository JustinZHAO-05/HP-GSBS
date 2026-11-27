"""Reproduce the v0.2 timing table and speedup figure from archived, unrounded receipts.

No fitting is performed. Requires matplotlib and numpy. Existing v0.1 assets
are deliberately left intact. --check verifies the derived table/JSON only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/v02/merge_9baff92/revisions/trials.json'
WORKLOADS=['Small','Medium','Large','XL','BBBD0']
BACKENDS=['cpu-fast','torch-cuda','triton-cuda']
REVISIONS=['v0.1.0','accepted-9309874','final']


def aggregate():
    rows=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert len(rows)==225 and all(r['passed'] for r in rows)
    cells=[]
    for w in WORKLOADS:
        assert len({r['input_sha256'] for r in rows if r['workload']==w})==1
        for b in BACKENDS:
            versions={}
            for rev in REVISIONS:
                subset=[r for r in rows if (r['workload'],r['backend'],r['revision'])==(w,b,rev)]
                assert sorted(r['iteration'] for r in subset)==list(range(5))
                assert all(r['warmup']==(r['iteration']<2) for r in subset)
                times=[r['wall_s'] for r in subset if not r['warmup']]
                assert len(times)==3 and all(t>0 for t in times)
                versions[rev]=dict(median_s=median(times),min_s=min(times),max_s=max(times),trials_s=times)
            cells.append(dict(workload=w,backend=b,versions=versions,
                              speedup=versions['v0.1.0']['median_s']/versions['final']['median_s']))
    return dict(source=SOURCE.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                measured_revision='9baff923fea285d448fb3078c82943f365369b3b',
                protocol='Three interleaved source snapshots; 2 warmups + 3 timed fits each; median and min-max; validation excluded',
                cells=cells)


def table(data):
    lines=['| Workload | Backend | v0.1.0 (s) | v0.2 snapshot (s) | v0.2 range (s) | Speedup |',
           '|---|---|---:|---:|---:|---:|']
    for r in data['cells']:
        old=r['versions']['v0.1.0'];new=r['versions']['final']
        label={'cpu-fast':'CPU','torch-cuda':'PyTorch','triton-cuda':'Triton'}[r['backend']]
        lines.append(f"| {r['workload']} | {label} | {old['median_s']:.4f} | {new['median_s']:.4f} | {new['min_s']:.4f}–{new['max_s']:.4f} | {r['speedup']:.2f}× |")
    return '\n'.join(lines)+'\n'


def write(path,text):
    with path.open('w',encoding='utf-8',newline='\n') as f:f.write(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    data=aggregate();tab=table(data)
    summary=ROOT/'docs/assets/performance-v0.2.0.json';table_path=ROOT/'docs/v02_performance_table.md'
    if args.check:
        assert json.loads(summary.read_text(encoding='utf-8'))==data
        assert table_path.read_text(encoding='utf-8')==tab
        print("15 rows and 45 snapshot cells match the unrounded receipts.")
        return
    write(summary, json.dumps(data, indent=2) + "\n")
    write(table_path, tab)
    plot(data)


def plot(data):
    from matplotlib.ticker import FuncFormatter
    from _performance_plot import SERIES, save, setup

    fig, ax = setup("hpgsbs-v020")
    ax.set_xlim(.8, 6.45)
    ax.set_xticks(range(1, 7))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}×"))
    ax.axvline(1, color="#697788", linewidth=1.1, linestyle="--")
    ax.set_xlabel("Speedup over v0.1.0 (ratio of median fit times)", labelpad=8)
    for backend, offset in zip(BACKENDS, (-.22, 0, .22)):
        cells = [cell for cell in data["cells"] if cell["backend"] == backend]
        y = [index + offset for index in range(len(cells))]
        label, color, marker = SERIES[backend]
        ax.scatter([cell["speedup"] for cell in cells], y, label=label,
                   color=color, marker=marker, s=40, zorder=3)
    save(fig, ax, ROOT / "docs/assets/performance-v0.2.0", len(BACKENDS))



if __name__=='__main__':main()
