"""Reproduce v0.2 README tables and PNG/SVG from archived, unrounded receipts.

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
LABELS=['CPU FastGSBS','TensorGSBS / PyTorch','TurboGSBS / Triton']
COLORS=['#D57927','#008F97','#7154C4']


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
        for name in ('README.md','README.zh-CN.md'):
            content=(ROOT/name).read_text(encoding='utf-8')
            assert content.split('<!-- V02_TABLE_START -->\n')[1].split('<!-- V02_TABLE_END -->')[0]==tab+'\n'
        print('15 rows, 45 snapshot cells and both README tables match the unrounded receipts.')
        return
    write(summary,json.dumps(data,indent=2)+'\n');write(table_path,tab)
    for name in ('README.md','README.zh-CN.md'):
        path=ROOT/name;content=path.read_text(encoding='utf-8')
        if '<!-- V02_TABLE_START -->' in content:
            before=content.split('<!-- V02_TABLE_START -->')[0]
            after=content.split('<!-- V02_TABLE_END -->')[1]
            write(path,before+'<!-- V02_TABLE_START -->\n'+tab+'\n<!-- V02_TABLE_END -->'+after)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator,FuncFormatter
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'hpgsbs-v020'})
    fig,axes=plt.subplots(2,3,figsize=(16,10),gridspec_kw={'height_ratios':[1.2,1]})
    fig.patch.set_facecolor('#F5F7FA')
    fig.subplots_adjust(left=.08,right=.97,bottom=.22,top=.79,wspace=.32,hspace=.42)
    for j,(backend,label,color) in enumerate(zip(BACKENDS,LABELS,COLORS)):
        a,b=axes[:,j]
        selected=[r for r in data['cells'] if r['backend']==backend]
        y=np.arange(5)
        for ax in (a,b):
            ax.set_facecolor('#F5F7FA');ax.set_ylim(4.65,-.65);ax.set_yticks(y,WORKLOADS)
            ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#CBD3DE')
            ax.tick_params(length=0,pad=7,colors='#536178');ax.grid(axis='x',color='#DEE4ED',lw=.7);ax.set_axisbelow(True)
        a.set_xscale('log');a.set_xlim(.035,8)
        a.xaxis.set_major_locator(FixedLocator([.05,.1,.3,1,3]))
        a.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        a.xaxis.set_minor_locator(FixedLocator([]))
        a.set_title(label,loc='left',color=color,weight='bold',fontsize=15,pad=15)
        a.set_xlabel('Fit wall time (seconds, log scale)',labelpad=10)
        b.set_xlim(0,7);b.set_xticks([0,1,2,3,4,5,6]);b.axvline(1,color='#9AA5B5',lw=1,ls='--')
        b.set_xlabel('v0.1 median / v0.2 median (×)',labelpad=10)
        for i,r in enumerate(selected):
            for rev,offset,fill,c in [('v0.1.0',-.14,'white','#8591A3'),('final',.14,color,color)]:
                v=r['versions'][rev];a.plot([v['min_s'],v['max_s']],[i+offset]*2,color=c,lw=2)
                a.scatter(v['median_s'],i+offset,s=52,facecolor=fill,edgecolor=c,lw=1.5,zorder=3)
                a.annotate(f"{v['median_s']:.3f}",(v['max_s'],i+offset),xytext=(7,0),textcoords='offset points',va='center',fontsize=9,color=c)
            b.barh(i,r['speedup'],height=.48,color=color)
            b.text(r['speedup']+.1,i,f"{r['speedup']:.2f}×",va='center',weight='bold',color=color)
    fig.text(.055,.95,'HP-GSBS v0.2',fontsize=29,weight='bold',color='#142A43')
    fig.text(.055,.91,'Streaming search: measured changes from v0.1',fontsize=18,color='#243B53')
    handles=[Line2D([0],[0],marker='o',color='#8591A3',markerfacecolor='white',linestyle='',label='v0.1.0'),
             Line2D([0],[0],marker='o',color='#243B53',linestyle='',label='v0.2 measured snapshot 9baff92')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.05,.883),ncol=2,frameon=False)
    fig.text(.055,.107,'Ryzen 7 7840H  ·  RTX 4050 Laptop 6 GB  ·  WSL2  ·  FP64 / TF32 off  ·  CUDA synchronized',fontsize=11,color='#243B53',weight='bold')
    fig.text(.055,.076,'All workloads: median of 3 timed fits after 2 warmups; whiskers show observed min–max, not confidence intervals.',fontsize=10,color='#536178')
    fig.text(.055,.051,'Three source snapshots interleaved. Validation excluded from fit wall. Later scheduler/version changes were not rebenchmarked.',fontsize=10,color='#536178')
    fig.text(.055,.027,'Source: results/v02/merge_9baff92/revisions/trials.json  ·  BBBD0 is one recording, fitted repeatedly  ·  Ratios use unrounded medians',fontsize=9,color='#718096')
    for ext in ('png','svg'):
        fig.savefig(ROOT/f'docs/assets/performance-v0.2.0.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    print(tab)


if __name__=='__main__':main()
