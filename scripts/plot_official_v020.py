"""Plot retained official GSBS timings versus measured v0.2.0 wheel timings.

No fitting. --check verifies derived data and both README tables.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/releases/v0.2.0/official_comparison.json'
OLD=ROOT/'results/benchmarks/delivery_latency.json'
ASSETS=ROOT/'docs/assets'
NAMES=['Small','Medium','Large','XL','BBBD0']
BACKENDS=['reference','cpu-fast','torch-cuda','triton-cuda']
COLORS=['#718096','#D57927','#008F97','#7154C4']
LABELS=['Official GSBS (retained baseline)','v0.2.0 CPU','v0.2.0 PyTorch','v0.2.0 Triton']


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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator,FuncFormatter,MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'hpgsbs-official-v020'})
    fig=plt.figure(figsize=(16,10.5),facecolor='#F5F7FA')
    grid=fig.add_gridspec(1,2,left=.15,right=.97,bottom=.235,top=.755,width_ratios=[1.15,1],wspace=.12)
    ax,bx=[fig.add_subplot(grid[0,i]) for i in range(2)]; centers=np.arange(5)*5
    for axis in (ax,bx):
        axis.set_facecolor('#F5F7FA');axis.set_ylim(22,-2);axis.set_axisbelow(True)
        axis.grid(axis='x',color='#DDE3EC',lw=.7);axis.tick_params(length=0,pad=8,colors='#536178')
        for spine in axis.spines.values():spine.set_visible(False)
        for i,y in enumerate(centers):
            if i%2==0:axis.axhspan(y-2,y+2,color='white',zorder=0)
    ax.set_xscale('log');ax.set_xlim(.025,60000)
    ax.xaxis.set_major_locator(FixedLocator([.1,1,10,100,1000,10000]));ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}' if v>=1 else '0.1'))
    ax.set_title('01   Complete-fit wall time',loc='left',pad=23,fontsize=16,fontweight='bold',color='#142A43')
    ax.set_xlabel('Seconds  ·  logarithmic axis',labelpad=16)
    bx.set_title('02   Speedup vs official GSBS',loc='left',pad=23,fontsize=16,fontweight='bold',color='#142A43')
    maximum=max(c['speedup'] for c in report['cells']);bx.set_xlim(0,maximum*1.27)
    bx.xaxis.set_major_locator(MaxNLocator(5));bx.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}×'))
    bx.set_xlabel('Official seconds ÷ v0.2.0 seconds  ·  linear axis',labelpad=16);bx.set_yticks([])
    labels=[]
    for i,name in enumerate(NAMES):
        group=[c for c in report['cells'] if c['workload']==name];ref=group[0];T,P=ref['shape']
        labels.append(f"{'Real EEG*' if name=='BBBD0' else name}\n{T:,} × {P}  |  Kmax {ref['kmax']}")
        for j,c in enumerate(group):
            y=centers[i]+[-1.35,-.45,.45,1.35][j]
            if c['timed_fits']>1:ax.plot([c['min_s'],c['max_s']],[y,y],lw=2,color=COLORS[j])
            ax.scatter(c['seconds'],y,s=49,color=COLORS[j],edgecolor='white',lw=.8,zorder=3)
            ax.annotate(f"{c['seconds']:,.3f} s",(c['max_s'],y),xytext=(7,0),textcoords='offset points',va='center',fontsize=10,color='#243B53')
            if j:
                by=centers[i]+[-1.05,0,1.05][j-1];bx.barh(by,c['speedup'],height=.7,color=COLORS[j])
                bx.text(c['speedup']+maximum*.012,by,f"{c['speedup']:,.1f}×",va='center',fontsize=11,fontweight='bold',color=COLORS[j])
    ax.set_yticks(centers,labels);ax.tick_params(axis='y',pad=18,labelcolor='#243B53')
    fig.text(.045,.95,'HP-GSBS v0.2.0 vs official GSBS',fontsize=27,fontweight='bold',color='#142A43')
    fig.text(.045,.907,'Measured release-wheel performance on the tested workloads',fontsize=18,color='#142A43')
    fig.text(.045,.871,'Historical official baseline + new v0.2.0 measurements  /  cross-session comparison, not a fresh paired run',fontsize=11,color='#536178')
    fig.legend(handles=[Line2D([0],[0],marker='o',linestyle='',color=c,markersize=8,label=l) for c,l in zip(COLORS,LABELS)],loc='upper left',bbox_to_anchor=(.04,.844),ncol=4,frameon=False,columnspacing=2.0,fontsize=11)
    foot=[('Ryzen 7 7840H  ·  RTX 4050 Laptop 6 GB  ·  WSL2  ·  FP64 / TF32 off  ·  BLAS/Numba: 1 thread',.151),('Synthetic: median of 3 timed fits after 1 warmup; whiskers = observed min–max.  *Real EEG: 1 timed fit per backend; no range.',.122),('Same scientific parameters and input recipe; Medium/Large/XL/EEG hashes verified. Small historical receipt lacks a hash; seed reconstructed.',.093),('v0.2.0: published wheel, commit b86b409. All 51 new warmup/timed fits passed official-output comparison. Validation and I/O excluded.',.064),('Official: statesegmentation 0.0.6, retained measurements. Different sessions can change clock/load; ratios are descriptive, not causal estimates.',.035)]
    for label,y in foot:fig.text(.045,y,label,fontsize=9.4,color='#536178')
    fig.savefig(ASSETS/'performance-official-v0.2.0.png',dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(ASSETS/'performance-official-v0.2.0.svg',facecolor=fig.get_facecolor(),metadata={'Date':None})
    svg=ASSETS/'performance-official-v0.2.0.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    report,table=derive();serialized=json.dumps(report,indent=2)+'\n'
    json_path=ASSETS/'performance-official-v0.2.0.json';table_path=ROOT/'docs/v020_official_table.md'
    if a.check:
        assert json_path.read_text()==serialized and table_path.read_text()==table
    else:
        json_path.write_text(serialized,encoding='utf-8',newline='\n');table_path.write_text(table,encoding='utf-8',newline='\n');plot(report)
    for name in ('README.md','README.zh-CN.md'):
        file=ROOT/name;text=file.read_text(encoding='utf-8')
        match=re.search(r'<!-- OFFICIAL_V020_TABLE_START -->\n(.*?)<!-- OFFICIAL_V020_TABLE_END -->',text,re.S)
        assert match,name
        if a.check:assert match.group(1)==table
        else:file.write_text(text[:match.start(1)]+table+text[match.end(1):],encoding='utf-8',newline='\n')
    print('20 official/release cells and both README tables match measured receipts')


if __name__=='__main__':main()
