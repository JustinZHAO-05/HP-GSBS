"""Render measured benchmarks for the README (pip install matplotlib).

Run from any directory: python scripts/plot_performance.py
No fitting is performed. Figures use the saved, unrounded measurements.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['Small', 'Medium', 'Large', 'XL', 'BBBD0']
BACKENDS = ['reference', 'cpu-fast', 'torch-cuda', 'triton-cuda']
COLORS = ['#718096', '#D57927', '#008F97', '#7154C4']
LABELS = ['Official GSBS', 'CPU FastGSBS', 'TensorGSBS / PyTorch', 'TurboGSBS / Triton']


def main():
    rows = json.loads((ROOT/'results/benchmarks/delivery_latency.json').read_text())
    pairs = {(r['workload'], r['backend']): r for r in rows}
    wall = lambda r: r.get('single_fit_s', r.get('steady_median_s'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(15, 10), facecolor='#F5F7FA')
    grid = fig.add_gridspec(1, 2, left=.155, right=.965, bottom=.195,
                           top=.735, width_ratios=[1.16, 1], wspace=.13)
    ax, bx = [fig.add_subplot(grid[0, i]) for i in range(2)]
    centers = np.arange(5)*5
    for a in (ax, bx):
        a.set_facecolor('#F5F7FA')
        a.set_ylim(22, -2)
        for spine in a.spines.values():
            spine.set_visible(False)
        a.tick_params(axis='both', length=0, colors='#536178', pad=8)
        a.set_axisbelow(True)
        a.grid(axis='x', color='#DDE3EC', linewidth=.7)
        for j, y in enumerate(centers):
            if j % 2 == 0:
                a.axhspan(y-2, y+2, color='white', zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(.055, 60000)
    ax.xaxis.set_major_locator(FixedLocator([.1, 1, 10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else '0.1'))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel('Wall time (seconds)  ·  logarithmic axis', labelpad=16, color='#536178')
    ax.set_title('01   Run time', loc='left', pad=26, fontsize=16, color='#142A43')
    bx.set_title('02   Speedup vs official GSBS', loc='left', pad=26, fontsize=16, color='#142A43')
    bx.set_xlim(0, 5100)
    bx.set_xticks([0, 1000, 2000, 3000, 4000])
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}×'))
    bx.set_xlabel('Official time ÷ backend time  ·  linear axis', labelpad=16, color='#536178')
    bx.set_yticks([])
    workload_labels = []
    for i, name in enumerate(NAMES):
        ref = pairs[name, 'reference']
        T, P = ref['shape']
        label = 'Real EEG*' if name == 'BBBD0' else name
        workload_labels.append(f'{label}\n{T:,} × {P}  |  Kmax {ref["algorithm"]["kmax"]}')
        for j, backend in enumerate(BACKENDS):
            r = pairs[name, backend]
            value = wall(r)
            assert value > 0 and r['precision'] == 'float64'
            trials = r['trials'] if 'single_fit_s' in r else r['trials'][1:]
            assert all(t.get('validation_against_cpu', r['validation_against_cpu'])['passed'] for t in trials)
            ys = centers[i] + [-1.35, -.45, .45, 1.35][j]
            times = [t['wall_s'] for t in trials]
            if len(times) > 1:
                ax.plot([min(times), max(times)], [ys, ys], color=COLORS[j], lw=2, zorder=2)
            ax.scatter(value, ys, s=48, color=COLORS[j], edgecolor='white', linewidth=.9, zorder=3)
            ax.annotate(f'{value:,.3f} s', (max(times), ys), xytext=(8, 0), textcoords='offset points',
                        va='center', fontsize=10, color='#233950')
            if j:
                ratio = wall(ref)/value
                by = centers[i] + [-1.05, 0, 1.05][j-1]
                bx.barh(by, ratio, height=.7, color=COLORS[j], zorder=2)
                bx.text(ratio+65, by, f'{ratio:,.1f}×', va='center', fontsize=11,
                        fontweight='bold', color=COLORS[j])
    ax.set_yticks(centers, workload_labels)
    ax.tick_params(axis='y', pad=20, labelcolor='#243B53', labelsize=10)
    fig.text(.045, .947, 'HP-GSBS', color='#142A43', weight='bold', fontsize=29)
    fig.text(.045, .903, 'Less computation. The same tested boundaries.', color='#142A43', fontsize=19)
    fig.text(.045, .866, 'Measured FP64 performance  /  identical input and algorithm parameters within each workload',
             color='#536178', fontsize=11)
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=9, color=c, label=l)
               for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.04, .836), ncol=4,
               frameon=False, columnspacing=2.3, handletextpad=.5, fontsize=11)
    fig.text(.045, .103, 'Ryzen 7 7840H  ·  RTX 4050 Laptop 6 GB  ·  WSL2  ·  FP64 / TF32 off  ·  CUDA synchronized',
             fontsize=11, color='#243B53', weight='bold')
    fig.text(.045, .074, 'Synthetic: median of 3 fits after warmup; whiskers show min–max.  *Real EEG: 1 timed fit per backend; no repeat range.',
             fontsize=10, color='#536178')
    fig.text(.045, .049, 'Small: retained v0.1.0 measurements. Other workloads: paired remeasurement. Saving and validation excluded from timing.',
             fontsize=10, color='#536178')
    fig.text(.045, .024, 'Source: results/benchmarks/delivery_latency.json  ·  Speedups calculated from unrounded times  ·  Workload-specific results',
             fontsize=9, color='#718096')
    out = ROOT/'docs/assets'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'performance-comparison.png', dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(out/'performance-comparison.svg', facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved PNG and SVG from measured benchmark receipts')


if __name__ == '__main__':
    main()
