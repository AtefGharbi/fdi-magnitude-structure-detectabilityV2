"""Figure 3 (motivating example): Random Forest F1-score by class and test system,
mean +/- SD across the 8 seeds. Reads the per-seed files results/diag_<system>_s<seed>.json
written by run_all.py (field "rf" = [F1 normal, F1 sparse/high-amplitude, F1 dense/low-amplitude]).
These values are identical to the V1 multiseed CSVs; set FDI_V1_MULTISEED to plot from those instead."""
import os, glob, json
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.environ.get('FDI_V1_MULTISEED')


def load(system):
    if V1:
        import pandas as pd
        d = pd.read_csv(os.path.join(V1, f'{system}_perseed_results.csv'))
        return d[['multiclass_f1_normal', 'multiclass_f1_simple', 'multiclass_f1_stealthy']].values
    files = sorted(glob.glob(os.path.join(ROOT, 'results', f'diag_{system}_s*.json')))
    assert len(files) == 8, f'expected 8 seeds for {system}, found {len(files)}'
    return np.array([json.load(open(f))['rf'] for f in files])


NAVY, BLUE = '#203864', '#2E75B6'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
labs = ['Normal', 'Sparse / high-amplitude\n(8% of channels, 4σ)', 'Dense / low-amplitude\n(all channels, 2σ)']
x = np.arange(3); w = 0.36
for j, (s, c) in enumerate([('14-bus', NAVY), ('118-bus', BLUE)]):
    a = load(s); m = a.mean(0); sd = a.std(0, ddof=1)
    ax.bar(x + (j - 0.5) * w, m, w, yerr=sd, capsize=4, color=c, label=f'IEEE {s}', edgecolor='white', linewidth=0.5)
    for xi, mi, si in zip(x + (j - 0.5) * w, m, sd):
        ax.text(xi, mi + si + 0.02, f'{mi:.3f}', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('F1-score (mean ± SD, 8 seeds)'); ax.set_ylim(0, 1.12)
ax.yaxis.grid(True, ls='--', alpha=0.4); ax.set_axisbelow(True)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, 'figures', 'fig3_motivating_example.png'), dpi=300)
print('figure 3 written')
