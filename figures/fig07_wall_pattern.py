"""Fig. 7: wall-temperature pattern of the SHAC policies at Ra = 1e5 and 1e6.

Mean wall temperature per heater segment (top) and spatial power spectrum of the
time-averaged pattern (bottom), from results/diagnostics/wall_pattern.json
(scripts/analysis/wall_pattern.py).  Writes figures/fig07_wall_pattern.pdf.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
summary = json.load(open(os.path.join(ROOT,
    'results/diagnostics/wall_pattern.json')))
rb = [r for r in summary.get('rb_results', []) if r is not None]

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 9.5, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})
C_SHAC, C_GRAY = '#2E5894', '#555555'
n = len(rb)


def panel_label(ax, L):
    ax.text(0.03, 0.96, f'({L})', transform=ax.transAxes, ha='left',
            va='top', fontsize=9.5, fontweight='bold')


letters = [['a', 'b', 'c'], ['d', 'e', 'f']] if n == 3 else [['a', 'b'], ['c', 'd']]
fig, axes = plt.subplots(2, n, figsize=(5.3, 3.7))
axes = np.atleast_2d(axes)
for i, r in enumerate(rb):
    # top: mean wall temperature per segment
    ax = axes[0, i]
    vals = np.array(r['mean_T_wall']); x = np.arange(len(vals)) + 1
    ax.bar(x, vals, color=plt.cm.RdBu_r((vals - 0.25) / 1.5), edgecolor='k', lw=0.3)
    ax.axhline(1.0, color=C_GRAY, ls='--', lw=0.8)
    ax.set_ylabel(r'Mean $T_{\mathrm{wall}}$')
    ax.set_ylim(0, 2.0); ax.set_xticks([1, 4, 8, 12])
    ax.set_xlabel('Segment index')
    ra_exp = int(round(np.log10(r['Ra'])))
    ax.set_title(rf'$\mathrm{{Ra}}=10^{{{ra_exp}}}$', fontsize=9.5)
    panel_label(ax, letters[0][i])
    # bottom: spatial FFT power
    ax = axes[1, i]
    power = np.array(r['fft_power']); k = np.arange(len(power))
    ax.bar(k[1:], power[1:] / np.max(power[1:]), color=C_SHAC,
           edgecolor='none', width=0.7)
    ax.set_xlabel(r'Wavenumber $k$'); ax.set_ylabel('Normalised power')
    ax.set_ylim(0, 1.3)
    panel_label(ax, letters[1][i])
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, 'fig07_wall_pattern.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig07_wall_pattern_preview.png'), dpi=160,
            bbox_inches='tight')
print('saved fig6_policy_physics (5.3in, 2x%d)' % n)
