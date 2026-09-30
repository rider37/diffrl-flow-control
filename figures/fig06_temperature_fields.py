"""Fig. 6: temperature fields of the uncontrolled flow, PPO and SHAC at Ra = 1e4, 1e5, 1e6.

Reads results/figure_data/fig06_temperature_ra*.npz from
scripts/analysis/rayleigh_benard_fields.py (same IC, warm-up and number of
control actions in the three rows).  Writes figures/fig06_temperature_fields.pdf.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAS = ['1e4', '1e5', '1e6']
RALAB = [r'$\mathrm{Ra}=10^4$', r'$\mathrm{Ra}=10^5$', r'$\mathrm{Ra}=10^6$']
ROWS = [('Uncontrolled', 'uncontrolled'), ('PPO', 'ppo'), ('SHAC', 'shac')]

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9.5,
                     'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
data = {ra: np.load(os.path.join(ROOT, 'results', 'figure_data', f'fig06_temperature_ra{ra}.npz')) for ra in RAS}

fig, axes = plt.subplots(3, 3, figsize=(5.3, 3.45))
im = None
for r, (rlab, key) in enumerate(ROWS):
    for c, ra in enumerate(RAS):
        ax = axes[r, c]
        d = data[ra]
        T = np.asarray(d[key]); xc = np.asarray(d['xc']); yc = np.asarray(d['yc'])
        im = ax.pcolormesh(xc, yc, T.T, cmap='RdBu_r', vmin=0.0, vmax=1.2,
                      shading='auto', rasterized=True)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(labelsize=8.5, length=2, pad=1.5)
        if r == len(ROWS) - 1:
            ax.set_xticks([0, np.pi, 2 * np.pi])
            ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
            if c == 1:
                ax.set_xlabel(r'$x$', fontsize=9.5, labelpad=1)
        if c == 0:
            ax.set_yticks([0, 2])
        if r == 0:
            ax.set_title(RALAB[c], fontsize=9.5)
        if c == 0:
            ax.set_ylabel(rlab, fontsize=9.5)
fig.subplots_adjust(right=0.995, left=0.09, top=0.93, bottom=0.215, hspace=0.42, wspace=0.16)
# shared horizontal colourbar below the grid (same scale in every panel)
cax = fig.add_axes([0.35, 0.075, 0.40, 0.022])
cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 0.4, 0.8, 1.2])
cb.set_label(r'$T$', fontsize=9.5, labelpad=2)
cb.ax.tick_params(labelsize=8.5, length=2)
fig.savefig(os.path.join(HERE, 'fig06_temperature_fields.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(HERE, 'fig06_temperature_fields_preview.png'), dpi=150, bbox_inches='tight')
print('saved fig5_flow_fields (time-averaged 3x3 RB temperature)')
