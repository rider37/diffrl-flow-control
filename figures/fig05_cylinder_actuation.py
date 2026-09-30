"""Fig. 5: cylinder actuation against Reynolds number.

(a) mean jet amplitude <|a|> and (b) RMS of its fluctuation, per seed and seed
mean, for SHAC and PPO.  Reads results/figure_data/fig05_cylinder_actuation.npz
from scripts/analysis/cylinder_actuation.py.  Writes figures/fig05_cylinder_actuation.pdf.
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
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
A = np.load(os.path.join(ROOT, 'results/figure_data/fig05_cylinder_actuation.npz'))
Re = list(A['Re']); EP = A[f'shac_re{Re[0]}'].shape[1]; H = EP // 2

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 10, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.frameon': False})
BLUE, RED = '#2E5894', '#B6324B'
x = np.arange(len(Re))


def stats(algo):
    ma, rms, per_ma, per_rms = [], [], [], []
    for re in Re:
        late = A[f'{algo}_re{re}'][:, H:]
        m = np.nanmean(np.abs(late), 1); r = np.nanstd(late, 1)
        ma.append(np.nanmean(m)); rms.append(np.nanmean(r)); per_ma.append(m); per_rms.append(r)
    return np.array(ma), np.array(rms), per_ma, per_rms


sh_ma, sh_rms, sh_pma, sh_prms = stats('shac')
pp_ma, pp_rms, pp_pma, pp_prms = stats('ppo')


def scatter(ax, per, off, color):
    for i in range(len(Re)):
        ax.scatter(np.full(len(per[i]), x[i] + off), per[i], s=13, color=color,
                   alpha=0.5, edgecolors='none', zorder=2)


def plabel(ax, s):
    ax.text(0.04, 0.95, s, transform=ax.transAxes, ha='left', va='top', fontsize=9.5, fontweight='bold')


fig = plt.figure(figsize=(5.2, 2.25))
gs = gridspec.GridSpec(1, 2, wspace=0.30, left=0.095, right=0.985, top=0.955, bottom=0.175)

axm = fig.add_subplot(gs[0])
scatter(axm, sh_pma, -0.12, BLUE); scatter(axm, pp_pma, 0.12, RED)
axm.plot(x, sh_ma, '-o', color=BLUE, lw=1.5, ms=4.5, zorder=3, label='SHAC')
axm.plot(x, pp_ma, '-o', color=RED, lw=1.5, ms=4.5, zorder=3, label='PPO')
axm.axhline(1, color='0.6', lw=0.6, ls=':')
axm.set_xticks(x); axm.set_xticklabels([f'{r}' for r in Re]); axm.set_xlabel(r'$\mathrm{Re}$')
axm.set_ylim(0, 1.15); axm.set_xlim(-0.42, len(Re) - 0.58)
axm.set_ylabel(r'Mean amplitude $\langle|a|\rangle$')
axm.grid(alpha=0.2, lw=0.4); plabel(axm, '(a)')
axm.legend(loc='lower right', fontsize=8.5, handlelength=1.5, labelspacing=0.3, borderpad=0.3)

axr = fig.add_subplot(gs[1])
scatter(axr, sh_prms, -0.12, BLUE); scatter(axr, pp_prms, 0.12, RED)
axr.plot(x, sh_rms, '-s', color=BLUE, lw=1.5, ms=4.5, zorder=3)
axr.plot(x, pp_rms, '-s', color=RED, lw=1.5, ms=4.5, zorder=3)
axr.set_xticks(x); axr.set_xticklabels([f'{r}' for r in Re]); axr.set_xlabel(r'$\mathrm{Re}$')
axr.set_xlim(-0.42, len(Re) - 0.58); axr.set_ylim(bottom=0)
axr.set_ylabel(r'RMS fluctuation of $a$'); axr.grid(alpha=0.2, lw=0.4); plabel(axr, '(b)')

fig.savefig(os.path.join(HERE, 'fig05_cylinder_actuation.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig05_cylinder_actuation_preview.png'), dpi=175, bbox_inches='tight')
print('saved fig_cyl_actuation')
for i, r in enumerate(Re):
    print(f"  Re{r}: SHAC |a|={sh_ma[i]:.2f} rms={sh_rms[i]:.2f} | PPO |a|={pp_ma[i]:.2f} rms={pp_rms[i]:.2f}")
