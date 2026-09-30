"""Fig. 9: gradient-estimator variance and its consequence.

(a) relative variance of the pathwise (SHAC) and score-function (PPO) gradient
estimators against Ra; (b) held-out Nu reduction per seed; (c) pathwise variance
against the differentiated horizon.  Reads results/diagnostics/gradient_variance*_ra*.json
and results/figure_data/fig09b_reductions.json.  Writes figures/fig09_gradient_variance.pdf.
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
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(__file__), '..')
GV = os.path.join(ROOT, 'results/diagnostics')
RAS = ['1e4', '1e5', '1e6']
RAx = [4, 5, 6]
RALAB = [r'$10^4$', r'$10^5$', r'$10^6$']
SH, PP = '#2E5894', '#B6324B'
BLU = ['#9ecae1', '#3E6DA3', '#08306b']                # Ra=1e4/1e5/1e6 light->dark
# physical time per action in the horizon diagnostic = SHAC training interval x dt
# (grad_variance_horizon.py rolls out with shac_cfg['action_interval'] = 150 at every Ra)
TPA = {'1e4': 1.5, '1e5': 0.6, '1e6': 0.225}

plt.rcParams.update({'font.family': 'serif', 'font.size': 9.5, 'mathtext.fontset': 'cm',
                     'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                     'xtick.direction': 'out', 'ytick.direction': 'out',
                     'legend.frameon': False})

fig = plt.figure(figsize=(5.3, 3.95))
gs = fig.add_gridspec(2, 2, left=0.105, right=0.975, top=0.93, bottom=0.095,
                      hspace=0.55, wspace=0.32)
a = fig.add_subplot(gs[0, 0])      # (a) estimator gap     [SHAC vs PPO]
bc = fig.add_subplot(gs[0, 1])     # (b) consequence       [SHAC vs PPO]
cc = fig.add_subplot(gs[1, :])     # (c) curse of chaos    [SHAC pathwise]
_p = cc.get_position()             # centre (c) at a moderate width (not full)
cc.set_position([0.305, _p.y0, 0.47, _p.height])

# ── (a) estimator variance gap vs Ra ───────────────────────────────────────
gv = {ra: json.load(open(os.path.join(GV, f'gradient_variance_ra{ra}.json'))) for ra in RAS}
# leave-one-out baseline: with b_{-i} = (K b - R_i)/(K-1) each per-trajectory SF
# gradient is exactly K/(K-1) times the batch-mean-baseline one, so its variance
# scales by (K/(K-1))^2 (unbiased LOO estimator, no re-simulation needed)
LOO = {ra: (gv[ra]['K'] / (gv[ra]['K'] - 1)) ** 2 for ra in RAS}
a.semilogy(RAx, [gv[ra]['V_SF'] * LOO[ra] for ra in RAS], 'o-', color=PP, lw=1.9, ms=5)
a.semilogy(RAx, [gv[ra]['V_PW'] for ra in RAS], 's-', color=SH, lw=1.9, ms=5)
a.set_xticks(RAx); a.set_xticklabels(RALAB); a.set_xlim(3.85, 6.15)
a.set_xlabel(r'$\mathrm{Ra}$'); a.set_ylabel(r'Per-trajectory variance $\mathcal{V}$')
a.set_title('(a) Estimator gap', fontsize=9.5)
a.grid(alpha=0.18, which='both', lw=0.4)

# ── (b) consequence: per-seed Nu reduction at a FIXED x per Ra+method ───────
d = json.load(open(os.path.join(ROOT, 'results/figure_data/fig09b_reductions.json')))
for i, ra in enumerate(RAS):
    for algo, col, off in (('ppo', PP, +0.16), ('shac', SH, -0.16)):
        v = np.array(d[ra][algo]['reduction_per_seed'])
        bc.scatter(np.full(len(v), i + off), v, s=12, color=col,
                   alpha=0.65, edgecolor='white', lw=0.3, zorder=3)
        bc.plot([i + off - 0.12, i + off + 0.12], [v.mean()] * 2,
                color=col, lw=2.3, zorder=4)
bc.axhline(0, color='0.6', lw=0.6, ls='--')
bc.set_xticks(range(3)); bc.set_xticklabels(RALAB); bc.set_xlim(-0.5, 2.5)
bc.set_xlabel(r'$\mathrm{Ra}$'); bc.set_ylabel(r'Nu reduction (%)')
bc.set_title('(b) Consequence', fontsize=9.5)
bc.grid(alpha=0.18, axis='y', lw=0.4)

# ── (c) pathwise variance explodes with horizon (full-width bottom row) ─────
for j, ra in enumerate(RAS):
    hz = json.load(open(os.path.join(GV, f'gradient_variance_horizon_ra{ra}.json')))
    h = np.array([r['h'] for r in hz['sweep']])
    V = np.array([r['V_PW'] for r in hz['sweep']])
    k = int(np.argmax(V)) + 1                           # trim heavy-tail noise past peak
    h, V = h[:k], V[:k]
    t = h * TPA[ra]
    cc.loglog(t, V, '-', color=BLU[j], lw=1.9, label=fr'$10^{RAx[j]}$')
    tsh = hz['h_shac'] * TPA[ra]
    cc.plot(tsh, np.exp(np.interp(np.log(tsh), np.log(t), np.log(V))), '*',
            color=BLU[j], ms=9, mec='k', mew=0.5, zorder=6)
cc.set_xlim(4, 34); cc.set_ylim(1.5, 5e3)
cc.set_xticks([5, 10, 20, 30]); cc.set_xticklabels(['5', '10', '20', '30'])
cc.minorticks_off()
cc.set_xlabel(r'Trajectory length ($t_{\mathrm{ff}}$)')
cc.set_ylabel(r'Pathwise variance $\mathcal{V}_{\mathrm{PW}}$')
cc.set_title('(c) Curse of chaos', fontsize=9.5)
cc.grid(alpha=0.18, which='both', lw=0.4)
# framed legend with just the Ra values (no 'Ra' title)
cc.legend(loc='upper left', title=r'$\mathrm{Ra}$', title_fontsize=9, fontsize=9, frameon=True, edgecolor='0.7',
          facecolor='white', framealpha=0.92, handlelength=1.4,
          labelspacing=0.3, borderpad=0.5, handletextpad=0.5)

# ── in-panel legends: estimator names in (a), algorithm names in (b) ─────────
a.legend(handles=[Line2D([0], [0], color=PP, marker='o', lw=1.9, ms=5, label='Score-function'),
                  Line2D([0], [0], color=SH, marker='s', lw=1.9, ms=5, label='Pathwise')],
         loc='center right', fontsize=8, frameon=True, edgecolor='0.7', facecolor='white',
         framealpha=0.92, handlelength=1.6, borderpad=0.4, labelspacing=0.3)
bc.legend(handles=[Line2D([0], [0], color=SH, lw=2.3, label='SHAC'),
                   Line2D([0], [0], color=PP, lw=2.3, label='PPO')],
          loc='upper right', fontsize=8, frameon=True, edgecolor='0.7', facecolor='white',
          framealpha=0.92, handlelength=1.6, borderpad=0.4, labelspacing=0.3)

HERE = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(HERE, 'fig09_gradient_variance.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig09_gradient_variance_preview.png'), dpi=200, bbox_inches='tight')
print('wrote fig09_gradient_variance.pdf')
