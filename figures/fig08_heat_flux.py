"""Fig. 8: heat flux and wall actuation at Ra = 1e6 (IC seed 100).

(a) Nu(t) with time means; (b) Welch spectrum of the wall-action norm.  Reads
results/figure_data/fig08_heat_flux.npz from scripts/analysis/rayleigh_benard_heat_flux.py.
Writes figures/fig08_heat_flux.pdf.
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

CPC = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(ROOT, 'results', 'figure_data', 'fig08_heat_flux.npz'))
dt = float(d['dt_ctrl'])
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 9, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
SH, PP, BL = '#2E5894', '#B6324B', '0.55'

fig, ax = plt.subplots(1, 2, figsize=(5.3, 2.75))

# (a) Nu(t) time series + mean lines
a = ax[0]
for key, c, lab, lw in (('nu_bl_t', BL, 'Uncontrolled', 1.0),
                        ('nu_ppo_t', PP, 'PPO', 1.1),
                        ('nu_shac_t', SH, 'SHAC', 1.5)):
    y = np.asarray(d[key]); t = np.arange(len(y)) * dt
    a.plot(t, y, color=c, lw=lw, alpha=0.9, label=lab)
    m = float(np.mean(y[len(y) // 5:]))
    a.axhline(m, color=c, lw=0.8, ls='--', alpha=0.9)
a.set_xlabel(r'$t/t_{\mathrm{ff}}$'); a.set_ylabel(r'$\mathrm{Nu}(t)$')
a.set_title('(a) Heat flux', fontsize=9.5)
a.grid(alpha=0.2, lw=0.4)

# (b) wall-action spectrum
b = ax[1]
b.loglog(d['fa_ppo'], d['Pa_ppo'], color=PP, lw=1.3, label='PPO')
b.loglog(d['fa_shac'], d['Pa_shac'], color=SH, lw=1.7, label='SHAC')
b.set_xlabel(r'$f\,t_{\mathrm{ff}}$'); b.set_ylabel(r'PSD of $\|\boldsymbol{a}(t)\|_2$')
b.set_title('(b) Wall-action spectrum', fontsize=9.5)
b.grid(alpha=0.2, which='both', lw=0.4)

fig.tight_layout(rect=(0, 0, 1, 0.9))
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.6)
fig.savefig(os.path.join(CPC, 'fig08_heat_flux.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(CPC, 'fig08_heat_flux_preview.png'), dpi=170,
            bbox_inches='tight')
print('wrote fig08_heat_flux.pdf')
