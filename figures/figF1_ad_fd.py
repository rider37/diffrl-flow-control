"""Fig. F.1: cosine similarity of AD and FD gradients.

(a) per Ra, at random policy states; (b) against the rollout horizon.  Reads
results/diagnostics/ad_fd_cosine.json and ad_fd_horizon.json from
scripts/analysis/ad_fd_gradients.py.  Writes figures/figF1_ad_fd.pdf.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'xtick.minor.size': 2.2,
    'ytick.minor.size': 2.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
                        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                        'legend.fontsize': 9.5})

C_SHAC = '#1f77b4'
C_PPO = '#d62728'
C_GRAY = '#555555'
C_GREEN = '#2ca02c'


def load_history(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def make_figure():
    cos_path = os.path.join(ROOT,
        'results/diagnostics/ad_fd_cosine.json')
    hor_path = os.path.join(ROOT,
        'results/diagnostics/ad_fd_horizon.json')

    if not (os.path.exists(cos_path) and os.path.exists(hor_path)):
        print('[Fig. F.1] skip')
        return

    cos_data = json.load(open(cos_path))
    hor_data = json.load(open(hor_path))

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))

    # (a) cosine similarity vs Ra
    ax = axes[0]
    ra_vals = [d['Ra'] for d in cos_data]
    cos_means = [d['mean_cosine_sim'] for d in cos_data]
    cos_stds = [d['std_cosine_sim'] for d in cos_data]
    cos_medians = [np.median([r['cosine_similarity'] for r in d['results']])
                   for d in cos_data]

    x = np.arange(len(ra_vals))
    # Individual per-state cosine similarities (jittered) so the
    # Ra=1e4 outlier states (a few near-zero loss landscape rollouts) are
    # visible directly, not only via the wide error bar.
    rng = np.random.default_rng(0)
    for i, d in enumerate(cos_data):
        per_state = [r['cosine_similarity'] for r in d['results']]
        jitter = rng.uniform(-0.10, 0.10, size=len(per_state))
        ax.scatter(x[i] + jitter, per_state, s=12, facecolor='white',
                   edgecolor=C_SHAC, linewidth=0.7, alpha=0.85, zorder=2)
    ax.errorbar(x, cos_means, yerr=cos_stds, fmt='o', color=C_SHAC,
                markersize=6.5, capsize=4, label=r'Mean $\pm$ SD',
                elinewidth=1.4, capthick=1.4, zorder=4)
    ax.plot(x, cos_medians, 's', color=C_PPO, markersize=6,
            label='Median', markerfacecolor='none', markeredgewidth=1.4,
            zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'$10^{{{int(np.log10(r))}}}$' for r in ra_vals])
    ax.set_xlabel(r'$\mathrm{Ra}$')
    ax.set_ylabel('$\\cos(\\boldsymbol{g}_{\\mathrm{AD}},\\boldsymbol{g}_{\\mathrm{FD}})$')
    ax.set_ylim(-0.5, 1.1)
    ax.axhline(0.95, color=C_GRAY, ls='--', alpha=0.5, lw=0.6)
    ax.axhline(0.0,  color='k',    ls=':',  alpha=0.4, lw=0.5)
    ax.legend(loc='lower right', fontsize=9.5)
    ax.text(0.0, 1.02, '(a)', transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold')
    # inset: zoom on Ra=1e5 and 1e6, whose per-state values crowd near 1
    axin = ax.inset_axes([0.36, 0.30, 0.60, 0.46])
    for i, d in enumerate(cos_data):
        if d['Ra'] < 5e4:
            continue
        per_state = [r['cosine_similarity'] for r in d['results']]
        jitter = rng.uniform(-0.10, 0.10, size=len(per_state))
        axin.scatter(x[i] + jitter, per_state, s=12, facecolor='white',
                     edgecolor=C_SHAC, linewidth=0.7, alpha=0.85, zorder=2)
        axin.errorbar([x[i]], [cos_means[i]], yerr=[cos_stds[i]], fmt='o',
                      color=C_SHAC, markersize=5, capsize=3, elinewidth=1.2,
                      capthick=1.2, zorder=4)
        axin.plot([x[i]], [cos_medians[i]], 's', color=C_PPO, markersize=5,
                  markerfacecolor='none', markeredgewidth=1.2, zorder=5)
    axin.axhline(0.95, color=C_GRAY, ls='--', alpha=0.5, lw=0.6)
    axin.set_xlim(0.6, 2.4); axin.set_ylim(0.88, 1.005)
    axin.set_xticks([1, 2]); axin.set_xticklabels([r'$10^5$', r'$10^6$'])
    axin.set_yticks([0.90, 0.95, 1.00])
    axin.tick_params(labelsize=9, length=2)
    axin.set_title('Zoom', fontsize=9, pad=2)

    # (b) horizon sweep
    ax = axes[1]
    markers = ['o', 's', '^']
    colors = [C_GREEN, C_SHAC, C_PPO]
    for i, hr in enumerate(hor_data):
        ra = hr['Ra']
        hs = [r['horizon'] for r in hr['results']]
        cs = [r['cosine_similarity'] for r in hr['results']]
        ax.plot(hs, cs, marker=markers[i], color=colors[i],
                markersize=6, markerfacecolor='none', markeredgewidth=1.2,
                label=rf'Ra = $10^{{{int(np.log10(ra))}}}$')

    ax.set_xlabel(r'Horizon $h$')
    ax.set_ylabel(r'$\cos(\boldsymbol{g}_{\mathrm{AD}},\boldsymbol{g}_{\mathrm{FD}})$')
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.95, color=C_GRAY, ls='--', alpha=0.5, lw=0.6)
    ax.legend(loc='lower right')
    ax.text(0.0, 1.02, '(b)', transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIG_DIR, 'figF1_ad_fd.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'figF1_ad_fd.png'))
    plt.close()
    print('[Fig. F.1] saved.')


if __name__ == '__main__':
    make_figure()
