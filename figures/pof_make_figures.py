"""
Shared plotting style and the gradient-quality renderer (make_fig7) used by
make_figF8_gradient_quality.py.

PoF / AIP style:
- Serif font, small sizes (8-10pt)
- No titles (caption conveys meaning)
- Inward minor ticks
- Muted colors, minimal decoration
- (a), (b) labels inside panels
"""

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# ═══════════════════════════════════════════════════════════════════
# PoF style configuration (global)
# ═══════════════════════════════════════════════════════════════════
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

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.dirname(os.path.abspath(__file__))

# PoF-style muted colors
C_SHAC = '#1f77b4'     # muted blue
C_PPO  = '#d62728'     # muted red
C_PD   = '#2ca02c'     # muted green
C_GRAY = '#555555'

# Figure widths (inches): single column ~3.4", double column ~7.0"
W_SINGLE = 3.4
W_DOUBLE = 7.0


def panel_label(ax, letter, loc='upper left', fontsize=10):
    """Place (a), (b), ... label inside panel, PoF style."""
    locs = {
        'upper left':  (0.02, 0.97, 'left', 'top'),
        'upper right': (0.98, 0.97, 'right', 'top'),
    }
    x, y, ha, va = locs[loc]
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.9))


def load_history(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def make_fig7():
    cos_path = os.path.join(_root,
        'results/gradient_quality/cosine_similarity.json')
    hor_path = os.path.join(_root,
        'results/gradient_quality/horizon_sweep.json')

    if not (os.path.exists(cos_path) and os.path.exists(hor_path)):
        print('[Fig. F.1] skip')
        return

    cos_data = json.load(open(cos_path))
    hor_data = json.load(open(hor_path))

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.8))

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
                markersize=6.5, capsize=4, label=r'mean $\pm$ std',
                elinewidth=1.4, capthick=1.4, zorder=4)
    ax.plot(x, cos_medians, 's', color=C_PPO, markersize=6,
            label='median', markerfacecolor='none', markeredgewidth=1.4,
            zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'$10^{{{int(np.log10(r))}}}$' for r in ra_vals])
    ax.set_xlabel('Ra')
    ax.set_ylabel('cosine similarity (AD vs FD)')
    ax.set_ylim(-0.5, 1.1)
    ax.axhline(0.95, color=C_GRAY, ls='--', alpha=0.5, lw=0.6)
    ax.axhline(0.0,  color='k',    ls=':',  alpha=0.4, lw=0.5)
    ax.legend(loc='lower right', fontsize=8)
    panel_label(ax, 'a')

    # (b) horizon sweep
    ax = axes[1]
    markers = ['o', 's', '^']
    colors = [C_PD, C_SHAC, C_PPO]
    for i, hr in enumerate(hor_data):
        ra = hr['Ra']
        hs = [r['horizon'] for r in hr['results']]
        cs = [r['cosine_similarity'] for r in hr['results']]
        ax.plot(hs, cs, marker=markers[i], color=colors[i],
                markersize=5, markerfacecolor='none', markeredgewidth=1.0,
                label=rf'Ra = $10^{{{int(np.log10(ra))}}}$')

    ax.set_xlabel(r'rollout horizon $h$')
    ax.set_ylabel('cosine similarity')
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.95, color=C_GRAY, ls='--', alpha=0.5, lw=0.6)
    ax.legend(loc='lower right')
    panel_label(ax, 'b')

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIG_DIR, 'fig7_gradient_quality.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig7_gradient_quality.png'))
    plt.close()
    print('[Fig. F.1] saved.')
