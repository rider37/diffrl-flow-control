"""Training-efficiency figures (Figs. 3 and 4), five-seed mean with a +-1 SD band.

  fig03_cylinder_training.pdf         cylinder, Re = 100-500 (Fig. 3)
  fig04_rayleigh_benard_training.pdf  Rayleigh-Benard, Ra = 1e4-1e6 (Fig. 4)
Top row: training reduction against wall-clock time, which includes the SHAC
backward pass; bottom row: against CFD steps.  Reads the training histories
(runs/*/*/history.json).
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 10.5,
                     'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'mathtext.fontset': 'cm',
                     'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0,
                     'xtick.direction': 'out', 'ytick.direction': 'out',
                     'legend.frameon': False})

DF = ROOT
OUT = os.path.dirname(os.path.abspath(__file__))
SHAC_BLUE, PPO_RED = '#2E5894', '#B6324B'
SEEDS = [42, 123, 456, 517, 1223]


def smooth(y, w):
    y = np.asarray(y, float)
    if len(y) < 3 or w < 2:
        return y
    half = min(w, len(y)) // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = y[max(0, i - half):min(len(y), i + half + 1)].mean()
    return out


def load_seed(path, base_key, val_key):
    """Return (t_min, steps_M, reduction_pct) arrays, or None."""
    h = os.path.join(path, 'history.json')
    if not os.path.exists(h):
        return None
    d = json.load(open(h))
    if not all(k in d for k in (base_key, val_key, 'cumulative_time', 'total_cfd_steps')):
        return None
    base = float(d[base_key])
    val = np.asarray(d[val_key], float)
    if base == 0 or len(val) == 0:
        return None
    red = (base - val) / base * 100.0          # positive = reduction
    t = np.asarray(d['cumulative_time'], float) / 60.0
    steps = np.asarray(d['total_cfd_steps'], float) / 1e6
    n = min(len(red), len(t), len(steps))
    return t[:n], steps[:n], red[:n]


def collect(patterns, base_key, val_key):
    runs = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            r = load_seed(os.path.dirname(p) if p.endswith('history.json') else p,
                          base_key, val_key)
            if r is not None:
                runs.append(r)
    return runs


def band(ax, runs, xkey, color, label, w):
    """Interpolate seeds onto a common grid, plot mean + +-1 std band."""
    if not runs:
        return
    xi = 0 if xkey == 'wall' else 1
    xs = [r[xi] for r in runs]
    ys = [smooth(r[2], w) for r in runs]
    hi = min(x.max() for x in xs)          # common range all seeds cover
    lo = max(min(x.min() for x in xs), 0.0)
    xg = np.linspace(lo, hi, 250)
    Y = np.array([np.interp(xg, x, y) for x, y in zip(xs, ys)])
    m = Y.mean(0)
    sd = Y.std(0, ddof=1)
    ax.fill_between(xg, m - sd, m + sd, color=color, alpha=0.16,
                    lw=0, zorder=2)
    ax.plot(xg, m, color=color, lw=1.9, zorder=4,
            solid_capstyle='round', label=f'{label} (n={len(runs)})')


# Unified panel geometry (inches): a FIXED per-panel width PW so each graph is
# the SAME size in the cylinder (4-col) and RB (3-col) figures.  The RB figure is
# just narrower overall and placed at a smaller \includegraphics width in LaTeX
# (rb width / cyl width = 4.19/5.49 ~ 0.76, so fig8 uses width=0.76\textwidth).
PW, GAP, LM, RM, FH = 1.10, 0.15, 0.58, 0.06, 4.4


def make_figure(cases, base_key, val_key, dir_fn, xlabels, fname, ra=False):
    """2 rows (wall-clock / env-steps) x N cols (cases), FIXED per-panel width PW
    so the graphs match across the cylinder and RB figures.  A shared y-axis
    (one common reduction scale) puts the y-tick numbers and a single y-label
    only on the far-left column; the freed inner-tick space widens the panels.
    Per-case x-ranges, never clipped."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    ncol = len(cases)
    qty = 'Nusselt-number reduction (%)' if ra else 'Drag reduction (%)'
    fig_w = LM + ncol * PW + (ncol - 1) * GAP + RM
    fig, axes = plt.subplots(2, ncol, figsize=(fig_w, FH), sharey=True)  # y once, left
    axes = np.atleast_2d(axes)
    TOP, BOT, HSP = 0.86, 0.105, 0.50
    fig.subplots_adjust(left=LM / fig_w, right=1 - RM / fig_w,
                        top=TOP, bottom=BOT, hspace=HSP, wspace=GAP / PW)
    for j, c in enumerate(cases):
        shac = collect(dir_fn(c, 'shac'), base_key, val_key)
        ppo = collect(dir_fn(c, 'ppo'), base_key, val_key)
        # identical smoothing rule for both algorithms: centred moving
        # average with a window of 1/8 of each run's recorded epochs
        wsh = max(5, (max((len(r[2]) for r in shac), default=30)) // 8)
        wpp = max(5, (max((len(r[2]) for r in ppo), default=30)) // 8)
        for row, xkey in enumerate(('wall', 'step')):
            ax = axes[row, j]
            band(ax, shac, xkey, SHAC_BLUE, 'SHAC', wsh)
            band(ax, ppo, xkey, PPO_RED, 'PPO', wpp)
            ho = c['ho']
            ax.axhline(ho['shac'], color=SHAC_BLUE, ls=':', lw=1.0, alpha=0.9)
            ax.axhline(ho['ppo'], color=PPO_RED, ls=':', lw=1.0, alpha=0.9)
            ax.axhline(0, color='0.6', lw=0.5)
            ax.grid(alpha=0.18, lw=0.4)
            ax.tick_params(length=2.5)
            ax.margins(x=0.02)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        axes[0, j].set_title(c['title'], fontsize=10.5)
    fig.supylabel(qty, fontsize=10.5, x=0.014)   # one shared y-label
    # one shared x-label per row (every panel in a row shares the same quantity)
    xc = (LM / fig_w + 1 - RM / fig_w) / 2
    ph = (TOP - BOT) / (2 + HSP)                              # one row height (frac)
    fig.text(xc, TOP - ph - 0.052, xlabels[0], ha='center', va='top', fontsize=9.5)
    fig.text(xc, BOT - 0.052, xlabels[1], ha='center', va='top', fontsize=9.5)
    leg = [Line2D([0], [0], color=SHAC_BLUE, lw=1.7, label='SHAC'),
           Line2D([0], [0], color=PPO_RED, lw=1.5, label='PPO'),
           Line2D([0], [0], color='0.45', lw=1.0, ls=':', label='Held-out test')]
    if ra:
        fig.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, 0.985),
                   ncol=3, fontsize=9.0, handlelength=1.4,
                   columnspacing=1.0, handletextpad=0.4)
    else:
        fig.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=3, fontsize=9.5, handlelength=1.8, columnspacing=1.5)
    fig.savefig(os.path.join(OUT, fname), format='pdf', dpi=300)
    fig.savefig(os.path.join(OUT, fname.replace('.pdf', '_preview.png')),
                format='png', dpi=150)
    plt.close(fig)
    print(f'wrote {fname} (figsize width {fig_w:.2f}in)')


# ---- cylinder ----
def cyl_dirs(c, algo):
    return [os.path.join(ROOT, 'runs/cylinder', f'{algo}_re{c["Re"]}_s{s}') for s in SEEDS]
cyl_cases = [
    {'Re': 100, 'grid': '768x144',  'title': 'Re$=100$',  'ho': {'shac': 7.7, 'ppo': 6.0}},
    {'Re': 200, 'grid': '768x144',  'title': 'Re$=200$',  'ho': {'shac': 23.1, 'ppo': 15.0}},
    {'Re': 400, 'grid': '1280x240', 'title': 'Re$=400$',  'ho': {'shac': 29.8, 'ppo': 20.4}},
    {'Re': 500, 'grid': '1280x240', 'title': 'Re$=500$',  'ho': {'shac': 30.6, 'ppo': 29.0}},
]
make_figure(cyl_cases, 'cd_baseline', 'cd_mean', cyl_dirs,
            ['Wall-clock time (min)', 'Environment interactions ($\\times10^6$)'],
            'fig03_cylinder_training.pdf')

# ---- Rayleigh-Benard ----
def rb_dirs(c, algo):
    return [os.path.join(ROOT, 'runs/rayleigh_benard', f'{algo}_ra{c["ra"]}_s{s}') for s in SEEDS]
rb_cases = [
    {'ra': '1e4', 'title': 'Ra$=10^4$', 'ho': {'shac': 26.1, 'ppo': 8.6}},
    {'ra': '1e5', 'title': 'Ra$=10^5$', 'ho': {'shac': 22.7, 'ppo': -0.2}},
    {'ra': '1e6', 'title': 'Ra$=10^6$', 'ho': {'shac': 10.5, 'ppo': -6.8}},
]
make_figure(rb_cases, 'nu_baseline', 'nu_mean', rb_dirs,
            ['Wall-clock time (min)', 'Environment interactions ($\\times10^6$)'],
            'fig04_rayleigh_benard_training.pdf', ra=True)
print('done')
