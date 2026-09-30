"""Figs. 1 and 2: the two control problems.

Fig. 1: cylinder channel with the probe layout and a vorticity snapshot, and the
jet pair.  Fig. 2: Rayleigh--Benard cell with the heater segments and a
temperature snapshot.  The snapshots are results/figure_data/fig01_background.npz
and fig02_background.npz, flow fields saved by the training scripts (snapshots/).  Writes figures/fig01_cylinder_setup.pdf and
figures/fig02_rayleigh_benard_setup.pdf.
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
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 8, 'axes.linewidth': 0.5,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
HERE = os.path.dirname(os.path.abspath(__file__))
FS, FSS = 10.0, 9.0                      # label / small-label sizes


def A(ax, p0, p1, lw=0.9, ms=7, color='k'):
    ax.annotate('', xy=p1, xytext=p0, annotation_clip=False, zorder=9,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0))


def cyl_probes(cx, cy, R):
    px, py = [], []
    for xi in np.linspace(cx + 0.75, cx + 4.5, 9):
        for yi in np.linspace(cy - 1.5, cy + 1.5, 7):
            px.append(xi); py.append(yi)
    for xi in np.linspace(cx - 0.25, cx + 0.5, 4):
        for yi in np.r_[np.linspace(cy - 1.5, cy - 1.0, 2),
                        np.linspace(cy + 1.0, cy + 1.5, 2)]:
            px.append(xi); py.append(yi)
    for rr in (R + 0.2, R + 0.5):
        for k in range(36):
            th = 2 * np.pi * k / 36
            px.append(cx + rr * np.cos(th)); py.append(cy + rr * np.sin(th))
    return np.array(px), np.array(py)


def cylinder():
    d = np.load(os.path.join(ROOT, 'results/figure_data/fig01_background.npz'))
    xc, yc, w = d['xc'], d['yc'], np.asarray(d['vorticity'])
    cx, R, D = 2.0, 0.5, 1.0
    cy = 0.5 * (yc[0] + yc[-1])
    X, Y = (xc - cx) / D, (yc - cy) / D
    yw = (yc[-1] - cy) / D
    x0, x1 = -2.1, 9.2

    # laid out at the final printed width (\columnwidth = 390 pt = 5.40 in)
    LM, AW = 0.46, 3.05                       # ylabel margin / field width
    AH = AW * (2 * (yw + 0.13)) / (x1 - x0)   # field height, equal aspect
    BS = AH * 0.95                            # side of the actuation panel
    CBX, BXX = LM + AW + 0.09, LM + AW + 0.09 + 0.075 + 0.50
    fw, fh = BXX + BS + 0.04, 0.38 + AH + 0.20
    ax = plt.figure(figsize=(fw, fh)).add_axes(
        [LM / fw, 0.38 / fh, AW / fw, AH / fh])
    fig = ax.figure

    ax.pcolormesh(X, Y, w.T, cmap='RdBu_r', vmin=-5, vmax=5,
                  shading='auto', rasterized=True, zorder=0)
    pcm = ax.collections[0]
    px, py = cyl_probes(cx, cy, R)
    ax.scatter((px - cx) / D, (py - cy) / D, s=3.0, c='k', linewidths=0,
               zorder=7)
    ax.add_patch(Circle((0, 0), R, facecolor='0.85', edgecolor='k',
                        lw=0.9, zorder=5))
    for ang in (90, 270):                     # actuator surfaces
        ax.add_patch(Wedge((0, 0), R + 0.03, ang - 12, ang + 12, width=0.13,
                           facecolor='k', edgecolor='none', zorder=6))
    for s in (+1, -1):                        # channel walls
        ax.add_patch(Rectangle((x0, s * yw), x1 - x0, s * 0.11,
                               facecolor='0.75', edgecolor='none',
                               hatch='////', zorder=4))
        ax.add_line(Line2D([x0, x1], [s * yw, s * yw], color='k', lw=0.9,
                           zorder=5))
    A(ax, (x0 + 0.15, 0), (x0 + 0.95, 0), lw=0.9)
    ax.text(x0 + 0.55, 0.16, r'$U_\infty$', fontsize=FS, ha='center',
            va='bottom')
    ax.text(4.55, -yw + 0.14, 'velocity probes', fontsize=FSS, color='0.15',
            ha='left', va='bottom', zorder=9)
    ax.add_patch(Rectangle((-1.0, -1.0), 2.0, 2.0, facecolor='none',
                           edgecolor='0.25', lw=0.55, ls=(0, (2.5, 2)),
                           zorder=8))
    ax.set_xlim(x0, x1); ax.set_ylim(-yw - 0.13, yw + 0.13)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/D$', fontsize=FS, labelpad=1)
    ax.set_ylabel(r'$y/D$', fontsize=FS, labelpad=1)
    ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=FSS, length=2.2, width=0.5, pad=1.5)
    ax.text(0.0, 1.045, '(a)', transform=ax.transAxes, fontsize=FS,
            ha='left', va='bottom')

    cax = fig.add_axes([CBX / fw, 0.38 / fh, 0.075 / fw, AH / fh])
    cb = fig.colorbar(pcm, cax=cax, ticks=[-5, 0, 5])
    cb.set_label(r'$\omega_z D/U_\infty$', fontsize=FS, labelpad=2)
    cb.ax.tick_params(labelsize=FSS, length=2.0, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)

    bx = fig.add_axes([BXX / fw, 0.38 / fh, BS / fw, BS / fh])
    bx.set_xlim(-1.45, 1.45); bx.set_ylim(-1.45, 1.45)
    bx.set_aspect('equal'); bx.axis('off')
    bx.text(0.0, 1.045, '(b)', transform=bx.transAxes, fontsize=FS,
            ha='left', va='bottom')
    bx.add_patch(Circle((0, 0), R, facecolor='0.85', edgecolor='k',
                        lw=0.9, zorder=3))
    for ang in (90, 270):
        bx.add_patch(Wedge((0, 0), R + 0.03, ang - 12, ang + 12, width=0.13,
                           facecolor='k', edgecolor='none', zorder=4))
    A(bx, (0, 0.58), (0, 1.10), lw=1.1, ms=8)
    A(bx, (0, -1.10), (0, -0.58), lw=1.1, ms=8)
    bx.text(0.10, 0.86, r'$+Q$', fontsize=FS, ha='left', va='center')
    bx.text(0.10, -0.86, r'$-Q$', fontsize=FS, ha='left', va='center')
    bx.text(-0.10, 0.86, 'blowing', fontsize=FSS, ha='right', va='center',
            color='0.3')
    bx.text(-0.10, -0.86, 'suction', fontsize=FSS, ha='right', va='center',
            color='0.3')
    A(bx, (-1.35, 0), (-0.62, 0), lw=0.9)
    bx.text(-0.95, 0.10, r'$U_\infty$', fontsize=FSS, ha='center',
            va='bottom')
    A(bx, (0.62, 0), (1.35, 0), lw=1.0)
    bx.text(0.95, 0.10, r'$C_d$', fontsize=FSS, ha='center', va='bottom')
    bx.text(1.40, -1.38, r'$a>0$', fontsize=FSS, ha='right', va='bottom',
            color='0.3')

    out = os.path.join(HERE, 'fig01_cylinder_setup.pdf')
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.pdf', '_preview.png'), dpi=200)
    plt.close(fig); print('wrote', out)


def rb(cbar='bottom'):
    """cbar: 'right' (vertical, outside keys), 'top' or 'bottom'
    (horizontal bar hugging the field, left-aligned)."""
    d = np.load(os.path.join(ROOT, 'results/figure_data/fig02_background.npz'))
    xc, yc, T = d['xc'], d['yc'], np.asarray(d['T'])
    Lx = 2 * np.pi
    Ly = yc[-1] + yc[0]
    nseg, hs, tw = 12, 0.16, 0.11
    TGT, LMAR, RMAR = 5.38, 0.46, 0.94    # printed width / ylabel / key column
    LPAD = RMAR - LMAR                     # left pad centres the field
    aw = TGT - 2 * RMAR
    sc = aw / Lx
    ah = (Ly + tw + hs) * sc
    CBH = 0.075                                   # bar thickness (in)
    top_pad = 0.54 if cbar == 'top' else 0.14
    bot_pad = 0.34 + (0.42 if cbar == 'bottom' else 0.0)
    fw = LMAR + LPAD + aw + RMAR + (0.16 + CBH + 0.46 if cbar == 'right' else 0.0)
    fh = bot_pad + ah + top_pad
    ax = plt.figure(figsize=(fw, fh)).add_axes(
        [(LMAR + LPAD) / fw, bot_pad / fh, aw / fw, ah / fh])
    fig = ax.figure

    ax.pcolormesh(xc, yc, T.T, cmap='RdBu_r', vmin=0, vmax=1,
                  shading='auto', rasterized=True, zorder=0)
    pcm = ax.collections[0]
    npx, npy = 48, 8
    PX, PY = np.meshgrid(np.linspace(Lx / (2 * npx), Lx - Lx / (2 * npx), npx),
                         np.linspace(Ly / (2 * npy), Ly - Ly / (2 * npy), npy),
                         indexing='ij')
    ax.scatter(PX.ravel(), PY.ravel(), s=1.6, c='k', linewidths=0, zorder=6)

    ax.add_patch(Rectangle((0, Ly), Lx, tw, facecolor='0.75',
                           edgecolor='none', hatch='////', zorder=4))
    ax.add_line(Line2D([0, Lx], [Ly, Ly], color='k', lw=0.9, zorder=5))
    xs = np.linspace(0, Lx, nseg + 1)
    for i in range(nseg):
        ax.add_patch(Rectangle((xs[i], -hs), xs[i + 1] - xs[i], hs,
                               facecolor='0.88', edgecolor='k', lw=0.5,
                               zorder=5))
    ax.add_line(Line2D([0, Lx], [0, 0], color='k', lw=0.9, zorder=6))
    for xv in (0, Lx):
        ax.add_line(Line2D([xv, xv], [0, Ly], color='k', lw=0.7,
                           ls=(0, (2.5, 2)), zorder=5))

    kx, tx = Lx + 0.32, Lx + 0.44
    lead = dict(color='k', lw=0.6, clip_on=False, zorder=8)
    ax.add_line(Line2D([Lx + 0.03, kx], [Ly + tw / 2, Ly + tw / 2], **lead))
    ax.text(tx, Ly + tw / 2, 'cold wall\n$T=0$', fontsize=FSS,
            ha='left', va='center', linespacing=1.35)
    # gravity: arrow in the right key column, centred on the cell height
    A(ax, (kx, 1.38), (kx, 0.62), lw=1.3, ms=11)
    ax.text(tx, 1.00, r'$g$', fontsize=FS + 1, ha='left', va='center')
    ax.add_line(Line2D([Lx + 0.03, kx], [-hs / 2, -hs / 2], **lead))
    ax.text(tx, -hs / 2, 'hot wall\n12 heaters',
            fontsize=FSS, ha='left', va='center', linespacing=1.35)

    ax.set_xlim(0, Lx); ax.set_ylim(-hs, Ly + tw)
    ax.set_aspect('equal')
    ax.set_xticks([0, np.pi, Lx]); ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel(r'$x$', fontsize=FS, labelpad=1)
    ax.set_ylabel(r'$y$', fontsize=FS, labelpad=6, rotation=0, va='center')
    ax.tick_params(labelsize=FSS, length=2.2, width=0.5, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    if cbar == 'right':
        cax = fig.add_axes([(LMAR + LPAD + aw + RMAR + 0.16) / fw,
                            (bot_pad + 0.22) / fh, CBH / fw, (ah - 0.44) / fh])
        cb = fig.colorbar(pcm, cax=cax, ticks=[0, 0.5, 1])
        cb.set_label(r'$T$', fontsize=FS, labelpad=2)
    else:
        bw = 0.45 * aw
        y = (bot_pad + ah + 0.20) if cbar == 'top' else (bot_pad - 0.46)
        bx0 = (LMAR + LPAD) + (0.0 if cbar == 'top' else 0.5 * (aw - bw))
        cax = fig.add_axes([bx0 / fw, y / fh, bw / fw, CBH / fh])
        cb = fig.colorbar(pcm, cax=cax, orientation='horizontal',
                          ticks=[0, 0.5, 1])
        if cbar == 'top':
            cax.xaxis.set_ticks_position('top')
            cax.xaxis.set_label_position('top')
        cb.set_label(r'temperature $T$', fontsize=FSS, labelpad=2)
    cb.ax.tick_params(labelsize=FSS, length=2.0, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    if cbar == 'bottom':                       # probe key beside the colourbar
        fig.legend(handles=[Line2D([0], [0], marker='o', ls='none', color='k',
                                   markersize=2.4,
                                   label=r'probes ($48{\times}8$)')],
                   loc='center left',
                   bbox_to_anchor=((bx0 + bw + 0.28) / fw, (y + CBH / 2) / fh),
                   bbox_transform=fig.transFigure, fontsize=FSS,
                   frameon=False, handletextpad=0.5)

    suf = {'bottom': '', 'top': '_cbtop', 'right': '_cbright'}[cbar]
    out = os.path.join(HERE, 'fig02_rayleigh_benard_setup%s.pdf' % suf)
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.pdf', '_preview.png'), dpi=200)
    plt.close(fig); print('wrote', out)


if __name__ == '__main__':
    cylinder(); rb()
