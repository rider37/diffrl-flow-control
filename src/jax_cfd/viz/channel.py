"""
Post-processing for plane Poiseuille (channel) flow.

Analytic reference
------------------
For body-force F_x = 2/Re driving a channel y ∈ [0, H=2] with Re = U_max·δ/ν:
    u(y) = y · (2 - y)          u_max = 1 at y = 1
    u_bulk = 2/3
    Re_b = u_bulk · δ / ν = Re / 1.5  (≈ Re_b = 67 for Re = 100)

Usage
-----
    from jax_cfd.viz.channel import plot_poiseuille_profile, print_summary

    print_summary('output/run.h5')
    plot_poiseuille_profile('output/run.h5', save='profile.png')
"""

from __future__ import annotations

import numpy as np


# ── Analytic Poiseuille reference ──────────────────────────────────────────

def _analytic_poiseuille(y: np.ndarray, H: float = 2.0) -> np.ndarray:
    """u(y) = y·(H−y) / (H/2)²  normalised so u_max = 1 at y = H/2."""
    delta = H / 2.0
    return y * (H - y) / delta ** 2


def _load_last_snapshot(h5_path: str):
    """Return (grid_y, u_2d, grid_xc) from the last saved field snapshot."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['fields'].keys())
        if not keys:
            raise RuntimeError(f"No snapshots found in {h5_path}")
        grp  = f['fields'][keys[-1]]
        u    = grp['u'][:]           # (Nx+1, Ny)
        yv   = f['grid']['yv'][:]    # v-face y coords (Ny+1,)
        xc   = f['grid']['xc'][:]    # cell-centre x
        yc   = f['grid']['yc'][:]    # cell-centre y
        t    = float(grp.attrs['t'])
        step = int(grp.attrs['step'])
    return u, xc, yc, yv, t, step


# ── Main functions ──────────────────────────────────────────────────────────

def plot_poiseuille_profile(
    h5_path: str,
    H: float = 2.0,
    save: str | None = None,
    show: bool = True,
) -> None:
    """
    Compare the streamwise-averaged velocity profile u(y) with the analytic
    Poiseuille solution  u(y) = y·(H−y) / (H/2)².

    The profile is extracted from the last snapshot in *h5_path*.

    Parameters
    ----------
    h5_path : path to HDF5 output file
    H       : full channel height (default 2.0)
    save    : if given, save figure to this path
    show    : if True, call plt.show()
    """
    import matplotlib.pyplot as plt

    u, xc, yc, yv, t, step = _load_last_snapshot(h5_path)

    # u has shape (Nx+1, Ny); average the interior u-faces over x
    u_int  = 0.5 * (u[:-1, :] + u[1:, :])   # cell-centred u, shape (Nx, Ny)
    u_mean = np.mean(u_int, axis=0)           # streamwise average, shape (Ny,)

    y_analytic = np.linspace(0, H, 200)
    u_analytic = _analytic_poiseuille(y_analytic, H)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(u_mean,    yc,         'b-',  lw=2, label='Simulation')
    ax.plot(u_analytic, y_analytic, 'r--', lw=1.5, label='Poiseuille (analytic)')
    ax.set_xlabel('u')
    ax.set_ylabel('y')
    ax.set_title(f'Poiseuille profile  (t = {t:.2f}, step {step})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(h5_path: str, H: float = 2.0) -> None:
    """
    Print a brief comparison between the simulated and analytic Poiseuille profiles.

    Reports u_max, u_bulk, and the L∞ error versus the analytic solution.
    """
    u, xc, yc, yv, t, step = _load_last_snapshot(h5_path)

    u_int  = 0.5 * (u[:-1, :] + u[1:, :])
    u_mean = np.mean(u_int, axis=0)           # shape (Ny,)

    u_analytic = _analytic_poiseuille(yc, H)

    u_max   = float(np.max(u_mean))
    u_bulk  = float(np.mean(u_mean))
    linf    = float(np.max(np.abs(u_mean - u_analytic)))
    l2      = float(np.sqrt(np.mean((u_mean - u_analytic) ** 2)))

    print()
    print("── Channel Poiseuille Summary ───────────────────────────")
    print(f"  Snapshot   : step {step},  t = {t:.3f}")
    print(f"  u_max      : {u_max:.5f}   (analytic: 1.00000)")
    print(f"  u_bulk     : {u_bulk:.5f}   (analytic: {2/3:.5f})")
    print(f"  L∞ error   : {linf:.2e}")
    print(f"  L2 error   : {l2:.2e}")
    print("─────────────────────────────────────────────────────────")
    print()
