"""
Post-processing utilities for the Lid-Driven Cavity case.

Ghia et al. (1982) reference data is embedded directly so no external
files are needed.

Usage
-----
    from jax_cfd.viz.cavity import plot_centerline_profiles, print_summary

    # Print vortex centre and extrema vs. Ghia
    print_summary('output/run.h5', Re=1000)

    # Plot u(y) and v(x) along centrelines
    plot_centerline_profiles('output/run.h5', Re=1000)
"""

from __future__ import annotations

import numpy as np

from jax_cfd.io.reader import load_grid, load_snapshot, list_steps

# ── Ghia et al. (1982) tabulated reference data ─────────────────────────────
# u-velocity along the vertical centreline x = 0.5L  (17 points)
# v-velocity along the horizontal centreline y = 0.5L (17 points)
# All values non-dimensionalised by U_lid = 1.

_GHIA: dict = {
    100: {
        'u_cl': {  # u(y) at x = 0.5
            'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
                  0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
                  0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
                  0.0547, 0.0000],
            'u': [1.0000, 0.84123, 0.78871, 0.73722, 0.68717,
                  0.23151, 0.00332, -0.13641, -0.20581, -0.21090,
                  -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                  -0.03717, 0.0000],
        },
        'v_cl': {  # v(x) at y = 0.5
            'x': [1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
                  0.9063, 0.8594, 0.8047, 0.5000, 0.2344,
                  0.2266, 0.1563, 0.0938, 0.0781, 0.0703,
                  0.0625, 0.0000],
            'v': [0.0000, -0.05906, -0.07391, -0.08864, -0.10313,
                  -0.16914, -0.22445, -0.24533, 0.05454, 0.17527,
                  0.17507, 0.16077, 0.12085, 0.10890, 0.10091,
                  0.09233, 0.0000],
        },
    },
    400: {
        'u_cl': {
            'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
                  0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
                  0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
                  0.0547, 0.0000],
            'u': [1.0000, 0.75837, 0.68439, 0.61756, 0.55892,
                  0.29093, 0.16256, 0.02135, -0.11477, -0.17119,
                  -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
                  -0.08186, 0.0000],
        },
        'v_cl': {
            'x': [1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
                  0.9063, 0.8594, 0.8047, 0.5000, 0.2344,
                  0.2266, 0.1563, 0.0938, 0.0781, 0.0703,
                  0.0625, 0.0000],
            'v': [0.0000, -0.12146, -0.15663, -0.19254, -0.22847,
                  -0.23827, -0.44993, -0.38598, 0.05186, 0.30174,
                  0.30203, 0.28124, 0.22965, 0.20920, 0.19713,
                  0.18360, 0.0000],
        },
    },
    1000: {
        'u_cl': {
            'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
                  0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
                  0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
                  0.0547, 0.0000],
            'u': [1.0000, 0.65928, 0.57492, 0.51117, 0.46604,
                  0.33304, 0.18719, 0.05702, -0.06080, -0.10648,
                  -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
                  -0.18109, 0.0000],
        },
        'v_cl': {
            'x': [1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
                  0.9063, 0.8594, 0.8047, 0.5000, 0.2344,
                  0.2266, 0.1563, 0.0938, 0.0781, 0.0703,
                  0.0625, 0.0000],
            'v': [0.0000, -0.21388, -0.27669, -0.33714, -0.39188,
                  -0.51550, -0.42665, -0.31966, 0.02526, 0.32235,
                  0.33075, 0.37095, 0.32627, 0.30353, 0.29012,
                  0.27485, 0.0000],
        },
    },
}

# Vortex centre locations (Ghia 1982, Table 1)
_VORTEX_CENTRES: dict = {
    100:  {'x': 0.6172, 'y': 0.7344},
    400:  {'x': 0.5547, 'y': 0.6055},
    1000: {'x': 0.5308, 'y': 0.5652},
}


# ── Helper: extract centreline profiles from HDF5 snapshot ──────────────────

def _get_centreline(h5_path: str, step: int | None = None):
    """Return (xc, yc, u_2d, v_2d) from an HDF5 snapshot."""
    grid = load_grid(h5_path)
    xc = np.array(grid['xc'])   # shape (Nx,)
    yc = np.array(grid['yc'])   # shape (Ny,)

    if step is None:
        steps = list_steps(h5_path)
        step = steps[-1]

    snap = load_snapshot(h5_path, step=step)
    u = np.array(snap['u'])   # shape (Nx+1, Ny)
    v = np.array(snap['v'])   # shape (Nx, Ny+1)

    # Interpolate u to cell centres: average adjacent x-faces
    u_cc = 0.5 * (u[:-1, :] + u[1:, :])   # (Nx, Ny)
    # Interpolate v to cell centres: average adjacent y-faces
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])   # (Nx, Ny)

    return xc, yc, u_cc, v_cc


# ── Public API ───────────────────────────────────────────────────────────────

def plot_centerline_profiles(
    h5_path: str,
    Re: int = 1000,
    step: int | None = None,
    save: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot u(y) along the vertical centreline (x = 0.5L) and
    v(x) along the horizontal centreline (y = 0.5L), overlaid
    with Ghia et al. (1982) reference data.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 output file.
    Re : int
        Reynolds number — used to select the Ghia reference set.
        Supported: 100, 400, 1000.
    step : int or None
        Snapshot step index.  None → last saved snapshot.
    save : str or None
        If given, save the figure to this path (e.g. 'cavity.png').
    show : bool
        Call plt.show() if True.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.  pip install matplotlib")

    xc, yc, u_cc, v_cc = _get_centreline(h5_path, step=step)
    Nx, Ny = len(xc), len(yc)

    # Nearest-index centreline (x ≈ 0.5 and y ≈ 0.5)
    ix = int(np.argmin(np.abs(xc - 0.5)))
    iy = int(np.argmin(np.abs(yc - 0.5)))

    u_prof = u_cc[ix, :]   # u(y) at x = xc[ix]
    v_prof = v_cc[:, iy]   # v(x) at y = yc[iy]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Lid-Driven Cavity  Re = {Re}", fontsize=13)

    # Left: u(y) at x = 0.5
    ax = axes[0]
    ax.plot(u_prof, yc, lw=1.5, label='JAX-CFD')
    if Re in _GHIA:
        g = _GHIA[Re]['u_cl']
        ax.plot(g['u'], g['y'], 'o', ms=5, mfc='none', label='Ghia 1982')
    ax.set_xlabel('u / U_lid')
    ax.set_ylabel('y / L')
    ax.set_title('u-profile  (x = 0.5)')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)

    # Right: v(x) at y = 0.5
    ax = axes[1]
    ax.plot(xc, v_prof, lw=1.5, label='JAX-CFD')
    if Re in _GHIA:
        g = _GHIA[Re]['v_cl']
        ax.plot(g['x'], g['v'], 'o', ms=5, mfc='none', label='Ghia 1982')
    ax.set_xlabel('x / L')
    ax.set_ylabel('v / U_lid')
    ax.set_title('v-profile  (y = 0.5)')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(h5_path: str, Re: int = 1000) -> None:
    """
    Print primary-vortex centre location and velocity extrema,
    compared with Ghia et al. (1982).

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 output file.
    Re : int
        Reynolds number.  Supported: 100, 400, 1000.
    """
    xc, yc, u_cc, v_cc = _get_centreline(h5_path)

    # Vertical centreline u(y) at x ≈ 0.5
    ix = int(np.argmin(np.abs(xc - 0.5)))
    u_prof = u_cc[ix, :]
    u_min = float(np.min(u_prof))
    y_umin = float(yc[np.argmin(u_prof)])

    # Horizontal centreline v(x) at y ≈ 0.5
    iy = int(np.argmin(np.abs(yc - 0.5)))
    v_prof = v_cc[:, iy]
    v_min = float(np.min(v_prof))
    x_vmin = float(xc[np.argmin(v_prof)])

    # Approximate vortex centre via stream function ψ.
    # For 2D incompressible: ∂ψ/∂y = u  →  ψ(x,y) = ∫₀ʸ u(x,y') dy'
    # Clockwise primary vortex corresponds to the global minimum of ψ.
    psi = np.zeros_like(u_cc)
    for j in range(1, len(yc)):
        psi[:, j] = psi[:, j - 1] + 0.5 * (u_cc[:, j - 1] + u_cc[:, j]) * (yc[j] - yc[j - 1])
    idx = np.unravel_index(np.argmin(psi), psi.shape)
    x_vort = float(xc[idx[0]])
    y_vort = float(yc[idx[1]])

    # Reference
    ref_vc = _VORTEX_CENTRES.get(Re, {})
    ref_u  = _GHIA.get(Re, {}).get('u_cl', {})
    ref_v  = _GHIA.get(Re, {}).get('v_cl', {})

    print()
    print(f"{'─'*56}")
    print(f"  Lid-Driven Cavity  Re = {Re}  (Ghia et al. 1982)")
    print(f"{'─'*56}")
    print(f"  {'Quantity':<30}  {'JAX-CFD':>8}  {'Ghia':>8}")
    print(f"  {'─'*50}")

    # Vortex centre x
    ref_x = ref_vc.get('x', float('nan'))
    print(f"  {'Vortex centre  x':<30}  {x_vort:8.4f}  {ref_x:8.4f}")

    # Vortex centre y
    ref_y = ref_vc.get('y', float('nan'))
    print(f"  {'Vortex centre  y':<30}  {y_vort:8.4f}  {ref_y:8.4f}")

    # u_min along vertical centreline
    if ref_u:
        ref_umin = min(ref_u['u'])
        ref_y_umin = ref_u['y'][ref_u['u'].index(ref_umin)]
    else:
        ref_umin = ref_y_umin = float('nan')
    print(f"  {'u_min  (x=0.5)':<30}  {u_min:8.4f}  {ref_umin:8.4f}")
    print(f"  {'  at y':<30}  {y_umin:8.4f}  {ref_y_umin:8.4f}")

    # v_min along horizontal centreline
    if ref_v:
        ref_vmin = min(ref_v['v'])
        ref_x_vmin = ref_v['x'][ref_v['v'].index(ref_vmin)]
    else:
        ref_vmin = ref_x_vmin = float('nan')
    print(f"  {'v_min  (y=0.5)':<30}  {v_min:8.4f}  {ref_vmin:8.4f}")
    print(f"  {'  at x':<30}  {x_vmin:8.4f}  {ref_x_vmin:8.4f}")

    print(f"{'─'*56}")
    print()
