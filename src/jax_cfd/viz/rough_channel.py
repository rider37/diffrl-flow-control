"""
Post-processing for 3D rough channel flow (hemisphere roughness DNS).

Compares the plane-averaged velocity profile u(y) with smooth Poiseuille,
reports drag increase, and visualises vortical structures around roughness.

Usage
-----
    from jax_cfd.viz.rough_channel import print_summary, plot_mean_profile
    from jax_cfd.viz.rough_channel import plot_roughness_slice

    print_summary('output/run.h5')
    plot_mean_profile('output/run.h5', save='profile.png')
    plot_roughness_slice('output/run.h5', save='slice.png')
"""

from __future__ import annotations

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _analytic_poiseuille(y: np.ndarray, H: float = 2.0) -> np.ndarray:
    """u(y) = y*(H-y) / (H/2)^2, normalised so u_max = 1."""
    delta = H / 2.0
    return y * (H - y) / delta ** 2


def _load_last_snapshot_3d(h5_path: str):
    """Load last 3D velocity snapshot and grid coordinates."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['fields'].keys())
        if not keys:
            raise RuntimeError(f"No snapshots found in {h5_path}")
        grp = f['fields'][keys[-1]]
        u = grp['u'][:]       # (Nx+1, Ny, Nz)
        v = grp['v'][:]       # (Nx, Ny+1, Nz)
        w = grp['w'][:]       # (Nx, Ny, Nz+1)
        p = grp['p'][:]       # (Nx, Ny, Nz)
        xc = f['grid']['xc'][:]
        yc = f['grid']['yc'][:]
        zc = f['grid']['zc'][:]
        t = float(grp.attrs['t'])
        step = int(grp.attrs['step'])
    return u, v, w, p, xc, yc, zc, t, step


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(h5_path: str, H: float = 2.0) -> None:
    """Print comparison between rough-channel and smooth Poiseuille profiles."""
    u, v, w, p, xc, yc, zc, t, step = _load_last_snapshot_3d(h5_path)

    # Cell-centred u, plane-averaged
    u_cc = 0.5 * (u[:-1, :, :] + u[1:, :, :])   # (Nx, Ny, Nz)
    u_mean = np.mean(u_cc, axis=(0, 2))            # (Ny,)

    u_analytic = _analytic_poiseuille(yc, H)

    u_max_sim = float(np.max(u_mean))
    u_bulk_sim = float(np.mean(u_mean))
    u_bulk_smooth = float(np.mean(u_analytic))
    u_max_smooth = 1.0
    linf = float(np.max(np.abs(u_mean - u_analytic)))

    drag_increase = (1.0 - u_bulk_sim / u_bulk_smooth) * 100.0 if u_bulk_smooth > 0 else 0.0

    print()
    print("── 3D Rough Channel Summary ─────────────────────────────")
    print(f"  Snapshot     : step {step},  t = {t:.3f}")
    print(f"  Grid         : {u.shape[0]-1} x {u.shape[1]} x {u.shape[2]}")
    print(f"  u_max (rough) : {u_max_sim:.5f}  (smooth: {u_max_smooth:.5f})")
    print(f"  u_bulk (rough): {u_bulk_sim:.5f}  (smooth: {u_bulk_smooth:.5f})")
    print(f"  L∞ vs smooth : {linf:.2e}")
    print(f"  Drag increase: {drag_increase:.1f}%")
    print("─────────────────────────────────────────────────────────")
    print()


# ── Mean velocity profile ─────────────────────────────────────────────────────

def plot_mean_profile(
    h5_path: str,
    H: float = 2.0,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Plot plane-averaged u(y) vs smooth Poiseuille reference."""
    import matplotlib.pyplot as plt

    u, v, w, p, xc, yc, zc, t, step = _load_last_snapshot_3d(h5_path)

    u_cc = 0.5 * (u[:-1, :, :] + u[1:, :, :])
    u_mean = np.mean(u_cc, axis=(0, 2))

    y_ref = np.linspace(0, H, 200)
    u_ref = _analytic_poiseuille(y_ref, H)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: velocity profiles
    ax1.plot(u_mean, yc, 'b-', lw=2, label='Rough channel')
    ax1.plot(u_ref, y_ref, 'r--', lw=1.5, label='Smooth Poiseuille')
    ax1.set_xlabel('$\\langle u \\rangle_{x,z}$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Mean velocity profile  (t = {t:.1f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: velocity deficit
    u_analytic = _analytic_poiseuille(yc, H)
    deficit = u_analytic - u_mean
    ax2.plot(deficit, yc, 'k-', lw=2)
    ax2.axvline(0, color='gray', ls=':', alpha=0.5)
    ax2.set_xlabel('$u_{smooth} - u_{rough}$')
    ax2.set_ylabel('$y$')
    ax2.set_title('Velocity deficit (roughness effect)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ── Roughness vortex slice ────────────────────────────────────────────────────

def plot_roughness_slice(
    h5_path: str,
    z_slice: float | None = None,
    roughness_k: float = 0.1,
    roughness_centers_x: list | np.ndarray | None = None,
    y_max_plot: float = 0.5,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Plot x-y vorticity slice through a hemisphere centre.

    Parameters
    ----------
    roughness_centers_x : list of float, optional
        x-coordinates of roughness element centres for overlay drawing.
        If None, no hemisphere outlines are drawn.
    y_max_plot : float
        Upper y-limit for the zoomed view (default 0.5).
    """
    import matplotlib.pyplot as plt

    u, v, w, p, xc, yc, zc, t, step = _load_last_snapshot_3d(h5_path)

    # Default: slice at first hemisphere z-centre
    if z_slice is None:
        z_slice = zc[len(zc) // 5]  # approximate first roughness row z

    # Find nearest z-index
    iz = int(np.argmin(np.abs(zc - z_slice)))

    # Compute z-vorticity on the slice: omega_z = dv/dx - du/dy
    u_cc = 0.5 * (u[:-1, :, iz] + u[1:, :, iz])  # (Nx, Ny)
    v_cc = 0.5 * (v[:, :-1, iz] + v[:, 1:, iz])  # (Nx, Ny)

    dy = np.diff(yc)
    dx = np.diff(xc)

    dudy = np.zeros_like(u_cc)
    dudy[:, 1:-1] = (u_cc[:, 2:] - u_cc[:, :-2]) / (dy[:-1] + dy[1:])[np.newaxis, :]
    dvdx = np.zeros_like(v_cc)
    dvdx[1:-1, :] = (v_cc[2:, :] - v_cc[:-2, :]) / (dx[:-1] + dx[1:])[:, np.newaxis]

    omega_z = dvdx - dudy

    fig, ax = plt.subplots(figsize=(12, 4))
    vmax = max(float(np.percentile(np.abs(omega_z), 99)), 0.1)
    im = ax.pcolormesh(xc, yc, omega_z.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'$\\omega_z$ slice at z = {zc[iz]:.2f}  (t = {t:.1f})')
    ax.set_aspect('equal')
    ax.set_ylim(0, y_max_plot)

    # Draw hemisphere outlines if centres provided
    if roughness_centers_x is not None:
        theta = np.linspace(0, np.pi, 50)
        for cx in roughness_centers_x:
            ax.plot(cx + roughness_k * np.cos(theta),
                    roughness_k * np.sin(theta),
                    'k-', lw=1, alpha=0.7)

    plt.colorbar(im, ax=ax, label='$\\omega_z$', shrink=0.8)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
