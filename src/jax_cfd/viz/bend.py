"""
Post-processing for 2D 90-degree bend (Dean flow).

Validates centrifugal pressure gradient, velocity profile skewing
toward the inner wall, and overall flow development through the bend.

Usage
-----
    from jax_cfd.viz.bend import print_summary, plot_velocity_field
    from jax_cfd.viz.bend import plot_pressure_field, plot_radial_profiles

    print_summary('output/run.h5')
    plot_velocity_field('output/run.h5', save='speed.png')
    plot_pressure_field('output/run.h5', save='pressure.png')
    plot_radial_profiles('output/run.h5', save='radial.png')
"""

from __future__ import annotations

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_last_snapshot(h5_path: str, step: int | None = None):
    """Load velocity, pressure, and grid from the last (or given) snapshot."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        xc = f['grid']['xc'][:]
        yc = f['grid']['yc'][:]
        keys = sorted(f['fields'].keys())
        if not keys:
            raise RuntimeError(f"No snapshots in {h5_path}")
        key = keys[-1] if step is None else f'step_{step:07d}'
        grp = f['fields'][key]
        u = grp['u'][:]
        v = grp['v'][:]
        p = grp['p'][:]
        t = float(grp.attrs['t'])
        st = int(grp.attrs['step'])
    return u, v, p, xc, yc, t, st


def _draw_bend_walls(
    ax,
    d: float = 1.0,
    R_c: float = 3.0,
    L_in: float = 5.0,
    L_out: float = 5.0,
    color: str = 'k',
    lw: float = 1.5,
) -> None:
    """Draw the 5 wall segments + domain bottom wall on *ax*."""
    R_i = R_c - d / 2
    R_o = R_c + d / 2
    C = (L_in, d / 2 + R_c)
    x_out = C[0] + R_c
    theta = np.linspace(-np.pi / 2, 0, 100)

    ax.plot([0, L_in], [d, d], color=color, lw=lw)                   # upper inlet
    ax.plot(C[0] + R_i * np.cos(theta), C[1] + R_i * np.sin(theta),
            color=color, lw=lw)                                       # inner arc
    ax.plot(C[0] + R_o * np.cos(theta), C[1] + R_o * np.sin(theta),
            color=color, lw=lw)                                       # outer arc
    ax.plot([x_out - d / 2] * 2, [C[1], C[1] + L_out],
            color=color, lw=lw)                                       # left outlet
    ax.plot([x_out + d / 2] * 2, [C[1], C[1] + L_out],
            color=color, lw=lw)                                       # right outlet
    ax.plot([0, L_in], [0, 0], color=color, lw=lw, ls='--', alpha=0.4)  # domain y=0


def _interpolate_radial(
    u_cc: np.ndarray,
    v_cc: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    C: tuple,
    R_i: float,
    R_o: float,
    theta: float,
    n_pts: int = 60,
):
    """Interpolate velocity along a radial line at angle *theta* in the bend.

    Returns
    -------
    r_arr  : radial position from inner to outer wall (shape n_pts)
    u_tan  : tangential (streamwise) velocity component
    u_rad  : radial velocity component
    """
    from scipy.interpolate import RegularGridInterpolator

    r_arr = np.linspace(R_i, R_o, n_pts)
    x_pts = C[0] + r_arr * np.cos(theta)
    y_pts = C[1] + r_arr * np.sin(theta)

    interp_u = RegularGridInterpolator(
        (xc, yc), u_cc, method='linear', bounds_error=False, fill_value=0.0)
    interp_v = RegularGridInterpolator(
        (xc, yc), v_cc, method='linear', bounds_error=False, fill_value=0.0)

    pts = np.column_stack([x_pts, y_pts])
    u_vals = interp_u(pts)
    v_vals = interp_v(pts)

    # Tangential = perpendicular to radial, in streamwise sense
    e_tan = np.array([-np.sin(theta), np.cos(theta)])
    e_rad = np.array([np.cos(theta), np.sin(theta)])
    u_tan = u_vals * e_tan[0] + v_vals * e_tan[1]
    u_rad = u_vals * e_rad[0] + v_vals * e_rad[1]

    return r_arr, u_tan, u_rad


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(
    h5_path: str,
    Re: float = 200.0,
    d: float = 1.0,
    R_c: float = 3.0,
) -> None:
    """Print Dean number, peak speed, and pressure difference across bend."""
    u, v, p, xc, yc, t, step = _load_last_snapshot(h5_path)

    u_cc = 0.5 * (u[:-1, :] + u[1:, :])
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])
    speed = np.sqrt(u_cc ** 2 + v_cc ** 2)

    De = Re * np.sqrt(d / R_c)

    # Pressure at bend midpoint (θ = -π/4): inner vs outer
    C = (5.0, d / 2 + R_c)
    R_i = R_c - d / 2
    R_o = R_c + d / 2
    th_mid = -np.pi / 4
    x_in = C[0] + R_i * np.cos(th_mid)
    y_in = C[1] + R_i * np.sin(th_mid)
    x_out = C[0] + R_o * np.cos(th_mid)
    y_out = C[1] + R_o * np.sin(th_mid)

    from scipy.interpolate import RegularGridInterpolator
    interp_p = RegularGridInterpolator(
        (xc, yc), p, method='linear', bounds_error=False, fill_value=0.0)
    p_inner = float(interp_p(np.array([[x_in, y_in]]))[0])
    p_outer = float(interp_p(np.array([[x_out, y_out]]))[0])

    print()
    print("── 2D 90° Bend Summary ──────────────────────────────────")
    print(f"  Snapshot     : step {step},  t = {t:.3f}")
    print(f"  Grid         : {u.shape[0]-1} x {u.shape[1]}")
    print(f"  Re           : {Re:.0f}")
    print(f"  De           : {De:.1f}")
    print(f"  Max speed    : {speed.max():.4f}")
    print(f"  Δp (outer−inner) at θ=45°: {p_outer - p_inner:.4f}")
    print("─────────────────────────────────────────────────────────")
    print()


# ── Velocity field ────────────────────────────────────────────────────────────

def plot_velocity_field(
    h5_path: str,
    d: float = 1.0,
    R_c: float = 3.0,
    L_in: float = 5.0,
    L_out: float = 5.0,
    step: int | None = None,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Velocity magnitude contour with streamlines and wall outlines."""
    import matplotlib.pyplot as plt

    u, v, p, xc, yc, t, st = _load_last_snapshot(h5_path, step)
    u_cc = 0.5 * (u[:-1, :] + u[1:, :])
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])
    speed = np.sqrt(u_cc ** 2 + v_cc ** 2)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolormesh(xc, yc, speed.T, cmap='hot_r', vmin=0,
                       vmax=float(np.percentile(speed, 99.5)), shading='auto')

    # Streamlines (subsample for clarity)
    skip = max(1, len(xc) // 80)
    ax.streamplot(xc[::skip], yc[::skip], u_cc[::skip, ::skip].T,
                  v_cc[::skip, ::skip].T, color='steelblue', linewidth=0.5,
                  density=1.5, arrowsize=0.8)

    _draw_bend_walls(ax, d, R_c, L_in, L_out, color='white', lw=2)
    ax.set_xlim(0, L_in + R_c + d / 2 + 1)
    ax.set_ylim(0, d / 2 + R_c + L_out + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'Speed $|\\mathbf{{u}}|$  (t = {t:.1f})')
    plt.colorbar(im, ax=ax, label='$|u|$', shrink=0.7)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ── Pressure field ────────────────────────────────────────────────────────────

def plot_pressure_field(
    h5_path: str,
    d: float = 1.0,
    R_c: float = 3.0,
    L_in: float = 5.0,
    L_out: float = 5.0,
    step: int | None = None,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Pressure contour with wall outlines."""
    import matplotlib.pyplot as plt

    u, v, p, xc, yc, t, st = _load_last_snapshot(h5_path, step)

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = max(float(np.percentile(np.abs(p), 99)), 0.01)
    im = ax.pcolormesh(xc, yc, p.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    _draw_bend_walls(ax, d, R_c, L_in, L_out, color='k', lw=2)
    ax.set_xlim(0, L_in + R_c + d / 2 + 1)
    ax.set_ylim(0, d / 2 + R_c + L_out + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'Pressure  (t = {t:.1f})')
    plt.colorbar(im, ax=ax, label='$p$', shrink=0.7)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ── Radial velocity profiles ─────────────────────────────────────────────────

def plot_radial_profiles(
    h5_path: str,
    d: float = 1.0,
    R_c: float = 3.0,
    step: int | None = None,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Tangential velocity u_θ(r) at 0°, 30°, 60°, 90° through the bend."""
    import matplotlib.pyplot as plt

    u, v, p, xc, yc, t, st = _load_last_snapshot(h5_path, step)
    u_cc = 0.5 * (u[:-1, :] + u[1:, :])
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])

    C = (5.0, d / 2 + R_c)
    R_i = R_c - d / 2
    R_o = R_c + d / 2

    angles_deg = [0, 30, 60, 90]
    angles_rad = [-np.pi / 2 + np.radians(a) for a in angles_deg]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, ax = plt.subplots(figsize=(7, 5))
    for deg, th, clr in zip(angles_deg, angles_rad, colors):
        r_arr, u_tan, _ = _interpolate_radial(
            u_cc, v_cc, xc, yc, C, R_i, R_o, th)
        s = (r_arr - R_i) / d  # normalised distance from inner wall
        ax.plot(u_tan, s, color=clr, lw=2, label=f'θ = {deg}°')

    ax.set_xlabel('$u_{\\theta}$  (tangential velocity)')
    ax.set_ylabel('$(r - R_i) / d$  (inner → outer)')
    ax.set_title(f'Radial velocity profiles  (t = {t:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
