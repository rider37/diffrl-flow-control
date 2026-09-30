"""
Post-processing for the backward-facing step.

Reference
---------
Erturk (2008) "Numerical Solutions of 2-D Steady Incompressible Flow over a
Backward-Facing Step", Part I: High Reynolds Number Solutions.
International Journal for Numerical Methods in Fluids.

Re definition: Re = U_max · h / ν  (U_max = peak inlet velocity, h = step height)
Expansion ratio ER = 2  (inlet channel height h, downstream channel H = 2h)

Tabulated primary reattachment length x_r1 / h  (Erturk 2008, Table 1)
-----------------------------------------------------------------------
Re=100:  3.09,  Re=200: 4.68,  Re=300: 5.91,  Re=400: 6.84
Re=500:  7.65,  Re=600: 8.35,  Re=700: 8.87,  Re=800: 9.32, Re=1000: 10.05

Usage
-----
    from jax_cfd.viz.step import print_summary, plot_reattachment, plot_field_step

    print_summary('output/run.h5', Re=200)
    plot_reattachment('output/run.h5', Re=200)
    plot_field_step('output/run.h5', field='vorticity')
"""

from __future__ import annotations

import numpy as np

# ── Erturk (2008) reference data ─────────────────────────────────────────
_ERTURK_XR = {
    100:  3.09,
    200:  4.68,
    300:  5.91,
    400:  6.84,
    500:  7.65,
    600:  8.35,
    700:  8.87,
    800:  9.32,
    1000: 10.05,
}


def _load_snapshot(h5_path: str, step: int | None = None):
    """Return (grid, snap, Nx, Ny, dx, dy, xu)."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        xc = f['grid']['xc'][:]
        yc = f['grid']['yc'][:]
        xu = f['grid']['xu'][:]
        yv = f['grid']['yv'][:]

        keys = sorted(f['fields'].keys())
        key  = keys[-1] if step is None else f'step_{step:07d}'
        grp  = f['fields'][key]
        u    = grp['u'][:]   # (Nx+1, Ny)
        v    = grp['v'][:]   # (Nx, Ny+1)
        p    = grp['p'][:]   # (Nx, Ny)
        t    = float(grp.attrs['t'])
        step = int(grp.attrs['step'])

    Nx, Ny = len(xc), len(yc)
    dx = float(xc[1] - xc[0])
    dy = float(yc[1] - yc[0])
    grid = dict(xc=xc, yc=yc, xu=xu, yv=yv)
    snap = dict(u=u, v=v, p=p, t=t, step=step)
    return grid, snap, Nx, Ny, dx, dy, xu


def reattachment_length(
    h5_path: str,
    h: float = 1.0,
    step: int | None = None,
) -> dict:
    """
    Compute the primary (and secondary if present) reattachment length from
    the sign change of the bottom-wall shear stress τ_w ∝ u[i, 0].

    Returns
    -------
    dict with keys:
        'x_r1'   : primary reattachment length / h  (behind step at y=0)
        'x_r2'   : secondary reattachment length / h  (top wall, or NaN)
        't'      : simulation time of snapshot
    """
    grid, snap, Nx, Ny, dx, dy, xu = _load_snapshot(h5_path, step)
    u = snap['u']   # (Nx+1, Ny)
    t = snap['t']

    # ── Bottom-wall shear (y = 0) ──────────────────────────────────────
    # u[i, 0] is proportional to τ_w at the bottom wall via ghost-cell:
    #   τ_w ≈ ν · 2·u[i,0] / dy
    # Negative → recirculation (flow going left), positive → attached.
    u_bot = u[1:, 0]                       # skip inlet face; shape (Nx,)
    x_bot = xu[1:]                          # matching x positions

    x_r1 = _find_reattachment(u_bot, x_bot)

    # ── Top-wall shear (y = H) — secondary recirculation for Re > ~400 ──
    # Top wall: u[i, -1] is the first u-cell below the top wall.
    # Secondary recirculation has u[i, -1] > 0 (rightward near top wall,
    # which is the reversed flow in the secondary bubble).
    # Sign convention flipped: look for positive → negative transition.
    u_top = u[1:, -1]
    x_r2 = _find_reattachment(-u_top, x_bot)   # flipped sign for top wall

    return {'x_r1': x_r1 / h, 'x_r2': x_r2 / h if not np.isnan(x_r2) else np.nan, 't': t}


def _find_reattachment(
    u_wall: np.ndarray,
    x: np.ndarray,
    x_skip_min: float = 0.5,
) -> float:
    """
    Find where u_wall changes from negative to positive (reattachment).

    Skips cells within ``x_skip_min`` of the inlet to avoid the
    corner-singularity spurious sign flip (u oscillates ±1e-5 right
    behind the step corner before the real recirculation develops).

    Returns the linearly interpolated x position, or NaN if not found.
    """
    for i in range(len(u_wall) - 1):
        if float(x[i]) < x_skip_min:
            continue
        if u_wall[i] < 0 and u_wall[i + 1] >= 0:
            u1, u2 = float(u_wall[i]), float(u_wall[i + 1])
            x1, x2 = float(x[i]),     float(x[i + 1])
            return x1 - u1 * (x2 - x1) / (u2 - u1)
    return float('nan')


def print_summary(h5_path: str, Re: float = 200.0, h: float = 1.0) -> None:
    """
    Print reattachment lengths and compare with Erturk (2008).

    Parameters
    ----------
    h5_path : path to HDF5 output file
    Re      : Reynolds number used in simulation (U_max · h / ν)
    h       : step height (default 1.0)
    """
    res = reattachment_length(h5_path, h=h)

    # Nearest tabulated Erturk reference
    re_tab = min(_ERTURK_XR.keys(), key=lambda r: abs(r - Re))
    xr_ref = _ERTURK_XR.get(re_tab, float('nan'))

    print()
    print('── Backward-Facing Step Summary ──────────────────────────')
    print(f'  Snapshot  : t = {res["t"]:.3f}')
    print(f'  Re        : {Re:.0f}')
    print()
    print(f'  x_r1 / h  (sim)  : {res["x_r1"]:.3f}')
    if not np.isnan(xr_ref):
        err = abs(res['x_r1'] - xr_ref)
        print(f'  x_r1 / h  (ref)  : {xr_ref:.2f}'
              f'   [Erturk 2008, Re={re_tab}]')
        print(f'  |error|          : {err:.3f} h')
    if not np.isnan(res['x_r2']):
        print(f'  x_r2 / h  (sim)  : {res["x_r2"]:.3f}   (secondary / top-wall)')
    print('──────────────────────────────────────────────────────────')
    print()


def plot_reattachment(
    h5_path: str,
    Re: float = 200.0,
    h: float = 1.0,
    save: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot the bottom-wall shear stress profile and mark the reattachment point.
    Overlays the Erturk (2008) reference x_r1 / h for the given Re.
    """
    import matplotlib.pyplot as plt

    grid, snap, Nx, Ny, dx, dy, xu = _load_snapshot(h5_path)
    u = snap['u']

    u_bot = u[1:, 0]
    x_bot = xu[1:]

    res = reattachment_length(h5_path, h=h)
    re_tab = min(_ERTURK_XR.keys(), key=lambda r: abs(r - Re))
    xr_ref = _ERTURK_XR.get(re_tab, float('nan'))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x_bot / h, u_bot, 'b-', lw=1.5, label=r'$u(x,\,y{=}0)$  (∝ τ_w)')
    ax.axhline(0, color='k', lw=0.8, ls=':')

    if not np.isnan(res['x_r1']):
        ax.axvline(res['x_r1'], color='tab:blue', ls='--', lw=1.5,
                   label=f"$x_{{r1}}/h$ = {res['x_r1']:.2f}  (sim)")
    if not np.isnan(xr_ref):
        ax.axvline(xr_ref, color='tab:red', ls='--', lw=1.5,
                   label=f"$x_{{r1}}/h$ = {xr_ref:.2f}  [Erturk Re={re_tab}]")

    ax.set_xlabel('x / h')
    ax.set_ylabel(r'$u(x,\,y_1)$   [∝  τ$_w$]')
    ax.set_title(f'Backward-Facing Step — wall shear   Re={Re:.0f},  t={snap["t"]:.2f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_field_step(
    h5_path: str,
    field: str = 'vorticity',
    step: int | None = None,
    clim: tuple | None = None,
    save: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot a 2-D field (vorticity, u, v, pressure, speed) for the step geometry.

    The step face region (x = 0, y < h) is shaded grey to indicate the wall.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    grid, snap, Nx, Ny, dx, dy, xu = _load_snapshot(h5_path, step)
    xc, yc = grid['xc'], grid['yc']
    u_mac = snap['u']
    v_mac = snap['v']
    H = float(yc[-1]) + dy / 2
    h = H / 2.0

    # Cell-centred velocities
    u_cc = 0.5 * (u_mac[:-1, :] + u_mac[1:, :])
    v_cc = 0.5 * (v_mac[:, :-1] + v_mac[:, 1:])

    if field == 'vorticity':
        data = np.gradient(v_cc, dx, axis=0) - np.gradient(u_cc, dy, axis=1)
        cmap, label = 'RdBu_r', 'ω'
    elif field == 'speed':
        data = np.sqrt(u_cc**2 + v_cc**2)
        cmap, label = 'viridis', '|u|'
    elif field == 'u':
        data = u_cc
        cmap, label = 'RdBu_r', 'u'
    elif field == 'v':
        data = v_cc
        cmap, label = 'RdBu_r', 'v'
    elif field in ('p', 'pressure'):
        data = snap['p']
        cmap, label = 'viridis', 'p'
    else:
        raise ValueError(f"Unknown field '{field}'")

    fig, ax = plt.subplots(figsize=(14, 3.5))

    vmin, vmax = clim if clim else (None, None)
    if clim is None and cmap == 'RdBu_r':
        amax = float(np.percentile(np.abs(data), 99))
        vmin, vmax = -amax, amax

    im = ax.pcolormesh(xc, yc, data.T, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, label=label, shrink=0.9)

    # Shade the step-face region (step body approximation)
    step_patch = patches.Rectangle(
        (-dx, 0), dx, h,
        linewidth=0, facecolor='#888888', zorder=5
    )
    ax.add_patch(step_patch)

    ax.set_xlim(xc[0], xc[-1])
    ax.set_ylim(0, H)
    ax.set_xlabel('x / h')
    ax.set_ylabel('y / h')
    ax.set_title(f'{field}   (t = {snap["t"]:.2f},  step {snap["step"]})')
    ax.set_aspect('equal')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
