"""
Post-processing utilities for the Rayleigh-Bénard Convection case.

Usage
-----
    from jax_cfd.viz.rb import plot_temperature, plot_velocity, print_summary

    # Print Nusselt number and statistics
    print_summary('output/run.h5')

    # Plot temperature field at last snapshot
    plot_temperature('output/run.h5')

    # Plot velocity magnitude
    plot_velocity('output/run.h5')
"""

from __future__ import annotations

import numpy as np

from jax_cfd.io.reader import load_grid, load_snapshot, list_steps, load_monitor_window


# ── Helper ────────────────────────────────────────────────────────────────────

def _load_snapshot(h5_path: str, step: int | None = None):
    """Return (grid_dict, snap_dict, step) for the requested (or last) step."""
    grid = load_grid(h5_path)
    if step is None:
        step = list_steps(h5_path)[-1]
    snap = load_snapshot(h5_path, step=step)
    return grid, snap, step


# ── Public API ────────────────────────────────────────────────────────────────

def plot_temperature(
    h5_path: str,
    step: int | None = None,
    save: str | None = None,
    show: bool = True,
    cmap: str = 'RdBu_r',
) -> None:
    """
    Colour-map of the temperature field T(x, y) at the given snapshot.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 output file.
    step : int or None
        Snapshot step index.  None → last saved snapshot.
    save : str or None
        If given, save the figure to this path (e.g. 'rb_T.png').
    show : bool
        Call plt.show() if True.
    cmap : str
        Matplotlib colourmap.  'RdBu_r' (hot=red, cold=blue) by default.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.  pip install matplotlib")

    grid, snap, step = _load_snapshot(h5_path, step)
    xc = np.array(grid['xc'])   # (Nx,)
    yc = np.array(grid['yc'])   # (Ny,)

    T = snap.get('T')
    if T is None:
        raise KeyError("No temperature field 'T' found in snapshot — "
                       "was the heat solver used?")
    T = np.array(T)              # (Nx, Ny)

    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(xc, yc, T.T, cmap=cmap, vmin=0.0, vmax=1.0, shading='auto')
    fig.colorbar(pcm, ax=ax, label='T  (non-dim)')
    ax.set_xlabel('x / H')
    ax.set_ylabel('y / H')
    ax.set_title(f'Rayleigh-Bénard  —  temperature  (step {step})')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_velocity(
    h5_path: str,
    step: int | None = None,
    save: str | None = None,
    show: bool = True,
    cmap: str = 'viridis',
) -> None:
    """
    Colour-map of the velocity magnitude |u| at the given snapshot,
    with a quiver overlay (downsampled for readability).

    Parameters
    ----------
    h5_path, step, save, show : see plot_temperature.
    cmap : str
        Colourmap for the magnitude field.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.  pip install matplotlib")

    grid, snap, step = _load_snapshot(h5_path, step)
    xc = np.array(grid['xc'])   # (Nx,)
    yc = np.array(grid['yc'])   # (Ny,)

    u = np.array(snap['u'])     # (Nx+1, Ny)
    v = np.array(snap['v'])     # (Nx, Ny+1)

    u_cc = 0.5 * (u[:-1, :] + u[1:, :])   # (Nx, Ny)
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])   # (Nx, Ny)
    mag  = np.sqrt(u_cc ** 2 + v_cc ** 2)

    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(xc, yc, mag.T, cmap=cmap, shading='auto')
    fig.colorbar(pcm, ax=ax, label='|u|  (non-dim)')

    # Quiver (downsampled)
    skip = max(1, len(xc) // 20)
    Xq, Yq = np.meshgrid(xc[::skip], yc[::skip], indexing='ij')
    ax.quiver(Xq, Yq, u_cc[::skip, ::skip].T, v_cc[::skip, ::skip].T,
              scale=15, width=0.003, color='white', alpha=0.7)

    ax.set_xlabel('x / H')
    ax.set_ylabel('y / H')
    ax.set_title(f'Rayleigh-Bénard  —  velocity magnitude  (step {step})')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(
    h5_path: str,
    t_start: float = 100.0,
    T_hot: float = 1.0,
    T_cold: float = 0.0,
) -> None:
    """
    Print time-averaged Nusselt number and basic statistics.

    Nu is taken from the ``monitor/Nu`` time series (recorded every
    ``diag_every`` steps) and averaged over ``t > t_start`` to exclude
    the initial transient.  A spot-check from the last snapshot wall
    gradient is also shown for comparison.

    Reference values
    ----------------
    Ra=1e6,  Pr=0.71, 2D Γ=2: Nu ≈  8.7  (Grossmann-Lohse 2000)
    Ra=1e9,  Pr=0.71, 2D Γ=2: Nu ≈ 17-25 (Wang et al. 2020)

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 output file.
    t_start : float
        Only average Nu for t > t_start (default 100.0).
    T_hot : float
        Bottom-wall temperature (default 1.0).
    T_cold : float
        Top-wall temperature (default 0.0).
    """
    # ── Time-averaged Nu from monitor ─────────────────────────────────
    try:
        mon = load_monitor_window(h5_path, t_start=t_start)
        nu_ts  = mon.get('Nu')
        t_ts   = mon.get('t', np.array([]))
    except Exception:
        nu_ts = None
        t_ts  = np.array([])

    has_ts = nu_ts is not None and len(nu_ts) > 0

    # ── Spot-check from last snapshot ─────────────────────────────────
    grid, snap, step = _load_snapshot(h5_path)
    yc = np.array(grid['yc'])

    T = snap.get('T')
    if T is None:
        print("No temperature field found — was the heat solver used?")
        return
    T = np.array(T)   # (Nx, Ny)

    dT = T_hot - T_cold
    dy = float(yc[1] - yc[0])
    Ly = float(yc[-1] - yc[0]) + dy    # domain height from cell centers
    Nu_bot = float(np.mean(2.0 * Ly * (T_hot  - T[:, 0])  / (dy * dT)))
    Nu_top = float(np.mean(2.0 * Ly * (T[:, -1] - T_cold) / (dy * dT)))
    Nu_snap = 0.5 * (Nu_bot + Nu_top)

    v_cc    = 0.5 * (np.array(snap['v'])[:, :-1] + np.array(snap['v'])[:, 1:])
    vT_mean = float(np.mean(v_cc * T))

    print()
    print(f"{'─'*58}")
    print(f"  Rayleigh-Bénard Convection")
    print(f"{'─'*58}")
    if has_ts:
        Nu_mean = float(np.mean(nu_ts))
        Nu_std  = float(np.std(nu_ts))
        t_end   = float(t_ts[-1])
        print(f"  Time-averaged Nu  (t ∈ [{t_start:.0f}, {t_end:.0f}], "
              f"N={len(nu_ts)})")
        print(f"    Nu = {Nu_mean:.4f}  ±  {Nu_std:.4f}")
        print()
    else:
        print(f"  [No monitor/Nu data after t={t_start}; using snapshot only]")
        print()

    print(f"  {'Quantity':<34}  {'Value':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Nu (spot-check, bottom wall)':<34}  {Nu_bot:10.4f}")
    print(f"  {'Nu (spot-check, top wall)':<34}  {Nu_top:10.4f}")
    print(f"  {'Nu (spot-check, avg)':<34}  {Nu_snap:10.4f}  (step {step})")
    print(f"  {'<vT> (convective flux)':<34}  {vT_mean:10.6f}")
    print()
    print(f"  Reference (GL 2000 / Wang 2020):")
    print(f"    Ra=1e6, Pr=0.71 → Nu ≈  8.7")
    print(f"    Ra=1e9, Pr=0.71 → Nu ≈ 17-25  (2D, Γ=2)")
    print(f"{'─'*58}")
    print()
