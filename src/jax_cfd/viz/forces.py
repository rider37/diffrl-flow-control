"""
Force / aerodynamic coefficient visualization.

Usage examples
--------------
from jax_cfd.viz.forces import plot_cd_cl, strouhal_number

# Time series of Cd and Cl (averaging after t=100):
plot_cd_cl('output/run.h5', t_start=100.0)

# Strouhal number from Cl spectrum:
St = strouhal_number('output/run.h5', t_start=100.0, U_ref=1.0, D=1.0)
print(f'St = {St:.4f}')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from jax_cfd.io.reader import load_diagnostics_window, load_diagnostics

# Probe ring distances (in units of min_dx) for pressure extrapolation.
# Two rings outside the IBM smearing zone (~2 dx) are sampled and
# linearly extrapolated back to the body surface.
_PROBE_RING_OFFSETS = (1.5, 2.5)


def plot_cd_cl(
    h5_path: str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 5),
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Cd and Cl time series, with mean values annotated.

    Parameters
    ----------
    t_start : float
        Start of the averaging / display window (skips initial transient).
    t_end : float, optional
        End of display window. Defaults to end of simulation.
    """
    full = load_diagnostics(h5_path)
    win  = load_diagnostics_window(h5_path, t_start, t_end)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Cd ────────────────────────────────────────────────────────────────
    ax1.plot(full['t'], full['Cd'], lw=0.8, color='tab:blue', alpha=0.5)
    ax1.plot(win['t'],  win['Cd'],  lw=1.2, color='tab:blue')
    if len(win['Cd']) > 0:
        cd_mean = float(np.mean(win['Cd']))
        ax1.axhline(cd_mean, color='tab:blue', ls='--', lw=1.2,
                    label=f'$\\bar{{C}}_d$ = {cd_mean:.3f}')
    ax1.set_ylabel('$C_d$')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Cl ────────────────────────────────────────────────────────────────
    ax2.plot(full['t'], full['Cl'], lw=0.8, color='tab:red', alpha=0.5)
    ax2.plot(win['t'],  win['Cl'],  lw=1.2, color='tab:red')
    if len(win['Cl']) > 0:
        cl_rms = float(np.sqrt(np.mean(win['Cl'] ** 2)))
        ax2.axhline(0, color='k', ls=':', lw=0.8)
        ax2.set_ylabel('$C_l$')
        ax2.set_title(f"$C_{{l,\\mathrm{{rms}}}}$ = {cl_rms:.3f}", fontsize=9,
                      loc='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('t')

    if t_start > 0:
        for ax in (ax1, ax2):
            ax.axvline(t_start, color='k', ls=':', lw=1.0, alpha=0.5,
                       label='stat window')

    fig.suptitle('Aerodynamic Force Coefficients', y=1.01)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    elif show:
        plt.show()
    return fig


def strouhal_number(
    h5_path: str,
    t_start: float = 0.0,
    U_ref: float = 1.0,
    D: float = 1.0,
    plot: bool = False,
    save: Optional[str] = None,
) -> float:
    """
    Estimate the Strouhal number from the dominant frequency in the Cl spectrum.

    St = f_dominant * D / U_ref

    Parameters
    ----------
    t_start : float
        Ignore data before this time (transient).
    U_ref : float
        Reference velocity (used for non-dimensionalisation).
    D : float
        Reference length (cylinder diameter).
    plot : bool
        If True, plot the power spectrum of Cl.

    Returns
    -------
    St : float
        Strouhal number.
    """
    win = load_diagnostics_window(h5_path, t_start)
    t  = win['t']
    Cl = win['Cl']

    if len(t) < 8:
        raise ValueError("Not enough diagnostic data after t_start to estimate Strouhal number.")

    # Uniform time step assumed (interpolate if needed)
    dt_diag = float(np.mean(np.diff(t)))
    freqs   = np.fft.rfftfreq(len(Cl), d=dt_diag)
    power   = np.abs(np.fft.rfft(Cl - Cl.mean())) ** 2

    # Dominant frequency (exclude DC)
    idx_peak = int(np.argmax(power[1:])) + 1
    f_peak   = float(freqs[idx_peak])
    St       = f_peak * D / U_ref

    if plot or save:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(freqs[1:] * D / U_ref, power[1:], lw=1.0, color='tab:blue')
        ax.axvline(St, color='tab:red', ls='--', lw=1.5,
                   label=f'St = {St:.4f}')
        ax.set_xlabel('St = f D / U')
        ax.set_ylabel('Power')
        ax.set_title('Cl Power Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    return St


def compute_surface_coefficients(
    h5_path: str,
    t_start: float = 0.0,
    U_ref: float = 1.0,
    D: float = 1.0,
    cx: float = 4.0,
    cy: float = 10.0,
    R: float = 0.5,
) -> dict:
    """
    Time-averaged pressure-based drag and lift coefficients.

    Cd_p = -∮ p nx ds / q_ref,  Cl_p = -∮ p ny ds / q_ref
    where q_ref = 0.5 ρ U² D  (ρ=1).

    Pressure is sampled at two probe rings outside the IBM smearing
    zone (R + 1.5 dx and R + 2.5 dx) and linearly extrapolated back
    to the body surface (r = R).  The arc-length element ds uses the
    true body radius R.

    Parameters
    ----------
    cx, cy : float
        Cylinder centre (physical coordinates).
    R : float
        Cylinder radius.

    Returns
    -------
    dict with keys: 'Cd_mean', 'Cd_std', 'Cl_rms', 'Cl_mean'
    """
    import h5py
    from jax_cfd.io.reader import load_grid

    grid  = load_grid(h5_path)
    xc    = np.array(grid['xc'])
    yc    = np.array(grid['yc'])
    q_ref = 0.5 * U_ref ** 2 * D

    min_dx = min(float(xc[1] - xc[0]), float(yc[1] - yc[0]))
    n_pts  = int(2.0 * np.pi * R / min_dx)
    theta  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    nx_hat = np.cos(theta)
    ny_hat = np.sin(theta)
    ds = 2.0 * np.pi * R / n_pts   # body-surface arc length

    # Two probe rings outside the IBM smearing zone (~2 dx wide)
    r1 = R + _PROBE_RING_OFFSETS[0] * min_dx
    r2 = R + _PROBE_RING_OFFSETS[1] * min_dx
    xs1 = cx + r1 * nx_hat;  ys1 = cy + r1 * ny_hat
    xs2 = cx + r2 * nx_hat;  ys2 = cy + r2 * ny_hat
    # Linear extrapolation weights:  p_R = w1 * p(r1) + w2 * p(r2)
    w1 = (R - r2) / (r1 - r2)   # = 2.5
    w2 = (R - r1) / (r2 - r1)   # = -1.5

    def _interp(x_arr, y_arr, p_field):
        out = np.empty(len(x_arr))
        for k in range(len(x_arr)):
            xi, yi = x_arr[k], y_arr[k]
            ix = int(np.clip(np.searchsorted(xc, xi) - 1, 0, len(xc) - 2))
            iy = int(np.clip(np.searchsorted(yc, yi) - 1, 0, len(yc) - 2))
            tx = (xi - xc[ix]) / (xc[ix + 1] - xc[ix])
            ty = (yi - yc[iy]) / (yc[iy + 1] - yc[iy])
            out[k] = ((1 - tx) * (1 - ty) * p_field[ix,   iy  ] +
                       tx      * (1 - ty) * p_field[ix+1,  iy  ] +
                      (1 - tx) *       ty * p_field[ix,    iy+1] +
                       tx      *       ty * p_field[ix+1,  iy+1])
        return out

    with h5py.File(h5_path, 'r') as f:
        mon_step = f['monitor/step'][:]
        mon_t    = f['monitor/t'][:]

    cd_list, cl_list = [], []
    with h5py.File(h5_path, 'r') as f:
        for key in sorted(f['fields'].keys()):
            step_num = int(key.split('_')[1])
            idx = int(np.argmin(np.abs(mon_step - step_num)))
            if float(mon_t[idx]) < t_start:
                continue
            p = np.array(f[f'fields/{key}/p'])

            p_surf = w1 * _interp(xs1, ys1, p) + w2 * _interp(xs2, ys2, p)

            cd_list.append(-np.sum(p_surf * nx_hat) * ds / q_ref)
            cl_list.append(-np.sum(p_surf * ny_hat) * ds / q_ref)

    cd = np.array(cd_list)
    cl = np.array(cl_list)
    return {
        'Cd_mean': float(cd.mean()),
        'Cd_std' : float(cd.std()),
        'Cl_mean': float(cl.mean()),
        'Cl_rms' : float(np.sqrt(np.mean(cl ** 2))),
    }


def print_summary(
    h5_path: str,
    t_start: float = 0.0,
    U_ref: float = 1.0,
    D: float = 1.0,
    cx: float = 4.0,
    cy: float = 10.0,
    R: float = 0.5,
) -> None:
    """
    Print a concise validation table: pressure Cd, rms Cl, St.

    Notes
    -----
    Cd is computed from pressure integration on the cylinder surface
    (control-surface approach), which is accurate for IBM simulations.
    The IBM Lagrangian-force Cd stored in diagnostics/Cd is *not* used
    here because it includes a virtual-fluid inertia term that is
    O(1/dt) and does not represent aerodynamic drag.

    Reference values (Norberg 1994 / Ong & Wallace 1996, Re=3900):
        Cd ≈ 0.96,  St ≈ 0.21
    """
    win = load_diagnostics_window(h5_path, t_start)
    t = win['t']

    if len(t) == 0:
        print("No diagnostic data available after t_start.")
        return

    St   = strouhal_number(h5_path, t_start, U_ref, D)
    coef = compute_surface_coefficients(h5_path, t_start, U_ref, D, cx, cy, R)

    print()
    print("─" * 60)
    print("  Cylinder  —  pressure-based coefficients")
    print("─" * 60)
    print(f"  Statistics window       : t ∈ [{t[0]:.1f}, {t[-1]:.1f}]")
    print(f"  Cd  (pressure, mean±std): {coef['Cd_mean']:.4f} ± {coef['Cd_std']:.4f}"
          "   [ref ≈ 0.96]")
    print(f"  Cl  (pressure, rms)     : {coef['Cl_rms']:.4f}")
    print(f"  Strouhal number         : {St:.4f}"
          "             [ref ≈ 0.21]")
    print("─" * 60)
    print()
