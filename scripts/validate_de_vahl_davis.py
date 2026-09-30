#!/usr/bin/env python3
"""
de Vahl Davis (1983) Benchmark Validation.

Natural convection of air (Pr=0.71) in a square cavity with
differentially heated side walls (left=hot, right=cold, top/bottom=adiabatic).

Reference: de Vahl Davis, G. (1983) IJNMF, Vol.3, pp.249-264.

Benchmark Nu_avg:
    Ra=1e3 → 1.118
    Ra=1e4 → 2.243
    Ra=1e5 → 4.519
    Ra=1e6 → 8.800
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState
from jax_cfd.core.solver import create_step_fn
from jax_cfd.core.boundaries import (
    apply_boundaries, pad_velocity_for_diffusion, pad_scalar_for_diffusion,
)
from jax_cfd.core.advection import advect_momentum_muscl
from jax_cfd.core.schemes import solve_pressure_fft


# ── Benchmark reference ──────────────────────────────────────────────
BENCHMARK = {
    1e3: {'Nu': 1.118},
    1e4: {'Nu': 2.243},
    1e5: {'Nu': 4.519},
    1e6: {'Nu': 8.800},
}

# Grid per Ra (higher Ra needs finer grid)
GRID_SIZE = {
    1e3: 64,
    1e4: 128,
    1e5: 128,
    1e6: 256,
}

OUT_DIR = 'results/de_vahl_davis'


def build_cavity_solver(Ra, Pr=0.71):
    """Build solver for side-heated square cavity."""
    N = GRID_SIZE[Ra]
    Re = float(np.sqrt(Ra / Pr))
    nu = 1.0 / Re

    grid = MACGrid((N, N), (1.0, 1.0))

    bc_cfg = {
        'x_min': {'type': 'wall_no_slip', 'T': 1.0},   # hot wall
        'x_max': {'type': 'wall_no_slip', 'T': 0.0},   # cold wall
        'y_min': {'type': 'wall_no_slip'},               # adiabatic
        'y_max': {'type': 'wall_no_slip'},               # adiabatic
    }

    def apply_bc(vel, _cfg, g, inlet_vel=None):
        return apply_boundaries(vel, bc_cfg, g, inlet_vel=inlet_vel)

    def pad_diff(vel, g, inlet_vel=None):
        return pad_velocity_for_diffusion(vel, bc_cfg, g, inlet_vel=inlet_vel)

    def no_turb(state, grid, cfg):
        return jnp.zeros(grid.cell_shape())

    def pad_scalar(s, name, g, extra=None):
        return pad_scalar_for_diffusion(s, name, bc_cfg, g)

    # Gravity: buoyancy acts upward for hot fluid
    # Free-fall nondim (Re=sqrt(Ra/Pr)): buoyancy coeff = 1.0, NOT Pr
    gravity_nd = (0.0, 1.0)

    step_fn = create_step_fn(
        grid=grid, nu_base=nu,
        apply_bc_fn=apply_bc, pad_diff_fn=pad_diff,
        get_nu_t_fn=no_turb,
        advect_fn=lambda vel, g: advect_momentum_muscl(vel, g),
        solve_p_fn=solve_pressure_fft,
        time_scheme='rk3',
        solve_scalars=True,
        solve_buoyancy=True,
        gravity_vector=gravity_nd,
        beta=1.0,
        T_ref=0.5,
        pad_scalar_fn=pad_scalar,
        scalar_params={'T': {'Pr': Pr}},
    )

    # Initial state: rest + linear T in x (hot left, cold right)
    u = jnp.zeros((N + 1, N))
    v = jnp.zeros((N, N + 1))
    p = jnp.zeros((N, N))
    xc = grid.cell_centers()[0]  # (N, N) or (N,) broadcast
    if xc.ndim == 1:
        T = (1.0 - xc)[:, None] * jnp.ones((1, N))
    else:
        T = 1.0 - xc

    state0 = FluidState(velocity=(u, v), pressure=p,
                        scalars={'T': T}, ibm_forces=None)

    return grid, step_fn, state0, nu


def compute_nu_hot_wall(T, dx):
    """Compute average Nu on hot wall (x=0) from 2nd-order wall gradient.

    Uses 3-point one-sided derivative at the wall (x=0):
      dT/dx|_wall ≈ (-8*T_wall + 9*T[0,:] - T[1,:]) / (3*dx)
      Nu = -dT/dx  (since L=1, ΔT=1)
    """
    T_wall = 1.0
    Nu_local = (8.0 * T_wall - 9.0 * T[0, :] + T[1, :]) / (3.0 * dx)
    return float(jnp.mean(Nu_local))


def run_to_steady(step_fn, state, grid, dt, t_end, diag_every=200):
    """Run simulation until t_end, monitoring Nu convergence."""
    dx = float(grid.spacings[0])
    t = 0.0
    step = 0
    Nu_prev = 0.0

    while t < t_end:
        # Chunk of steps
        def scan_body(carry, _):
            s = carry
            s = step_fn(s, dt, 0.0)
            return s, None

        state, _ = jax.lax.scan(scan_body, state, None, length=diag_every)
        step += diag_every
        t += diag_every * dt

        Nu = compute_nu_hot_wall(state.scalars['T'], dx)
        change = abs(Nu - Nu_prev) / max(abs(Nu), 1e-10)
        Nu_prev = Nu

        # Check convergence (relative change < 1e-5)
        if change < 1e-5 and t > 5.0:
            return state, Nu, t, step, True

    return state, Nu, t, step, False


# ── Visualization ────────────────────────────────────────────────────

def plot_fields(fields_data, out_dir):
    """Plot temperature and velocity fields for all Ra cases (2x4 layout)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col, (Ra, data) in enumerate(sorted(fields_data.items())):
        xc, yc = data['xc'], data['yc']
        T = data['T']
        u_cc = data['u_cc']
        v_cc = data['v_cc']
        Nu = data['Nu']
        Nu_ref = BENCHMARK[Ra]['Nu']
        N = len(xc)

        # Temperature
        ax = axes[0, col]
        pcm = ax.pcolormesh(xc, yc, T.T, cmap='RdBu_r', vmin=0, vmax=1,
                            shading='auto')
        # Isotherms
        X, Y = np.meshgrid(xc, yc, indexing='ij')
        ax.contour(X, Y, T, levels=np.linspace(0, 1, 11),
                   colors='k', linewidths=0.4, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Ra={Ra:.0e}  ({N}×{N})', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Temperature', fontsize=11)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

        # Velocity magnitude + streamlines
        ax = axes[1, col]
        mag = np.sqrt(u_cc**2 + v_cc**2)
        pcm2 = ax.pcolormesh(xc, yc, mag.T, cmap='viridis', shading='auto')
        # Streamlines
        skip = max(1, N // 30)
        ax.streamplot(xc, yc, u_cc.T, v_cc.T, density=1.2,
                      color='white', linewidth=0.5, arrowsize=0.6)
        ax.set_aspect('equal')
        ax.set_title(f'Nu={Nu:.3f} (ref {Nu_ref:.3f}, '
                     f'err {abs(Nu-Nu_ref)/Nu_ref*100:.2f}%)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Velocity + streamlines', fontsize=11)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

    fig.suptitle('de Vahl Davis (1983) Benchmark — Side-heated Square Cavity, Pr=0.71',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'dvd_fields.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_nu_comparison(results, out_dir):
    """Bar chart comparing computed vs reference Nu for each Ra."""
    import matplotlib.pyplot as plt

    Ra_list = [r[0] for r in results]
    Nu_list = [r[2] for r in results]
    Nu_ref_list = [r[3] for r in results]
    err_list = [r[4] for r in results]

    x = np.arange(len(Ra_list))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars1 = ax1.bar(x - width/2, Nu_ref_list, width, label='de Vahl Davis (1983)',
                    color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, Nu_list, width, label='JAX-CFD',
                    color='#FF5722', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Rayleigh number', fontsize=12)
    ax1.set_ylabel('Nusselt number (Nu)', fontsize=12)
    ax1.set_title('Nu Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{Ra:.0e}' for Ra in Ra_list])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, Nu_ref_list):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, Nu_list):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Error plot
    colors = ['#4CAF50' if e < 1.0 else '#FF9800' if e < 5.0 else '#F44336'
              for e in err_list]
    ax2.bar(x, err_list, 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='1% threshold')
    ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax2.set_xlabel('Rayleigh number', fontsize=12)
    ax2.set_ylabel('Relative error (%)', fontsize=12)
    ax2.set_title('Validation Error', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{Ra:.0e}' for Ra in Ra_list])
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(max(err_list) * 1.5, 2.0))
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('de Vahl Davis (1983) Benchmark Validation — JAX-CFD Solver',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'dvd_nu_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_wall_nu_profiles(fields_data, out_dir):
    """Plot local Nu distribution along the hot wall for each Ra."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1976D2', '#388E3C', '#F57C00', '#D32F2F']

    for (Ra, data), color in zip(sorted(fields_data.items()), colors):
        T = data['T']
        yc = data['yc']
        dx = data['dx']
        T_wall = 1.0
        Nu_local = (8.0 * T_wall - 9.0 * T[0, :] + T[1, :]) / (3.0 * dx)
        ax.plot(yc, Nu_local, color=color, linewidth=1.5,
                label=f'Ra={Ra:.0e} (Nu_avg={data["Nu"]:.3f})')

    ax.set_xlabel('y / L', fontsize=12)
    ax.set_ylabel('Local Nu (hot wall)', fontsize=12)
    ax.set_title('Local Nusselt Number Distribution Along Hot Wall',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'dvd_wall_nu.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    Pr = 0.71
    os.makedirs(OUT_DIR, exist_ok=True)

    print()
    print("=" * 66)
    print("  de Vahl Davis (1983) Benchmark Validation")
    print("  Square cavity, Pr=0.71, side-heated")
    print("=" * 66)

    results = []
    fields_data = {}

    for Ra in [1e3, 1e4, 1e5, 1e6]:
        N = GRID_SIZE[Ra]
        Re = float(np.sqrt(Ra / Pr))
        nu = 1.0 / Re
        dx = 1.0 / N

        # Adaptive dt based on CFL + diffusion stability
        kappa = nu / Pr
        diff_max = max(nu, kappa)
        dt = min(0.3 * dx, 0.25 * dx**2 / diff_max)

        # Time to reach steady state (rough estimate)
        t_end = {1e3: 50.0, 1e4: 50.0, 1e5: 100.0, 1e6: 200.0}[Ra]
        diag_every = max(int(1.0 / dt), 100)  # ~1 time unit per check

        print(f"\n  Ra={Ra:.0e}  grid={N}x{N}  Re={Re:.1f}  "
              f"dt={dt:.6f}  diag_every={diag_every}")

        grid, step_fn, state0, _ = build_cavity_solver(Ra, Pr)

        t0 = time.time()
        state, Nu, t_final, steps, converged = run_to_steady(
            step_fn, state0, grid, dt, t_end, diag_every)
        elapsed = time.time() - t0

        Nu_ref = BENCHMARK[Ra]['Nu']
        err = abs(Nu - Nu_ref) / Nu_ref * 100

        status = "CONVERGED" if converged else "t_end reached"
        results.append((Ra, N, Nu, Nu_ref, err, elapsed, status))

        # Save field data for visualization
        xc = np.array(grid.cell_centers()[0])
        if xc.ndim > 1:
            xc = xc[:, 0]
        yc = np.array(grid.cell_centers()[1])
        if yc.ndim > 1:
            yc = yc[0, :]
        u_np = np.array(state.velocity[0])
        v_np = np.array(state.velocity[1])
        u_cc = 0.5 * (u_np[:-1, :] + u_np[1:, :])
        v_cc = 0.5 * (v_np[:, :-1] + v_np[:, 1:])
        fields_data[Ra] = {
            'xc': xc, 'yc': yc,
            'T': np.array(state.scalars['T']),
            'u_cc': u_cc, 'v_cc': v_cc,
            'Nu': Nu, 'dx': dx,
        }

        print(f"    Nu={Nu:.4f}  ref={Nu_ref:.3f}  err={err:.2f}%  "
              f"t={t_final:.1f}  steps={steps}  {elapsed:.1f}s  [{status}]")

    # Summary table
    print(f"\n{'=' * 66}")
    print(f"  {'Ra':>8s}  {'Grid':>6s}  {'Nu':>8s}  {'Ref':>8s}  "
          f"{'Error':>7s}  {'Time':>7s}  {'Status'}")
    print(f"  {'-' * 60}")
    for Ra, N, Nu, ref, err, elapsed, status in results:
        print(f"  {Ra:8.0e}  {N:3d}x{N:<3d}  {Nu:8.4f}  {ref:8.3f}  "
              f"{err:6.2f}%  {elapsed:6.1f}s  {status}")
    print(f"{'=' * 66}\n")

    # Pass/fail
    all_pass = all(err < 5.0 for _, _, _, _, err, _, _ in results)
    if all_pass:
        print("  ALL PASSED (< 5% error)")
    else:
        failed = [f"Ra={Ra:.0e}({err:.1f}%)"
                  for Ra, _, _, _, err, _, _ in results if err >= 5.0]
        print(f"  FAILED: {', '.join(failed)}")

    # Generate visualizations
    print(f"\n  Generating visualizations → {OUT_DIR}/")
    plot_fields(fields_data, OUT_DIR)
    plot_nu_comparison(results, OUT_DIR)
    plot_wall_nu_profiles(fields_data, OUT_DIR)
    print("  Done.\n")


if __name__ == '__main__':
    main()
