"""de Vahl Davis (1983) differentially heated cavity benchmark (Section 2.3).

Natural convection of air (Pr=0.71) in a square cavity with
differentially heated side walls (left=hot, right=cold, top/bottom=adiabatic).

Reference: de Vahl Davis, G. (1983) IJNMF, Vol.3, pp.249-264.

Benchmark Nu_avg:
    Ra=1e3 → 1.118
    Ra=1e4 → 2.243
    Ra=1e5 → 4.519
    Ra=1e6 → 8.800

Writes results/checks/de_vahl_davis.json (hot-wall Nu, benchmark and relative error per Ra).
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

import json
import time
import numpy as np
import jax
import jax.numpy as jnp

from diffrl_flow_control.solver.grid import MACGrid
from diffrl_flow_control.solver.state import FluidState
from diffrl_flow_control.solver.step import create_step_fn
from diffrl_flow_control.solver.boundaries import (
    apply_boundaries, pad_velocity_for_diffusion, pad_scalar_for_diffusion,
)
from diffrl_flow_control.solver.advection import advect_momentum_muscl
from diffrl_flow_control.solver.schemes import solve_pressure_fft


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

OUT_DIR = 'results/checks'


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

    def apply_bc(vel, _cfg, g):
        return apply_boundaries(vel, bc_cfg, g)

    def pad_diff(vel, g):
        return pad_velocity_for_diffusion(vel, bc_cfg, g)

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
        results.append(dict(Ra=Ra, grid=N, Nu=Nu, Nu_ref=Nu_ref, rel_error_pct=err,
                            t_final=t_final, steps=steps, converged=converged))

        print(f"    Nu={Nu:.4f}  ref={Nu_ref:.3f}  err={err:.2f}%  "
              f"t={t_final:.1f}  steps={steps}  {elapsed:.1f}s  "
              f"[{'CONVERGED' if converged else 't_end reached'}]")

    path = os.path.join(OUT_DIR, 'de_vahl_davis.json')
    with open(path, 'w') as f:
        json.dump(dict(Pr=Pr, results=results,
                       max_rel_error_pct=max(r['rel_error_pct'] for r in results)), f, indent=1)
    print(f"\n  max error {max(r['rel_error_pct'] for r in results):.2f}%  -> {path}")


if __name__ == '__main__':
    main()
