"""Compressible Euler/Navier-Stokes solver.

Density-based explicit solver for compressible flow on a collocated
(cell-centred) structured grid.  Uses HLLC approximate Riemann solver
for inviscid fluxes and central differences for viscous terms.

Conservative variables:
    Q = (ρ, ρu, ρv, [ρw,] E)

Equation of state (ideal gas):
    p = (γ - 1)(E - 0.5 ρ |u|²)
    c = √(γ p / ρ)

Time integration: TVD-RK3 (Shu-Osher), same as incompressible solver.

Usage (via runner CONFIG):
    'solver': 'compressible_euler',     # or 'compressible_ns'
    'gamma': 1.4,
    'Mach': 2.0,                        # freestream Mach number
"""
from __future__ import annotations

from typing import Tuple
import jax
import jax.numpy as jnp
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState


# ── Equation of State ────────────────────────────────────────

def pressure_from_conservative(rho, rhou, E, gamma):
    """p = (γ-1)(E - 0.5 ρ|u|²)."""
    ke = 0.5 * sum(m ** 2 for m in rhou) / jnp.maximum(rho, 1e-14)
    return (gamma - 1.0) * (E - ke)


def sound_speed(rho, p, gamma):
    """c = √(γ p / ρ)."""
    return jnp.sqrt(gamma * jnp.maximum(p, 1e-14) / jnp.maximum(rho, 1e-14))


# ── MUSCL reconstruction ─────────────────────────────────────

def _minmod(a, b):
    """Minmod limiter."""
    return jnp.where(a * b > 0,
                     jnp.where(jnp.abs(a) < jnp.abs(b), a, b),
                     0.0)


def _reconstruct_muscl_1d(q, axis):
    """MUSCL reconstruction with minmod limiter along one axis.

    Returns (q_L, q_R) at i+1/2 faces.
    q_L: value approaching from left (cell i side)
    q_R: value approaching from right (cell i+1 side)
    """
    ndim = q.ndim
    pad_w = [(0, 0)] * ndim
    pad_w[axis] = (2, 2)
    q_pad = jnp.pad(q, pad_w, mode='edge')

    def sl(start, end=None):
        s = [slice(None)] * ndim
        s[axis] = slice(start, end)
        return tuple(s)

    q_mm = q_pad[sl(0, -3)]  # i-1
    q_m = q_pad[sl(1, -2)]   # i
    q_p = q_pad[sl(2, -1)]   # i+1
    q_pp = q_pad[sl(3, None)]  # i+2

    # Slopes
    dL = q_m - q_mm
    dR = q_p - q_m
    dRR = q_pp - q_p

    slope_L = _minmod(dL, dR)
    slope_R = _minmod(dR, dRR)

    q_L = q_m + 0.5 * slope_L
    q_R = q_p - 0.5 * slope_R

    return q_L, q_R


# ── HLLC Riemann Solver ──────────────────────────────────────

def _hllc_flux(rho_L, rho_R, u_L, u_R, v_L, v_R, p_L, p_R,
               E_L, E_R, gamma, w_L=None, w_R=None):
    """HLLC approximate Riemann solver for 2D/3D.

    Normal direction is the first velocity component (u).
    Tangential components (v, optionally w) are passively advected.

    Returns conservative flux tuple: (F_rho, F_rhou, F_rhov, F_E[, F_rhow])
    """
    # Sound speeds
    c_L = sound_speed(rho_L, p_L, gamma)
    c_R = sound_speed(rho_R, p_R, gamma)

    # Wave speed estimates (Davis)
    S_L = jnp.minimum(u_L - c_L, u_R - c_R)
    S_R = jnp.maximum(u_L + c_L, u_R + c_R)

    # Contact wave speed
    num = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    den = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_star = num / jnp.where(jnp.abs(den) > 1e-14, den, 1e-14)

    # Left flux
    F_rho_L = rho_L * u_L
    F_rhou_L = rho_L * u_L ** 2 + p_L
    F_rhov_L = rho_L * u_L * v_L
    F_E_L = u_L * (E_L + p_L)

    # Right flux
    F_rho_R = rho_R * u_R
    F_rhou_R = rho_R * u_R ** 2 + p_R
    F_rhov_R = rho_R * u_R * v_R
    F_E_R = u_R * (E_R + p_R)

    # Star states
    def _star_state(rho, u, v, p, E, S, S_s):
        coeff = rho * (S - u) / jnp.where(jnp.abs(S - S_s) > 1e-14, S - S_s, 1e-14)
        rho_s = coeff
        rhou_s = coeff * S_s
        rhov_s = coeff * v
        E_s = coeff * (E / jnp.maximum(rho, 1e-14) +
                       (S_s - u) * (S_s + p / (rho * jnp.where(jnp.abs(S - u) > 1e-14, S - u, 1e-14))))
        return rho_s, rhou_s, rhov_s, E_s

    rho_sL, rhou_sL, rhov_sL, E_sL = _star_state(rho_L, u_L, v_L, p_L, E_L, S_L, S_star)
    rho_sR, rhou_sR, rhov_sR, E_sR = _star_state(rho_R, u_R, v_R, p_R, E_R, S_R, S_star)

    # HLLC flux selection
    def _hllc_select(F_L, F_R, Q_L, Q_R, Q_sL, Q_sR):
        F_sL = F_L + S_L * (Q_sL - Q_L)
        F_sR = F_R + S_R * (Q_sR - Q_R)
        return jnp.where(S_L >= 0, F_L,
               jnp.where(S_star >= 0, F_sL,
               jnp.where(S_R >= 0, F_sR, F_R)))

    F_rho = _hllc_select(F_rho_L, F_rho_R, rho_L, rho_R, rho_sL, rho_sR)
    F_rhou = _hllc_select(F_rhou_L, F_rhou_R, rho_L * u_L, rho_R * u_R, rhou_sL, rhou_sR)
    F_rhov = _hllc_select(F_rhov_L, F_rhov_R, rho_L * v_L, rho_R * v_R, rhov_sL, rhov_sR)
    F_E = _hllc_select(F_E_L, F_E_R, E_L, E_R, E_sL, E_sR)

    result = (F_rho, F_rhou, F_rhov, F_E)

    # 3D: add w-momentum flux
    if w_L is not None:
        F_rhow_L = rho_L * u_L * w_L
        F_rhow_R = rho_R * u_R * w_R
        coeff_L = rho_L * (S_L - u_L) / jnp.where(jnp.abs(S_L - S_star) > 1e-14, S_L - S_star, 1e-14)
        coeff_R = rho_R * (S_R - u_R) / jnp.where(jnp.abs(S_R - S_star) > 1e-14, S_R - S_star, 1e-14)
        rhow_sL = coeff_L * w_L
        rhow_sR = coeff_R * w_R
        F_rhow = _hllc_select(F_rhow_L, F_rhow_R, rho_L * w_L, rho_R * w_R, rhow_sL, rhow_sR)
        result = result + (F_rhow,)

    return result


# ── Flux Divergence ───────────────────────────────────────────

def _flux_divergence_1d(flux, axis, dx):
    """(F_{i+1/2} - F_{i-1/2}) / dx for interior cells."""
    ndim = flux.ndim
    sp = [slice(None)] * ndim
    sm = [slice(None)] * ndim
    sp[axis] = slice(1, None)
    sm[axis] = slice(None, -1)
    return (flux[tuple(sp)] - flux[tuple(sm)]) / dx


# ── Viscous Fluxes ────────────────────────────────────────────

def _viscous_rhs(rho, velocity, E, grid, mu):
    """Compute viscous RHS: ∇·τ for momentum and ∇·(τ·u + κ∇T) for energy.

    Simplified: constant viscosity μ, Pr=0.72.
    """
    ndim = grid.ndim
    Pr = 0.72

    # Velocity Laplacian (cell-centred)
    visc_forces = []
    for i in range(ndim):
        lap_ui = jnp.zeros_like(velocity[i])
        padded = jnp.pad(velocity[i], [(1, 1)] * ndim, mode='edge')
        for ax in range(ndim):
            dx = grid.spacings[ax]
            s = [slice(1, -1)] * ndim
            sp = list(s); sp[ax] = slice(2, None)
            sm = list(s); sm[ax] = slice(None, -2)
            lap_ui = lap_ui + (padded[tuple(sp)] - 2.0 * padded[tuple(s)] + padded[tuple(sm)]) / dx ** 2
        visc_forces.append(mu * lap_ui)

    # Energy: viscous dissipation + heat conduction (κ∇²T)
    ke = 0.5 * sum(u ** 2 for u in velocity)
    p = (1.4 - 1.0) * (E - rho * ke)  # assuming gamma=1.4
    T = p / jnp.maximum(rho, 1e-14)   # ideal gas: p = ρ T (non-dim)
    kappa = mu / Pr
    lap_T = jnp.zeros_like(T)
    padded_T = jnp.pad(T, [(1, 1)] * ndim, mode='edge')
    for ax in range(ndim):
        dx = grid.spacings[ax]
        s = [slice(1, -1)] * ndim
        sp = list(s); sp[ax] = slice(2, None)
        sm = list(s); sm[ax] = slice(None, -2)
        lap_T = lap_T + (padded_T[tuple(sp)] - 2.0 * padded_T[tuple(s)] + padded_T[tuple(sm)]) / dx ** 2
    visc_E = sum(velocity[i] * visc_forces[i] for i in range(ndim)) + kappa * lap_T

    return visc_forces, visc_E


# ── Main Solver ───────────────────────────────────────────────

def _compressible_rhs(state, grid, gamma, mu=0.0):
    """Compute dQ/dt for compressible Euler/NS.

    state.velocity: cell-centred (u, v[, w])
    state.scalars['rho']: density
    state.scalars['E']: total energy
    state.pressure: from EOS (informational)
    """
    ndim = grid.ndim
    rho = state.scalars['rho']
    E = state.scalars['E']
    velocity = state.velocity

    # Momentum
    rho_u = tuple(rho * velocity[i] for i in range(ndim))

    # Pressure
    p = pressure_from_conservative(rho, rho_u, E, gamma)
    p = jnp.maximum(p, 1e-14)

    # Accumulate flux divergence along each axis
    drho_dt = jnp.zeros_like(rho)
    drhou_dt = [jnp.zeros_like(rho) for _ in range(ndim)]
    dE_dt = jnp.zeros_like(rho)

    for axis in range(ndim):
        dx = grid.spacings[axis]

        # MUSCL reconstruction of primitive variables
        rho_L, rho_R = _reconstruct_muscl_1d(rho, axis)
        p_L, p_R = _reconstruct_muscl_1d(p, axis)
        E_L, E_R = _reconstruct_muscl_1d(E, axis)

        # Reconstruct velocity components
        vel_L = [None] * ndim
        vel_R = [None] * ndim
        for i in range(ndim):
            vel_L[i], vel_R[i] = _reconstruct_muscl_1d(velocity[i], axis)

        # Rotate so normal velocity is first
        u_L, u_R = vel_L[axis], vel_R[axis]
        # Tangential: pick first non-axis component
        tang_indices = [i for i in range(ndim) if i != axis]
        v_L = vel_L[tang_indices[0]] if tang_indices else jnp.zeros_like(u_L)
        v_R = vel_R[tang_indices[0]] if tang_indices else jnp.zeros_like(u_R)
        w_L = vel_L[tang_indices[1]] if len(tang_indices) > 1 else None
        w_R = vel_R[tang_indices[1]] if len(tang_indices) > 1 else None

        # HLLC flux
        fluxes = _hllc_flux(rho_L, rho_R, u_L, u_R, v_L, v_R,
                            p_L, p_R, E_L, E_R, gamma, w_L, w_R)

        F_rho = fluxes[0]
        F_rhou_normal = fluxes[1]
        F_rhov_tang = fluxes[2]
        F_E = fluxes[3]

        # Accumulate divergence
        drho_dt = drho_dt - _flux_divergence_1d(F_rho, axis, dx)
        drhou_dt[axis] = drhou_dt[axis] - _flux_divergence_1d(F_rhou_normal, axis, dx)
        if tang_indices:
            drhou_dt[tang_indices[0]] = drhou_dt[tang_indices[0]] - \
                _flux_divergence_1d(F_rhov_tang, axis, dx)
        if len(tang_indices) > 1:
            F_rhow = fluxes[4]
            drhou_dt[tang_indices[1]] = drhou_dt[tang_indices[1]] - \
                _flux_divergence_1d(F_rhow, axis, dx)
        dE_dt = dE_dt - _flux_divergence_1d(F_E, axis, dx)

    # Viscous terms (Navier-Stokes only)
    if mu > 0:
        visc_forces, visc_E = _viscous_rhs(rho, velocity, E, grid, mu)
        for i in range(ndim):
            drhou_dt[i] = drhou_dt[i] + visc_forces[i]
        dE_dt = dE_dt + visc_E

    return drho_dt, tuple(drhou_dt), dE_dt


def make_compressible_step_fn(grid: MACGrid, gamma: float = 1.4,
                               mu: float = 0.0,
                               time_scheme: str = 'rk3'):
    """Create a step function for compressible Euler/NS.

    Returns step_fn(state, dt, extra) -> FluidState
    Compatible with differentiable_simulate.
    """
    ndim = grid.ndim

    def _recover_primitives(rho, rhou, E):
        """Recover velocity and pressure from conservative variables."""
        rho_safe = jnp.maximum(rho, 1e-14)
        vel = tuple(rhou[i] / rho_safe for i in range(ndim))
        p = pressure_from_conservative(rho_safe, rhou, E, gamma)
        p = jnp.maximum(p, 1e-14)
        return vel, p

    def _update(rho, velocity, E):
        """Single Euler step: compute RHS."""
        rhou = tuple(rho * velocity[i] for i in range(ndim))
        p = pressure_from_conservative(rho, rhou, E, gamma)
        state_tmp = FluidState(
            velocity=velocity,
            pressure=jnp.maximum(p, 1e-14),
            scalars={'rho': rho, 'E': E},
        )
        drho, drhou, dE = _compressible_rhs(state_tmp, grid, gamma, mu)
        return drho, drhou, dE

    def step_fn(state, dt, extra, **kwargs):
        rho = state.scalars['rho']
        E = state.scalars['E']
        vel = state.velocity
        rhou = tuple(rho * vel[i] for i in range(ndim))

        if time_scheme == 'rk3':
            # Stage 1: Q1 = Q0 + dt * R(Q0)
            drho, drhou_t, dE = _update(rho, vel, E)
            rho_1 = rho + dt * drho
            rhou_1 = tuple(rhou[i] + dt * drhou_t[i] for i in range(ndim))
            E_1 = E + dt * dE
            rho_1 = jnp.maximum(rho_1, 1e-14)
            vel_1, p_1 = _recover_primitives(rho_1, rhou_1, E_1)

            # Stage 2: Q2 = 0.75 Q0 + 0.25 (Q1 + dt R(Q1))
            drho, drhou_t, dE = _update(rho_1, vel_1, E_1)
            rho_2 = 0.75 * rho + 0.25 * (rho_1 + dt * drho)
            rhou_2 = tuple(0.75 * rhou[i] + 0.25 * (rhou_1[i] + dt * drhou_t[i])
                          for i in range(ndim))
            E_2 = 0.75 * E + 0.25 * (E_1 + dt * dE)
            rho_2 = jnp.maximum(rho_2, 1e-14)
            vel_2, p_2 = _recover_primitives(rho_2, rhou_2, E_2)

            # Stage 3: Q3 = 1/3 Q0 + 2/3 (Q2 + dt R(Q2))
            drho, drhou_t, dE = _update(rho_2, vel_2, E_2)
            rho_f = (1.0 / 3.0) * rho + (2.0 / 3.0) * (rho_2 + dt * drho)
            rhou_f = tuple((1.0 / 3.0) * rhou[i] + (2.0 / 3.0) * (rhou_2[i] + dt * drhou_t[i])
                          for i in range(ndim))
            E_f = (1.0 / 3.0) * E + (2.0 / 3.0) * (E_2 + dt * dE)
        else:
            # Explicit Euler
            drho, drhou_t, dE = _update(rho, vel, E)
            rho_f = rho + dt * drho
            rhou_f = tuple(rhou[i] + dt * drhou_t[i] for i in range(ndim))
            E_f = E + dt * dE

        rho_f = jnp.maximum(rho_f, 1e-14)
        vel_f, p_f = _recover_primitives(rho_f, rhou_f, E_f)

        return FluidState(
            velocity=vel_f,
            pressure=p_f,
            scalars={'rho': rho_f, 'E': E_f},
        )

    return step_fn


# ── Convenience: initial condition helpers ────────────────────

def sod_shock_tube_ic(grid: MACGrid, gamma: float = 1.4,
                       interface_pos: float = 0.5,
                       axis: int = 0):
    """Classic Sod (1978) shock tube initial condition.

    Left:  (ρ, p, u) = (1.0, 1.0, 0.0)
    Right: (ρ, p, u) = (0.125, 0.1, 0.0)
    """
    ndim = grid.ndim
    coords = grid.cell_centers()
    x = coords[axis]
    shape = [1] * ndim
    shape[axis] = len(x)
    x_bc = jnp.reshape(x, shape)
    L = grid.domain_size[axis]
    x_interface = interface_pos * L

    rho = jnp.where(x_bc <= x_interface, 1.0, 0.125)
    p = jnp.where(x_bc <= x_interface, 1.0, 0.1)
    rho = jnp.broadcast_to(rho, grid.cell_shape())
    p = jnp.broadcast_to(p, grid.cell_shape())

    velocity = tuple(jnp.zeros(grid.cell_shape()) for _ in range(ndim))
    E = p / (gamma - 1.0)  # zero KE

    return FluidState(
        velocity=velocity,
        pressure=p,
        scalars={'rho': rho, 'E': E},
    )


def explosion_ic(grid: MACGrid, gamma: float = 1.4,
                 center=None, radius: float = 0.1,
                 p_high: float = 100.0, p_low: float = 1.0,
                 rho_uniform: float = 1.0):
    """High-pressure circular/spherical blast in uniform ambient.

    Creates an expanding shock wave (Sedov-like blast).
    The initial shock Mach number is roughly sqrt(p_high / p_low).
    """
    ndim = grid.ndim
    cc = grid.cell_centers()
    if center is None:
        center = [0.5 * L for L in grid.domain_size]

    # Distance from centre
    dist_sq = sum((jnp.reshape(cc[i], _bcast_shape(i, ndim, grid.shape))
                   - center[i]) ** 2
                  for i in range(ndim))
    dist = jnp.sqrt(dist_sq)

    rho = jnp.full(grid.cell_shape(), rho_uniform)
    p = jnp.where(dist <= radius, p_high, p_low)
    velocity = tuple(jnp.zeros(grid.cell_shape()) for _ in range(ndim))
    E = p / (gamma - 1.0)

    return FluidState(
        velocity=velocity,
        pressure=p,
        scalars={'rho': rho, 'E': E},
    )


def mach_flow_ic(grid: MACGrid, gamma: float = 1.4,
                 Mach: float = 2.0, rho_inf: float = 1.0,
                 p_inf: float = 1.0, axis: int = 0):
    """Uniform supersonic freestream at a given Mach number."""
    ndim = grid.ndim
    c_inf = jnp.sqrt(gamma * p_inf / rho_inf)
    u_inf = Mach * c_inf

    rho = jnp.full(grid.cell_shape(), rho_inf)
    p = jnp.full(grid.cell_shape(), p_inf)
    velocity = []
    for i in range(ndim):
        if i == axis:
            velocity.append(jnp.full(grid.cell_shape(), float(u_inf)))
        else:
            velocity.append(jnp.zeros(grid.cell_shape()))
    velocity = tuple(velocity)

    ke = 0.5 * rho * sum(v ** 2 for v in velocity)
    E = p / (gamma - 1.0) + ke

    return FluidState(
        velocity=velocity,
        pressure=p,
        scalars={'rho': rho, 'E': E},
    )


def _bcast_shape(axis, ndim, shape):
    """Shape tuple for broadcasting a 1D coordinate along *axis*."""
    s = [1] * ndim
    s[axis] = shape[axis]
    return s
