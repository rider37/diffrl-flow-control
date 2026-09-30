"""MUSCL momentum advection and spectral Poisson solvers for the pressure
projection on a uniform MAC grid."""
from functools import partial
from typing import Tuple

import jax.numpy as jnp

from diffrl_flow_control.solver.advection import advect_momentum_muscl
from diffrl_flow_control.solver.grid import MACGrid


def muscl_advection(velocity: Tuple[jnp.ndarray, ...], grid: MACGrid, periodic_axes: Tuple[int, ...] = ()) -> Tuple[jnp.ndarray, ...]:
    """MUSCL advection with the van Leer limiter (TVD, second order in smooth regions)."""
    return advect_momentum_muscl(velocity, grid, periodic_axes=periodic_axes)


def make_muscl_advection(periodic_axes=()):
    """Return ``fn(velocity, grid)`` with the periodic axes bound."""
    if not periodic_axes:
        return muscl_advection
    return partial(muscl_advection, periodic_axes=periodic_axes)


def solve_pressure_fft(rhs: jnp.ndarray, grid: MACGrid) -> jnp.ndarray:
    """Spectral Poisson solver (DCT-II) for Neumann boundaries. N-dimensional."""
    import jax.scipy.fft as jfft

    # Forward DCT along each axis
    p_dct = rhs
    for ax in range(grid.ndim):
        p_dct = jfft.dct(p_dct, axis=ax, type=2)

    # N-D eigenvalue denominator via broadcasting
    denom = jnp.zeros(grid.cell_shape())
    for ax in range(grid.ndim):
        n_ax = grid.shape[ax]
        dx_ax = grid.spacings[ax]
        k = jnp.arange(n_ax)
        lam = 2.0 * (jnp.cos(jnp.pi * k / n_ax) - 1.0) / (dx_ax ** 2)
        shape = [1] * grid.ndim
        shape[ax] = n_ax
        denom = denom + lam.reshape(shape)

    denom = jnp.where(denom == 0.0, 1.0, denom)
    p_dct = p_dct / denom
    p_dct = p_dct.at[tuple(0 for _ in range(grid.ndim))].set(0.0)

    # Inverse DCT along each axis
    p = p_dct
    for ax in range(grid.ndim):
        p = jfft.idct(p, axis=ax, type=2)

    return p - jnp.mean(p)

def make_pressure_solver_fft_periodic(periodic_axes):
    """
    Factory: FFT Poisson solver for **uniform** grids with arbitrary
    periodic/Neumann axis combinations.

    Transform order: DCT (Neumann, real) axes first, then FFT (periodic)
    axes.  This ensures DCT always operates on real data.
    """
    periodic_set = frozenset(periodic_axes)

    def solve(rhs: jnp.ndarray, grid: MACGrid) -> jnp.ndarray:
        import jax.scipy.fft as jfft

        ndim = grid.ndim
        nx_orig = {ax: grid.shape[ax] for ax in range(ndim)}

        # ── Forward: DCT on Neumann axes first (data stays real) ─────
        data = rhs
        for ax in range(ndim):
            if ax not in periodic_set:
                data = jfft.dct(data, axis=ax, type=2)

        # ── Forward: FFT on periodic axes (data becomes complex) ─────
        rfft_axis = None
        for ax in range(ndim):
            if ax in periodic_set:
                if rfft_axis is None:
                    data = jnp.fft.rfft(data, axis=ax)
                    rfft_axis = ax
                else:
                    data = jnp.fft.fft(data, axis=ax)

        # ── Eigenvalue denominator ───────────────────────────────────
        lam_shape = [1] * ndim
        for ax in range(ndim):
            lam_shape[ax] = data.shape[ax]
        denom = jnp.zeros(lam_shape)

        for ax in range(ndim):
            n_ax = grid.shape[ax]
            dx_ax = grid.spacings[ax]
            if ax in periodic_set:
                k = jnp.arange(data.shape[ax])
                lam = 2.0 * (jnp.cos(2.0 * jnp.pi * k / n_ax) - 1.0) / dx_ax**2
            else:
                k = jnp.arange(n_ax)
                lam = 2.0 * (jnp.cos(jnp.pi * k / n_ax) - 1.0) / dx_ax**2
            shape_k = [1] * ndim
            shape_k[ax] = len(k)
            denom = denom + lam.reshape(shape_k)

        denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
        data = data / denom

        # Pin DC mode (fully spectral domain — safe here)
        dc_idx = tuple(0 for _ in range(ndim))
        data = data.at[dc_idx].set(
            0.0 + 0.0j if jnp.iscomplexobj(data) else 0.0
        )

        # ── Inverse: periodic axes first (complex → real) ───────────
        p = data
        for ax in reversed(range(ndim)):
            if ax in periodic_set:
                if ax == rfft_axis:
                    p = jnp.fft.irfft(p, n=nx_orig[ax], axis=ax)
                else:
                    p = jnp.fft.ifft(p, axis=ax)

        p = p.real if jnp.iscomplexobj(p) else p

        # ── Inverse: IDCT on Neumann axes (real → real) ─────────────
        for ax in reversed(range(ndim)):
            if ax not in periodic_set:
                p = jfft.idct(p, axis=ax, type=2)

        return p - jnp.mean(p)

    return solve
