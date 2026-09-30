import jax.numpy as jnp
from typing import Tuple
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.advection import (advect_momentum_central,
                                     advect_momentum_muscl,
                                     advect_momentum_weno5)

# ======================================================================
# Advection Schemes
# ======================================================================

def central_advection(velocity: Tuple[jnp.ndarray, ...], grid: MACGrid, periodic_axes: Tuple[int, ...] = ()) -> Tuple[jnp.ndarray, ...]:
    """2nd-order central scheme. Stable for Pe < 2."""
    return advect_momentum_central(velocity, grid, periodic_axes=periodic_axes)

def muscl_advection(velocity: Tuple[jnp.ndarray, ...], grid: MACGrid, periodic_axes: Tuple[int, ...] = ()) -> Tuple[jnp.ndarray, ...]:
    """MUSCL Van Leer limited scheme. TVD, stable for all Peclet numbers."""
    return advect_momentum_muscl(velocity, grid, periodic_axes=periodic_axes)

def weno5_advection(velocity: Tuple[jnp.ndarray, ...], grid: MACGrid, periodic_axes: Tuple[int, ...] = ()) -> Tuple[jnp.ndarray, ...]:
    """5th-order WENO scheme. High accuracy with sharp gradient preservation."""
    return advect_momentum_weno5(velocity, grid, periodic_axes=periodic_axes)

def get_advection_scheme(name: str):
    schemes = {
        'central':  central_advection,
        'muscl':    muscl_advection,
        'weno5':    weno5_advection,
    }
    if name.lower() not in schemes:
        raise ValueError(f"Unknown advection scheme '{name}'. Valid: {list(schemes.keys())}")
    return schemes[name.lower()]


def make_advection_scheme(name: str, periodic_axes=()):
    """
    Return ``fn(velocity, grid)`` for the named scheme with *periodic_axes* baked in.
    Use this factory in the runner when some axes are periodic.
    """
    from functools import partial
    base = get_advection_scheme(name)
    if not periodic_axes:
        return base
    return partial(base, periodic_axes=periodic_axes)

# ======================================================================
# Pressure Linear Solvers
# ======================================================================

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


def get_pressure_solver(name: str):
    solvers = {
        'fft': solve_pressure_fft,
    }
    if name.lower() not in solvers:
        raise ValueError(f"Unknown pressure solver '{name}'. Valid: {list(solvers.keys())}")
    return solvers[name.lower()]


# ======================================================================
# Hybrid FFT-Eigen Pressure Solver (for non-uniform grids)
# ======================================================================

def make_pressure_solver_hybrid(grid, periodic_axes=()):
    """
    Factory for a hybrid FFT + Eigen-decomposition pressure solver.

    Applies FFT (DCT-II or rFFT) along all **uniform** axes, then solves the
    stretched axis via a precomputed eigen-decomposition of the symmetrised
    tridiagonal Laplacian.  This replaces the sequential Thomas algorithm
    (``jax.lax.scan``) with two batched GEMM operations — much faster on GPU.

    Precomputation (done once at construction in float64):
      1. Build T_y  (non-symmetric tridiagonal, Neumann BC)
      2. Symmetrise: S = D^{1/2} T_y D^{-1/2},  D = diag(dy)
      3. Eigen-decompose: S = V Λ V^T  (V orthogonal, cond(V) = 1)

    Per-solve (inside JIT):
      p = D^{-1/2} V diag(1/(λ_i + shift)) V^T D^{1/2} rhs

    Parameters
    ----------
    grid : MACGrid
        Grid with ``stretch`` attribute (used for precomputation).
    periodic_axes : tuple of int
        Axes with periodic BC (use rFFT instead of DCT).

    Returns
    -------
    solve_fn : callable ``(rhs, grid) -> p``
    """
    import numpy as np

    stretch_axis = grid.stretch['axis']
    Ny = grid.shape[stretch_axis]
    dy_np = np.array(grid.spacings_array(stretch_axis), dtype=np.float64)

    # ── Tridiagonal coefficients (float64 for accuracy) ─────────────
    d_plus = np.empty(Ny)
    d_plus[:Ny-1] = 0.5 * (dy_np[:Ny-1] + dy_np[1:Ny])
    d_plus[Ny-1]  = dy_np[Ny-1]

    d_minus = np.empty(Ny)
    d_minus[0]   = dy_np[0]
    d_minus[1:Ny] = 0.5 * (dy_np[:Ny-1] + dy_np[1:Ny])

    a_np = 1.0 / (d_minus * dy_np)
    c_np = 1.0 / (d_plus  * dy_np)
    a_np[0]     = 0.0          # Neumann BC
    c_np[Ny-1]  = 0.0
    b_np = -(a_np + c_np)

    # ── Symmetrise: S = D^{1/2} T_y D^{-1/2} ──────────────────────
    S = np.diag(b_np)
    if Ny > 1:
        sub_diag = a_np[1:] * np.sqrt(dy_np[1:] / dy_np[:-1])
        sup_diag = c_np[:-1] * np.sqrt(dy_np[:-1] / dy_np[1:])
        S += np.diag(sub_diag, -1) + np.diag(sup_diag, 1)

    # ── Eigen-decompose (float64, then cast to float32) ─────────────
    lambda_y_np, V_np = np.linalg.eigh(S)

    V_jax        = jnp.array(V_np,              dtype=jnp.float32)
    VT_jax       = jnp.array(V_np.T,            dtype=jnp.float32)
    lambda_y_jax = jnp.array(lambda_y_np,       dtype=jnp.float32)
    sqrt_dy      = jnp.array(np.sqrt(dy_np),    dtype=jnp.float32)
    inv_sqrt_dy  = jnp.array(1.0 / np.sqrt(dy_np), dtype=jnp.float32)

    def solve(rhs: jnp.ndarray, grid: MACGrid) -> jnp.ndarray:
        import jax.scipy.fft as jfft

        ndim = grid.ndim

        # ── Forward FFT on uniform axes ───────────────────────────────
        # DCT (Neumann) axes first while data is real, then periodic axes.
        data = rhs
        nx_orig = {}
        rfft_axis = None
        for ax in range(ndim):
            if ax == stretch_axis:
                continue
            nx_orig[ax] = grid.shape[ax]
        # Pass 1: DCT on Neumann axes (real → real)
        for ax in range(ndim):
            if ax == stretch_axis or ax in periodic_axes:
                continue
            data = jfft.dct(data, axis=ax, type=2)
        # Pass 2: FFT on periodic axes (real → complex)
        for ax in range(ndim):
            if ax == stretch_axis or ax not in periodic_axes:
                continue
            if rfft_axis is None:
                data = jnp.fft.rfft(data, axis=ax)
                rfft_axis = ax
            else:
                data = jnp.fft.fft(data, axis=ax)

        # ── Eigenvalue sum for uniform axes ───────────────────────────
        lam_shape = [1] * ndim
        for ax in range(ndim):
            if ax != stretch_axis:
                lam_shape[ax] = data.shape[ax]
        lambda_sum = jnp.zeros(lam_shape)

        for ax in range(ndim):
            if ax == stretch_axis:
                continue
            n_ax = grid.shape[ax]
            dx_ax = grid.spacings[ax]
            if ax in periodic_axes:
                k = jnp.arange(data.shape[ax])
                lam = 2.0 * (jnp.cos(2.0 * jnp.pi * k / n_ax) - 1.0) / dx_ax**2
            else:
                k = jnp.arange(n_ax)
                lam = 2.0 * (jnp.cos(jnp.pi * k / n_ax) - 1.0) / dx_ax**2
            shape_k = [1] * ndim
            shape_k[ax] = len(k)
            lambda_sum = lambda_sum + lam.reshape(shape_k)

        # ── Move stretch axis to last for batched matmul ──────────────
        lam_moved = jnp.moveaxis(lambda_sum, stretch_axis, -1)   # (..., 1)
        lam_sq    = lam_moved[..., 0]                              # (...)

        data_moved = jnp.moveaxis(data, stretch_axis, -1)         # (..., Ny)

        # ── Eigen solve: p = D^{-1/2} V (Λ+shift)^{-1} V^T D^{1/2} rhs
        rhs_scaled = data_moved * sqrt_dy                          # (..., Ny)
        coeff      = rhs_scaled @ V_jax                            # (..., Ny)

        denom = lambda_y_jax + lam_sq[..., None]                   # (..., Ny)
        denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)    # DC regularisation
        coeff = coeff / denom

        p_moved = (coeff @ VT_jax) * inv_sqrt_dy                  # (..., Ny)
        p_hat   = jnp.moveaxis(p_moved, -1, stretch_axis)

        # ── Inverse FFT on uniform axes ───────────────────────────────
        # Periodic inverse first (complex → real), then IDCT (real → real).
        p = p_hat
        for ax in reversed(range(ndim)):
            if ax == stretch_axis or ax not in periodic_axes:
                continue
            if ax == rfft_axis:
                p = jnp.fft.irfft(p, n=nx_orig[ax], axis=ax)
            else:
                p = jnp.fft.ifft(p, axis=ax)
        p = p.real if jnp.iscomplexobj(p) else p
        for ax in reversed(range(ndim)):
            if ax == stretch_axis or ax in periodic_axes:
                continue
            p = jfft.idct(p, axis=ax, type=2)
        return p - jnp.mean(p)

    return solve

