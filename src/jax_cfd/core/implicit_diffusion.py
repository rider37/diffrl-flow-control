"""Implicit diffusion via ADI (Alternating Direction Implicit) splitting.

Crank-Nicolson: the molecular viscosity ``nu_base`` is split half-explicit,
half-implicit.  Turbulent viscosity ``nu_t`` is kept fully explicit.

Only non-periodic axes are solved implicitly; periodic axes keep their
diffusion fully explicit (no cyclic tridiagonal needed).

Usage
-----
In the solver, the explicit RHS uses ``nu_eff = nu_t + 0.5*nu_base`` for the
diffusion term.  After the explicit predictor, call::

    vel_star = implicit_solve(vel_star, dt_impl)

where ``dt_impl = 0.5 * dt_substep``.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable
from jax_cfd.core.mac_grid import MACGrid


def _thomas_solve_batched(a, b_full, c, rhs_T):
    """
    Batched tridiagonal solver via ``jax.lax.scan`` (Thomas algorithm).

    Parameters
    ----------
    a : (N,) sub-diagonal coefficients (shared across batch)
    b_full : (..., N) diagonal coefficients (varies per wavenumber)
    c : (N,) super-diagonal coefficients (shared across batch)
    rhs_T : (..., N) right-hand side (last axis is tridiagonal dim)

    Returns
    -------
    x : (..., N) solution array
    """
    # ── Forward sweep ─────────────────────────────────────────────────
    # Init j=0
    denom0 = b_full[..., 0]
    c0 = c[0] / denom0
    d0 = rhs_T[..., 0] / denom0

    def forward_step(carry, xs):
        c_prev, d_prev = carry
        a_j, c_j, b_j, rhs_j = xs
        denom = b_j - a_j * c_prev
        c_new = c_j / denom
        d_new = (rhs_j - a_j * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    # Pack scan inputs: a[1:] and c[1:] are (N-1,); b and rhs are (N-1, ...)
    xs_fwd = (
        a[1:],
        c[1:],
        jnp.moveaxis(b_full[..., 1:], -1, 0),
        jnp.moveaxis(rhs_T[..., 1:], -1, 0),
    )
    _, (c_all, d_all) = jax.lax.scan(forward_step, (c0, d0), xs_fwd)

    # Prepend j=0 values  →  (N, ...)
    c_all = jnp.concatenate([c0[None], c_all], axis=0)
    d_all = jnp.concatenate([d0[None], d_all], axis=0)

    # ── Backward sweep ────────────────────────────────────────────────
    def back_step(carry, xs):
        x_next = carry
        c_j, d_j = xs
        return d_j - c_j * x_next, d_j - c_j * x_next

    x_last = d_all[-1]
    xs_back = (c_all[-2::-1], d_all[-2::-1])
    _, x_rev = jax.lax.scan(back_step, x_last, xs_back)

    # Assemble: reversed interior + last element
    x = jnp.concatenate([x_rev[::-1], x_last[None]], axis=0)   # (N, ...)
    return jnp.moveaxis(x, 0, -1)   # (..., N)


def make_implicit_diffusion_fn(
    grid: MACGrid,
    nu_base: float,
    pad_diff_fn: Callable,
    periodic_axes: Tuple[int, ...] = (),
) -> Callable:
    """Build an ADI implicit diffusion function for Crank-Nicolson splitting.

    Parameters
    ----------
    grid : MACGrid
    nu_base : molecular viscosity (constant)
    pad_diff_fn : ghost-cell padding function ``(vel, grid) -> vel_padded``
    periodic_axes : axes with periodic BCs (kept fully explicit)

    Returns
    -------
    solve_fn(vel, dt_impl) -> vel_corrected
        ``dt_impl = 0.5 * dt_substep`` for CN half-step.
    """

    ndim = grid.ndim
    solve_axes = tuple(ax for ax in range(ndim) if ax not in periodic_axes)

    if not solve_axes:
        # All axes are periodic — nothing to do implicitly
        return lambda vel, dt_impl: vel

    # Precompute stencil geometry per (axis, is_stagger_axis).
    # Actual tridiagonal coefficients are: a = -dt * coeff_a, b = 1 + dt * coeff_b, etc.
    _stencil = {}
    for ax in solve_axes:
        _stencil[(ax, False)] = _transverse_stencil(grid, ax, nu_base)
        _stencil[(ax, True)] = _stagger_stencil(grid, ax, nu_base)

    def solve_fn(
        vel: Tuple[jnp.ndarray, ...],
        dt_impl: float,
    ) -> Tuple[jnp.ndarray, ...]:
        # Ghost cells from BCs (needed for boundary treatment)
        vel_padded = pad_diff_fn(vel, grid)

        for ax in solve_axes:
            new_comps = []
            for i in range(ndim):
                is_stagger = (i == ax)
                u = vel[i]
                u_pad = vel_padded[i]
                ca, cb, cc = _stencil[(ax, is_stagger)]
                u_new = _solve_component(
                    u, u_pad, ax, ndim, dt_impl, ca, cb, cc,
                    is_stagger, _thomas_solve_batched,
                )
                new_comps.append(u_new)
            vel = tuple(new_comps)
            # Re-pad for the next axis sweep
            if ax != solve_axes[-1]:
                vel_padded = pad_diff_fn(vel, grid)

        return vel

    return solve_fn


# ── Stencil coefficient builders ──────────────────────────────────────────

def _transverse_stencil(grid, axis, nu):
    """Coefficients for a cell-centred field along *axis* (M = N_axis points).

    Used when solving velocity component ``i != axis``.
    """
    N = grid.shape[axis]
    dx_arr = grid.spacings_array(axis)

    if isinstance(dx_arr, (int, float)):
        coeff = nu / (dx_arr * dx_arr)
        return (jnp.full(N, coeff),
                jnp.full(N, 2.0 * coeff),
                jnp.full(N, coeff))

    # Non-uniform: cell-centred stencil (matches diff_axis_padded_nu)
    d_plus = jnp.empty(N)
    d_plus = d_plus.at[:N - 1].set(0.5 * (dx_arr[:N - 1] + dx_arr[1:N]))
    d_plus = d_plus.at[N - 1].set(dx_arr[N - 1])  # to ghost

    d_minus = jnp.empty(N)
    d_minus = d_minus.at[0].set(dx_arr[0])          # from ghost
    d_minus = d_minus.at[1:N].set(0.5 * (dx_arr[:N - 1] + dx_arr[1:N]))

    h = dx_arr
    return (nu / (d_minus * h),
            nu * (1.0 / d_plus + 1.0 / d_minus) / h,
            nu / (d_plus * h))


def _stagger_stencil(grid, axis, nu):
    """Coefficients for interior face points along *axis* (M = N_axis - 1).

    Used when solving velocity component ``i == axis`` — boundary faces
    are Dirichlet (known values) and are excluded from the tridiagonal.
    """
    N = grid.shape[axis]
    M = N - 1  # interior faces
    dx_arr = grid.spacings_array(axis)

    if isinstance(dx_arr, (int, float)):
        coeff = nu / (dx_arr * dx_arr)
        return (jnp.full(M, coeff),
                jnp.full(M, 2.0 * coeff),
                jnp.full(M, coeff))

    # Non-uniform: face-interior stencil.
    # Interior face k (1-based in full field, 0-based here) sits between
    # cell k and cell k+1.  Distances:
    d_plus = dx_arr[1:N]      # to the next face (=cell width k+1), (M,)
    d_minus = dx_arr[:N - 1]  # from the prev face (=cell width k),  (M,)
    h = 0.5 * (d_minus + d_plus)
    return (nu / (d_minus * h),
            nu * (1.0 / d_plus + 1.0 / d_minus) / h,
            nu / (d_plus * h))


# ── Per-component tridiagonal solve ──────────────────────────────────────

def _solve_component(u, u_pad, axis, ndim, dt_impl,
                     coeff_a, coeff_b, coeff_c,
                     is_stagger, thomas_solve):
    """Solve the implicit diffusion tridiagonal for one velocity component
    along one axis.

    Parameters
    ----------
    u : the velocity component array (interior, no ghost cells)
    u_pad : the same component fully padded (ghost cells on ALL axes)
    axis : the axis being solved
    ndim : number of dimensions
    dt_impl : 0.5 * dt_substep  (CN half-step)
    coeff_a, coeff_b, coeff_c : precomputed stencil geometry (M,)
    is_stagger : True if this component's stagger axis == solve axis
    thomas_solve : the batched Thomas solver function
    """
    if is_stagger:
        # Solve for interior faces only (exclude boundary points 0 and -1)
        sl_int = [slice(None)] * ndim
        sl_int[axis] = slice(1, -1)
        u_solve = u[tuple(sl_int)]

        # Ghost values = boundary face values (Dirichlet)
        sl_left = [slice(None)] * ndim
        sl_left[axis] = 0
        sl_right = [slice(None)] * ndim
        sl_right[axis] = -1
        ghost_left = u[tuple(sl_left)]
        ghost_right = u[tuple(sl_right)]
    else:
        # Solve for all points; ghosts come from BC padding
        u_solve = u

        # Extract ghost cells for this axis from fully padded field:
        # strip padding on all other axes, keep padding on target axis.
        sl_strip = [slice(1, -1)] * ndim
        sl_strip[axis] = slice(None)
        u_ax_pad = u_pad[tuple(sl_strip)]  # (..., M+2, ...)

        sl_left = [slice(None)] * ndim
        sl_left[axis] = 0
        sl_right = [slice(None)] * ndim
        sl_right[axis] = -1
        ghost_left = u_ax_pad[tuple(sl_left)]
        ghost_right = u_ax_pad[tuple(sl_right)]

    M = u_solve.shape[axis]

    # Build tridiagonal: scale precomputed geometry by dt_impl
    a = -dt_impl * coeff_a          # (M,)
    b = 1.0 + dt_impl * coeff_b    # (M,)
    c = -dt_impl * coeff_c          # (M,)

    # Move solve axis to last position → (..., M)
    rhs = jnp.moveaxis(u_solve, axis, -1)

    # Absorb boundary ghost values into RHS (first & last row)
    rhs = rhs.at[..., 0].add(dt_impl * coeff_a[0] * ghost_left)
    rhs = rhs.at[..., -1].add(dt_impl * coeff_c[-1] * ghost_right)
    a = a.at[0].set(0.0)
    c = c.at[-1].set(0.0)

    # Broadcast b to match rhs batch shape
    b_full = jnp.broadcast_to(b, rhs.shape)

    # Thomas solve
    x = thomas_solve(a, b_full, c, rhs)

    # Move axis back
    x = jnp.moveaxis(x, -1, axis)

    if is_stagger:
        # Put solved interior back; keep original boundary faces
        sl_int2 = [slice(None)] * ndim
        sl_int2[axis] = slice(1, -1)
        return u.at[tuple(sl_int2)].set(x)
    else:
        return x
