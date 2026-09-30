"""Arbitrary Lagrangian-Eulerian (ALE) moving mesh framework.

Provides mesh velocity computation and the ALE correction to the
advection term.  In ALE the convective velocity is ``u - u_mesh``
so the mesh can track moving boundaries or deforming interfaces.

Usage (via runner CONFIG)::

    'ale': {
        'type': 'prescribed',       # or 'laplacian'
        'velocity_fn': <callable>,  # f(t, grid) -> mesh_vel tuple
    }

Theory
------
The ALE form of the incompressible NS momentum equation is:

    ∂u/∂t + (u - w)·∇u = -∇p + ν∇²u

where w is the mesh (grid) velocity.  The continuity constraint
becomes ∇·u = 0 (fluid incompressibility) and the mesh must satisfy
the Geometric Conservation Law (GCL):

    dJ/dt = J ∇·w

where J is the Jacobian of the coordinate mapping.

This module provides:
1. Prescribed mesh velocity (user-defined function of time)
2. Laplacian smoothing for mesh velocity diffusion from boundaries
3. ALE advection modifier that subtracts mesh velocity from convective velocity
"""
from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp
from jax_cfd.core.mac_grid import MACGrid


def prescribed_mesh_velocity(
    velocity_fn: Callable,
    t: float,
    grid: MACGrid,
) -> Tuple[jnp.ndarray, ...]:
    """Evaluate a prescribed mesh velocity field at time *t*.

    Parameters
    ----------
    velocity_fn : callable
        ``f(t, grid) -> tuple of face-centred arrays``
    t : float
        Current simulation time.
    grid : MACGrid

    Returns
    -------
    w : tuple of jnp.ndarray
        Face-centred mesh velocity components.
    """
    return velocity_fn(t, grid)


def laplacian_smooth_mesh_velocity(
    w_boundary: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
    n_iter: int = 20,
    omega: float = 0.5,
) -> Tuple[jnp.ndarray, ...]:
    """Diffuse boundary mesh velocity into the interior via Jacobi iteration.

    Solves  ∇²w = 0  with w = w_boundary on boundaries.
    This distributes the mesh deformation smoothly across the domain.

    Parameters
    ----------
    w_boundary : tuple of face-centred arrays
        Non-zero only at moving boundaries.
    grid : MACGrid
    n_iter : int
        Number of Jacobi relaxation sweeps.
    omega : float
        Under-relaxation factor (0 < ω ≤ 1).

    Returns
    -------
    w : tuple of face-centred arrays
        Smoothed mesh velocity field.
    """
    ndim = grid.ndim
    result = list(w_boundary)

    for cmp in range(ndim):
        w = result[cmp]
        mask = jnp.abs(w) > 1e-20  # boundary mask

        for _ in range(n_iter):
            # Average of neighbours along each axis
            avg = jnp.zeros_like(w)
            count = 0
            for ax in range(ndim):
                pad_w = [(0, 0)] * ndim
                pad_w[ax] = (1, 1)
                w_pad = jnp.pad(w, pad_w, mode='edge')
                sl_p = [slice(None)] * ndim
                sl_m = [slice(None)] * ndim
                sl_p[ax] = slice(2, None)
                sl_m[ax] = slice(None, -2)
                avg = avg + w_pad[tuple(sl_p)] + w_pad[tuple(sl_m)]
                count += 2
            avg = avg / count
            w = jnp.where(mask, w, omega * avg + (1 - omega) * w)

        result[cmp] = w

    return tuple(result)


def make_ale_advection_modifier(
    mesh_vel_fn: Callable,
) -> Callable:
    """Build a modifier that subtracts mesh velocity from convective velocity.

    Returns a function ``modify_vel(velocity, t, grid) -> velocity_ale``
    where ``velocity_ale = velocity - mesh_velocity``.  This modified
    velocity should be used in the advection term: ``(u-w)·∇u``.

    Parameters
    ----------
    mesh_vel_fn : callable
        ``f(t, grid) -> tuple of face-centred mesh velocity arrays``

    Returns
    -------
    modify_fn : callable
        ``(velocity, t, grid) -> tuple`` of ALE-corrected convective velocity.
    """
    def modify_fn(velocity, t, grid):
        w = mesh_vel_fn(t, grid)
        return tuple(u - wi for u, wi in zip(velocity, w))

    return modify_fn


def gcl_jacobian_update(
    J: jnp.ndarray,
    mesh_vel: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
    dt: float,
) -> jnp.ndarray:
    """Update the mesh Jacobian via the Geometric Conservation Law.

    dJ/dt = J * ∇·w

    Uses forward Euler.  The Jacobian J is cell-centred and measures
    the local volume change of the mapped cell.

    Parameters
    ----------
    J : cell-centred Jacobian array
    mesh_vel : face-centred mesh velocity
    grid : MACGrid
    dt : float

    Returns
    -------
    J_new : updated Jacobian
    """
    from jax_cfd.core.operators import divergence
    div_w = divergence(mesh_vel, grid)
    return J + dt * J * div_w
