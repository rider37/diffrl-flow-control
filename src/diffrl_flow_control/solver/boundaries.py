"""Boundary conditions on the MAC grid.

Four boundary types are used: the parabolic inlet and the zero-gradient outlet of
the cylinder channel, no-slip walls, and periodic sides (Rayleigh--Benard).  A
face without an entry in the configuration is treated as a zero-gradient outlet.
Velocity boundary values are set on the boundary faces (``apply_boundaries``);
the diffusion operator uses ghost cells (``pad_velocity_for_diffusion``,
``pad_scalar_for_diffusion``).  A wall entry may carry a Dirichlet value of a
scalar, e.g. ``{'type': 'wall_no_slip', 'T': 1.0}``; otherwise the scalar is
insulated at that face.
"""
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from diffrl_flow_control.solver.grid import MACGrid

_AXES = ['x', 'y', 'z']


def apply_boundaries(velocity: Tuple[jnp.ndarray, ...], boundaries_config: Dict[str, Any], grid: MACGrid) -> Tuple[jnp.ndarray, ...]:
    """Set the boundary-face values of every velocity component."""
    new_vel = list(velocity)
    for axis in range(grid.ndim):
        for side, is_min in [('min', True), ('max', False)]:
            key = f"{_AXES[axis]}_{side}"
            if key not in boundaries_config:
                continue
            bc = boundaries_config[key]
            for cmp_idx in range(grid.ndim):
                new_vel[cmp_idx] = _apply_bc_to_face(
                    new_vel[cmp_idx], cmp_idx, axis, is_min=is_min, bc=bc, grid=grid)
    return tuple(new_vel)


def _apply_bc_to_face(field: jnp.ndarray, cmp_idx: int, bnd_axis: int, is_min: bool, bc: Dict[str, Any], grid: MACGrid) -> jnp.ndarray:
    """Boundary value of velocity component ``cmp_idx`` on the ``bnd_axis`` face
    (normal component if ``cmp_idx == bnd_axis``, tangential otherwise)."""
    bctype = bc.get('type', 'outlet_zero_gradient')

    idx_bnd = [slice(None)] * grid.ndim
    idx_bnd[bnd_axis] = 0 if is_min else -1
    idx_bnd = tuple(idx_bnd)

    # zero gradient: copy the adjacent interior layer
    if bctype == 'outlet_zero_gradient':
        idx_inner = list(idx_bnd)
        idx_inner[bnd_axis] = 1 if is_min else -2
        idx_inner = tuple(idx_inner)
        return field.at[idx_bnd].set(field[idx_inner])

    # parabolic (Poiseuille) inlet
    elif bctype == 'inlet_parabolic':
        Um = bc.get('Um', 1.5)
        extent_axis = 1 - bnd_axis  # axis perpendicular to the inlet face
        L = grid.domain_size[extent_axis]
        if cmp_idx == bnd_axis:  # normal component: parabolic profile
            n = field.shape[extent_axis]
            if cmp_idx == extent_axis:
                dy = L / (n - 1) if n > 1 else L
                y = jnp.arange(n) * dy
            else:
                dy = L / n
                y = jnp.arange(n) * dy + dy / 2
            profile = Um * 4.0 * y * (L - y) / L**2
            return field.at[idx_bnd].set(profile)
        else:  # tangential component: zero
            return field.at[idx_bnd].set(0.0)

    # no-slip wall: the normal face lies on the wall and is set to zero; the
    # tangential face lies half a cell inside and is left to the ghost cell of
    # pad_velocity_for_diffusion, which places the zero on the wall
    elif bctype == 'wall_no_slip':
        if cmp_idx == bnd_axis:
            return field.at[idx_bnd].set(0.0)
        else:
            return field

    # periodic: handled by the padding of the operators
    elif bctype == 'periodic':
        return field
    else:
        raise ValueError(f"Unknown BC type '{bctype}'. Valid types: outlet_zero_gradient, "
                         f"inlet_parabolic, wall_no_slip, periodic")


def pad_velocity_for_diffusion(velocity: Tuple[jnp.ndarray, ...], boundaries_config: Dict[str, Any], grid: MACGrid) -> Tuple[jnp.ndarray, ...]:
    """Velocity components padded by one ghost cell per side for the Laplacian."""
    padded_vel = []
    for cmp_idx, field in enumerate(velocity):
        pad_widths = [(1, 1) for _ in range(grid.ndim)]
        padded_f = jnp.pad(field, pad_widths, mode='constant')
        for bnd_axis in range(grid.ndim):
            for side, is_min in [('min', True), ('max', False)]:
                key = f"{_AXES[bnd_axis]}_{side}"
                bc = boundaries_config.get(key, {'type': 'outlet_zero_gradient'})
                padded_f = _apply_ghost_pad(padded_f, cmp_idx, bnd_axis, is_min=is_min, bc=bc)
        padded_vel.append(padded_f)
    return tuple(padded_vel)


def _apply_ghost_pad(padded_f: jnp.ndarray, cmp_idx: int, bnd_axis: int, is_min: bool, bc: Dict[str, Any]) -> jnp.ndarray:
    """Ghost value (index 0 or -1 of the padded array) of one boundary."""
    bctype = bc.get('type', 'outlet_zero_gradient')
    ndim = padded_f.ndim
    idx_ghost = [slice(None)] * ndim
    idx_ghost[bnd_axis] = 0 if is_min else -1
    idx_ghost = tuple(idx_ghost)

    idx_first_interior = [slice(None)] * ndim
    idx_first_interior[bnd_axis] = 1 if is_min else -2
    idx_first_interior = tuple(idx_first_interior)

    # zero gradient: ghost = interior
    if bctype == 'outlet_zero_gradient':
        return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])

    # no-slip wall: symmetric normal, antisymmetric tangential ghost
    elif bctype == 'wall_no_slip':
        if cmp_idx == bnd_axis:
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])
        else:
            return padded_f.at[idx_ghost].set(-padded_f[idx_first_interior])

    # parabolic inlet: zero normal gradient, zero tangential velocity on the face
    elif bctype == 'inlet_parabolic':
        if cmp_idx == bnd_axis:
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])
        else:
            return padded_f.at[idx_ghost].set(-padded_f[idx_first_interior])

    # periodic: ghost = opposite interior layer
    elif bctype == 'periodic':
        src = [slice(None)] * ndim
        src[bnd_axis] = -2 if is_min else 1
        return padded_f.at[tuple(idx_ghost)].set(padded_f[tuple(src)])

    return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])


def pad_scalar_for_diffusion(scalar: jnp.ndarray, scalar_name: str, boundaries_config: Dict[str, Any], grid) -> jnp.ndarray:
    """Scalar padded by one ghost cell per side for the Laplacian.

    Dirichlet wall value T_w: ghost = 2 T_w - interior; periodic side: ghost =
    opposite interior layer; otherwise zero normal gradient (edge padding).
    """
    ndim = grid.ndim
    pad_widths = [(1, 1) for _ in range(ndim)]
    padded = jnp.pad(scalar, pad_widths, mode='edge')

    for bnd_axis in range(ndim):
        for is_min in (True, False):
            side = 'min' if is_min else 'max'
            bc = boundaries_config.get(f"{_AXES[bnd_axis]}_{side}", {})

            idx_ghost = [slice(None)] * ndim
            idx_ghost[bnd_axis] = 0 if is_min else -1
            idx_interior = [slice(None)] * ndim
            idx_interior[bnd_axis] = 1 if is_min else -2
            idx_ghost = tuple(idx_ghost)
            idx_interior = tuple(idx_interior)

            bctype = bc.get('type', '')
            if bctype == 'periodic':
                src = [slice(None)] * ndim
                src[bnd_axis] = -2 if is_min else 1
                padded = padded.at[idx_ghost].set(padded[tuple(src)])
            elif scalar_name in bc:
                T_wall = float(bc[scalar_name])
                ghost_val = 2.0 * T_wall - padded[idx_interior]
                padded = padded.at[idx_ghost].set(ghost_val)
    return padded
