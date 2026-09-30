"""Moving Reference Frame (MRF) — rotating frame source terms.

When solving in a frame rotating at angular velocity Ω, the momentum equation
gains two fictitious-force source terms:

    Coriolis:     -2 Ω × u
    Centrifugal:  -Ω × (Ω × r)

These are returned as a ``momentum_source_fn`` compatible with ``create_step_fn``.

Usage (via runner CONFIG):
    'mrf': {
        'omega': 10.0,                   # angular speed (rad/s)
        'axis': [0, 0, 1],              # rotation axis (3D) or ignored (2D → z-axis)
        'center': [0.5, 0.5],           # rotation centre
    }
"""
from __future__ import annotations

import jax.numpy as jnp
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.operators import interp_to_face


def _cell_to_face(field_cc, axis, ndim):
    """Interpolate cell-centred field to face centres along *axis*."""
    pad_w = [(0, 0)] * ndim
    pad_w[axis] = (1, 1)
    padded = jnp.pad(field_cc, pad_w, mode='edge')
    return interp_to_face(padded, axis)


def make_mrf_source_fn(grid: MACGrid, omega: float, center: tuple):
    """Build a momentum_source_fn for a rotating reference frame.

    Computes Coriolis + centrifugal forces at cell centres, then
    interpolates to face centres for each momentum component.

    Parameters
    ----------
    grid   : MACGrid
    omega  : scalar angular velocity (rad/s) about z-axis
    center : rotation centre (cx, cy[, cz])

    Returns
    -------
    source_fn : (scalars, velocity, grid) -> tuple of face-centred force arrays
    """
    ndim = grid.ndim
    cx = float(center[0])
    cy = float(center[1])

    # Precompute cell-centred position vectors relative to rotation centre
    cc = grid.cell_centers()
    s_x = [1] * ndim
    s_x[0] = grid.shape[0]
    s_y = [1] * ndim
    s_y[1] = grid.shape[1]
    rx_cc = jnp.reshape(cc[0], s_x) - cx  # broadcast-ready
    ry_cc = jnp.reshape(cc[1], s_y) - cy

    if ndim == 2:
        def source_fn(scalars, velocity, grid_arg):
            # Velocity at cell centres (face -> cell average)
            u_cc = interp_to_face(velocity[0], 0)
            v_cc = interp_to_face(velocity[1], 1)

            # Coriolis: -2Ω × u  →  (+2Ω·v, -2Ω·u) for z-axis rotation
            # Centrifugal: -Ω × (Ω × r)  →  (+Ω²·rx, +Ω²·ry)
            Fx_cc = 2.0 * omega * v_cc + omega ** 2 * rx_cc
            Fy_cc = -2.0 * omega * u_cc + omega ** 2 * ry_cc

            # Interpolate to face centres
            return (_cell_to_face(Fx_cc, 0, ndim),
                    _cell_to_face(Fy_cc, 1, ndim))

        return source_fn

    else:  # 3D
        def source_fn(scalars, velocity, grid_arg):
            u_cc = interp_to_face(velocity[0], 0)
            v_cc = interp_to_face(velocity[1], 1)

            # z-axis rotation only
            Fx_cc = 2.0 * omega * v_cc + omega ** 2 * rx_cc
            Fy_cc = -2.0 * omega * u_cc + omega ** 2 * ry_cc

            return (_cell_to_face(Fx_cc, 0, ndim),
                    _cell_to_face(Fy_cc, 1, ndim),
                    jnp.zeros(grid.face_shape(2)))

        return source_fn
