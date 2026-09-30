"""Sponge layer for non-reflective outflow boundaries.

Adds a damping source term  -σ(x)(u - u_ref)  to the momentum equations
in a buffer region near the domain boundary.  The damping coefficient σ
ramps smoothly from 0 (interior) to σ_max (boundary) using a cos² profile.

Usage (via runner CONFIG):
    'sponge': {
        'axes': ['x_max'],          # which boundaries to damp
        'width': 0.15,              # fraction of domain length
        'strength': 10.0,           # max damping coefficient σ_max
        'target': [1.0, 0.0],       # reference velocity (usually freestream)
    }
"""
from __future__ import annotations

import jax.numpy as jnp
from jax_cfd.core.mac_grid import MACGrid


def make_sponge_field(grid: MACGrid, config: dict):
    """Build cell-centred damping coefficient array σ(x).

    Returns (sigma, u_target) where sigma has shape grid.cell_shape().
    """
    axes_cfg = config.get('axes', ['x_max'])
    width_frac = config.get('width', 0.15)
    strength = config.get('strength', 10.0)
    u_target = tuple(config.get('target', [1.0] + [0.0] * (grid.ndim - 1)))

    sigma = jnp.zeros(grid.cell_shape())
    coords = grid.cell_centers()

    for tag in axes_cfg:
        parts = tag.split('_')            # e.g. ['x', 'max']
        axis = 'xyz'.index(parts[0])
        is_max = parts[1] == 'max'
        L = grid.domain_size[axis]
        sponge_width = width_frac * L

        x = coords[axis]                  # 1-D array
        shape = [1] * grid.ndim
        shape[axis] = len(x)
        x = jnp.reshape(x, shape)         # broadcast-ready

        if is_max:
            x_start = L - sponge_width
            xi = jnp.clip((x - x_start) / sponge_width, 0.0, 1.0)
        else:
            xi = jnp.clip((sponge_width - x) / sponge_width, 0.0, 1.0)

        # cos² ramp: 0 at interior edge, strength at boundary
        sigma = sigma + strength * xi ** 2

    return sigma, u_target


def make_sponge_source_fn(grid: MACGrid, sigma: jnp.ndarray, u_target: tuple):
    """Return a momentum_source_fn that applies sponge damping.

    Signature: (scalars, velocity, grid) -> tuple of face-centred force arrays.
    """
    from jax_cfd.core.advection import interpolate_to_face

    def source_fn(scalars, velocity, grid_arg):
        forces = []
        for i, u_i in enumerate(velocity):
            # Interpolate cell-centred σ to i-faces
            sigma_face = interpolate_to_face(
                jnp.pad(sigma, [(0, 0)] * i + [(1, 1)] + [(0, 0)] * (grid.ndim - i - 1),
                        mode='edge'),
                i,
            )
            forces.append(-sigma_face * (u_i - u_target[i]))
        return tuple(forces)

    return source_fn
