"""
Differentiable jet actuator model for flow control optimization.

Provides Gaussian-profile body-force actuators whose position, direction,
and amplitude are continuous design variables compatible with ``jax.grad``.

The resulting force fields are injected via the solver's
``momentum_source_fn(scalars, velocity, grid)`` interface.

Usage
-----
>>> from diffrl_flow_control.solver.actuators import make_actuator_source_fn
>>> params = {
...     'positions': jnp.array([[5.0, 3.0], [5.0, 3.0]]),  # (N, 2)
...     'directions': jnp.array([0.0, jnp.pi]),             # (N,)
...     'amplitudes': jnp.array([1.0, 1.0]),                 # (N,)
... }
>>> source_fn = make_actuator_source_fn(params, width=0.1)
>>> step_fn = create_step_fn(..., momentum_source_fn=source_fn)
"""

import jax
import jax.numpy as jnp


# ── Surface parameterization ──────────────────────────────────────────

def cylinder_surface_position(theta, center, radius):
    """Map angle theta to (x, y) on cylinder surface.

    Parameters
    ----------
    theta : scalar — angle in [0, 2*pi).
    center : (2,) array — (cx, cy).
    radius : scalar.

    Returns
    -------
    position : (2,) array — (x, y) on surface.
    """
    x = center[0] + radius * jnp.cos(theta)
    y = center[1] + radius * jnp.sin(theta)
    return jnp.array([x, y])


def cylinder_surface_normal(theta):
    """Outward normal direction at angle theta on cylinder surface.

    Returns
    -------
    alpha : scalar — angle of outward normal (same as theta for a circle).
    """
    return theta


def surface_constrained_params(thetas, center, radius, offsets=None,
                                amplitude_scale=None):
    """Convert surface angles to full actuator parameters.

    Parameters
    ----------
    thetas : (N,) array — angles on cylinder surface.
    center : (2,) array — cylinder centre.
    radius : scalar — cylinder radius.
    offsets : (N,) array, optional — angular offset from outward normal.
        Default 0 (pure blowing/suction along normal).
    amplitude_scale : (N,) array, optional — raw amplitude parameters
        (will be passed through softplus for positivity).

    Returns
    -------
    positions : (N, 2) array
    directions : (N,) array
    amplitudes : (N,) array
    """
    positions = jax.vmap(cylinder_surface_position, in_axes=(0, None, None))(
        thetas, center, radius
    )
    normals = jax.vmap(cylinder_surface_normal)(thetas)
    if offsets is None:
        offsets = jnp.zeros_like(thetas)
    directions = normals + offsets

    if amplitude_scale is not None:
        amplitudes = jax.nn.softplus(amplitude_scale)
    else:
        amplitudes = jnp.ones_like(thetas)

    return positions, directions, amplitudes
