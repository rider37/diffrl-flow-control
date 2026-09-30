import jax.numpy as jnp


def cylinder_coords_diff(
    radius,
    center,
    n_pts: int,
):
    """Differentiable cylinder geometry using pure JAX operations.

    All inputs except *n_pts* are JAX-traced scalars so that ``jax.grad``
    can propagate through the geometry.

    Parameters
    ----------
    radius : jnp scalar
        Cylinder radius (traced).
    center : tuple of two jnp scalars
        (cx, cy) centre coordinates (traced).
    n_pts : int
        Number of Lagrangian markers (static — determines array shape).

    Returns
    -------
    coords : (xs, ys) tuple of jnp arrays, each shape (n_pts,)
    ds_arr : jnp array shape (n_pts,)  — arc-length per marker.
    """
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_pts + 1)[:-1]
    xs = center[0] + radius * jnp.cos(theta)
    ys = center[1] + radius * jnp.sin(theta)
    ds_arr = jnp.full(n_pts, 2.0 * jnp.pi * radius / n_pts)
    return (xs, ys), ds_arr
