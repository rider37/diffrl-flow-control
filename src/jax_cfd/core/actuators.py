"""
Differentiable jet actuator model for flow control optimization.

Provides Gaussian-profile body-force actuators whose position, direction,
and amplitude are continuous design variables compatible with ``jax.grad``.

The resulting force fields are injected via the solver's
``momentum_source_fn(scalars, velocity, grid)`` interface.

Usage
-----
>>> from jax_cfd.core.actuators import make_actuator_source_fn
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
from typing import Dict, Tuple, Optional
from jax_cfd.core.mac_grid import MACGrid


# ── Single actuator force ──────────────────────────────────────────────

def _gaussian_2d(x, y, x0, y0, sigma):
    """Normalized 2D Gaussian: G(r; sigma) = 1/(2*pi*sigma^2) * exp(-r^2/(2*sigma^2))."""
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return jnp.exp(-r2 / (2.0 * sigma ** 2)) / (2.0 * jnp.pi * sigma ** 2)


def jet_actuator_force_cell(grid, position, direction, amplitude, width):
    """Compute body force from a single jet actuator on the cell-centred grid.

    Parameters
    ----------
    grid : MACGrid
    position : (2,) array — (x0, y0) location of the jet.
    direction : scalar — angle alpha (radians), 0=+x, pi/2=+y.
    amplitude : scalar — force magnitude.
    width : scalar — Gaussian width sigma.

    Returns
    -------
    (Fx, Fy) : tuple of cell-centred arrays, each shape ``grid.cell_shape()``.
    """
    xc, yc = grid.cell_centers()
    X, Y = jnp.meshgrid(xc, yc, indexing='ij')

    x0, y0 = position[0], position[1]
    G = _gaussian_2d(X, Y, x0, y0, width)

    Fx = amplitude * jnp.cos(direction) * G
    Fy = amplitude * jnp.sin(direction) * G
    return (Fx, Fy)


def jet_actuator_force_mac(grid, position, direction, amplitude, width):
    """Compute body force from a single jet actuator on MAC staggered grid.

    Forces are placed on the correct face locations:
    - Fx on u-faces (Nx+1, Ny)
    - Fy on v-faces (Nx, Ny+1)

    Parameters
    ----------
    grid : MACGrid
    position : (2,) array — (x0, y0).
    direction : scalar — angle alpha.
    amplitude : scalar — force magnitude.
    width : scalar — Gaussian width sigma.

    Returns
    -------
    (Fx, Fy) : tuple of face-centred arrays matching ``grid.face_shape(axis)``.
    """
    x0, y0 = position[0], position[1]

    # u-face grid (axis=0)
    xu, yu = grid.face_centers(0)
    Xu, Yu = jnp.meshgrid(xu, yu, indexing='ij')
    Gu = _gaussian_2d(Xu, Yu, x0, y0, width)
    Fx = amplitude * jnp.cos(direction) * Gu

    # v-face grid (axis=1)
    xv, yv = grid.face_centers(1)
    Xv, Yv = jnp.meshgrid(xv, yv, indexing='ij')
    Gv = _gaussian_2d(Xv, Yv, x0, y0, width)
    Fy = amplitude * jnp.sin(direction) * Gv

    return (Fx, Fy)


def multi_actuator_force_mac(grid, positions, directions, amplitudes, width):
    """Compute summed force from N actuators on MAC staggered grid.

    Parameters
    ----------
    grid : MACGrid
    positions : (N, 2) array — actuator locations.
    directions : (N,) array — angles.
    amplitudes : (N,) array — magnitudes.
    width : float or (N,) array — Gaussian width(s).

    Returns
    -------
    (Fx, Fy) : tuple of face-centred force arrays.
    """
    def single(pos, dirn, amp):
        return jet_actuator_force_mac(grid, pos, dirn, amp, width)

    # vmap over actuators
    forces = jax.vmap(single)(positions, directions, amplitudes)
    # forces[0] has shape (N, Nx+1, Ny), forces[1] has shape (N, Nx, Ny+1)
    Fx_total = jnp.sum(forces[0], axis=0)
    Fy_total = jnp.sum(forces[1], axis=0)
    return (Fx_total, Fy_total)


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


# ── NACA airfoil surface parameterization ────────────────────────────

def naca_surface_params(s_params, thickness, camber, camber_pos, alpha,
                        chord, center, n_pts, offsets=None,
                        amplitude_scale=None):
    """Convert arc-length parameters to actuator positions on NACA airfoil.

    Parameters
    ----------
    s_params : (N,) array — fractional arc-length positions in [0, 1).
        0 = trailing edge (upper), 0.5 = leading edge, 1.0 = trailing edge (lower).
    thickness, camber, camber_pos, alpha, chord, center, n_pts :
        NACA geometry parameters (same as ``naca_coords_diff``).
    offsets : (N,) array, optional — angular offset from outward normal.
    amplitude_scale : (N,) array, optional — raw amplitude (softplus applied).

    Returns
    -------
    positions : (N, 2) array — actuator locations on surface.
    directions : (N,) array — jet direction angles.
    amplitudes : (N,) array.
    """
    from jax_cfd.core.geometry import naca_coords_diff

    coords, ds_arr = naca_coords_diff(
        thickness, camber, camber_pos, alpha, chord, center, n_pts
    )
    xs, ys = coords
    n_actual = xs.shape[0]

    # Cumulative arc-length (normalized to [0, 1])
    cum_ds = jnp.cumsum(ds_arr)
    total_len = cum_ds[-1]
    cum_s = cum_ds / total_len  # (n_actual,) in [0, 1]

    def interp_position(s):
        """Interpolate position and normal at fractional arc-length s."""
        s_mod = s % 1.0
        # Find index via linear interpolation
        idx_f = jnp.interp(s_mod, cum_s, jnp.arange(n_actual, dtype=jnp.float32))
        idx_lo = jnp.floor(idx_f).astype(jnp.int32)
        idx_hi = (idx_lo + 1) % n_actual
        frac = idx_f - jnp.floor(idx_f)

        x_pos = xs[idx_lo] * (1 - frac) + xs[idx_hi] * frac
        y_pos = ys[idx_lo] * (1 - frac) + ys[idx_hi] * frac

        # Tangent vector (forward difference)
        tx = xs[idx_hi] - xs[idx_lo]
        ty = ys[idx_hi] - ys[idx_lo]

        # Outward normal (rotate tangent by -90 deg for CCW-ordered airfoil)
        nx = ty
        ny = -tx
        normal_angle = jnp.arctan2(ny, nx)

        return jnp.array([x_pos, y_pos]), normal_angle

    positions_list = []
    normals_list = []
    for i in range(s_params.shape[0]):
        pos, norm = interp_position(s_params[i])
        positions_list.append(pos)
        normals_list.append(norm)
    positions = jnp.stack(positions_list)
    normals = jnp.stack(normals_list)

    if offsets is None:
        offsets = jnp.zeros_like(s_params)
    directions = normals + offsets

    if amplitude_scale is not None:
        amplitudes = jax.nn.softplus(amplitude_scale)
    else:
        amplitudes = jnp.ones_like(s_params)

    return positions, directions, amplitudes


# ── Momentum source factory ──────────────────────────────────────────

def make_actuator_source_fn(params, width=0.1, surface='free',
                             body_center=None, body_radius=None):
    """Create a ``momentum_source_fn`` compatible with ``create_step_fn``.

    Parameters
    ----------
    params : dict
        For ``surface='free'``:
            'positions' : (N, 2), 'directions' : (N,), 'amplitudes' : (N,)
        For ``surface='cylinder'``:
            'thetas' : (N,), 'offsets' : (N,) optional,
            'amplitude_raw' : (N,) optional
    width : float — Gaussian width.
    surface : 'free' or 'cylinder' — parameterization mode.
    body_center : (2,) array — required if surface='cylinder'.
    body_radius : scalar — required if surface='cylinder'.

    Returns
    -------
    source_fn : callable(scalars, velocity, grid) -> (Fx, Fy)
    """
    def source_fn(scalars, velocity, grid):
        if surface == 'cylinder':
            thetas = params['thetas']
            offsets = params.get('offsets', None)
            amp_raw = params.get('amplitude_raw', None)
            positions, directions, amplitudes = surface_constrained_params(
                thetas, body_center, body_radius, offsets, amp_raw
            )
        else:
            positions = params['positions']
            directions = params['directions']
            amplitudes = params['amplitudes']

        return multi_actuator_force_mac(grid, positions, directions,
                                         amplitudes, width)

    return source_fn


# ── Loss functions ────────────────────────────────────────────────────

def drag_loss(state, grid, axis=0):
    """Compute a proxy for drag from momentum flux.

    For flows with IBM bodies, use the IBM force diagnostics instead.
    This is a simple proxy for channel/body-force flows.
    """
    # Mean streamwise velocity deficit as drag proxy
    u = state.velocity[axis]
    return -jnp.mean(u)


def time_averaged_drag(states_or_forces, key='Cd'):
    """Compute time-averaged drag coefficient from a sequence.

    Parameters
    ----------
    states_or_forces : list of dicts with 'Cd' key, or array of Cd values.

    Returns
    -------
    J : scalar — mean Cd.
    """
    if isinstance(states_or_forces, (list, tuple)):
        vals = jnp.array([s[key] for s in states_or_forces])
    else:
        vals = states_or_forces
    return jnp.mean(vals)


def discounted_drag(cd_series, tau=5.0, dt=1.0):
    """Exponentially discounted time-averaged drag.

    J = sum_t exp(-t/tau) * Cd(t) / sum_t exp(-t/tau)

    Reduces sensitivity to chaotic long-horizon dynamics.
    """
    n = cd_series.shape[0]
    t = jnp.arange(n) * dt
    weights = jnp.exp(-t / tau)
    return jnp.sum(weights * cd_series) / jnp.sum(weights)
