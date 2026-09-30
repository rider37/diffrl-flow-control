"""Ghost-cell (sharp-interface) Immersed Boundary Method for JAX-CFD.

Replaces the Peskin 4-point delta function with a signed-distance-field (SDF)
based approach.  The body boundary transitions over ~1 cell width (vs ~4dx
for Peskin), dramatically reducing the effective body thickening and giving
more accurate Cd/Cl at moderate grid resolutions.

All operations are differentiable (smooth SDF sign via tanh, sigmoid mask,
map_coordinates interpolation) for use with ``jax.grad`` optimization.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


# ── SDF Primitives ───────────────────────────────────────────────────────

def _point_to_segments_sqdist(qx, qy, seg_x0, seg_y0, seg_x1, seg_y1):
    """Vectorized squared distance from query points to line segments.

    Parameters
    ----------
    qx, qy : (N,) query point coordinates (flattened grid)
    seg_x0, seg_y0, seg_x1, seg_y1 : (S,) segment endpoint coordinates

    Returns
    -------
    sq_dist : (N, S) squared distance from each query to each segment
    """
    dx_s = seg_x1 - seg_x0          # (S,)
    dy_s = seg_y1 - seg_y0
    len_sq = dx_s ** 2 + dy_s ** 2   # (S,)
    len_sq = jnp.maximum(len_sq, 1e-30)

    # Vectors from segment start to query points: (N, S)
    px = qx[:, None] - seg_x0[None, :]
    py = qy[:, None] - seg_y0[None, :]

    # Clamped projection parameter
    t = (px * dx_s[None, :] + py * dy_s[None, :]) / len_sq[None, :]
    t = jnp.clip(t, 0.0, 1.0)

    # Nearest point on segment
    near_x = seg_x0[None, :] + t * dx_s[None, :]
    near_y = seg_y0[None, :] + t * dy_s[None, :]

    return (qx[:, None] - near_x) ** 2 + (qy[:, None] - near_y) ** 2


def _winding_number(qx, qy, poly_x, poly_y):
    """Winding number of query points w.r.t. a closed polygon via atan2.

    Returns ~+1 inside a CCW polygon, ~0 outside.  Robust for thin
    geometries (no nearest-marker sign ambiguity).
    """
    px_next = jnp.roll(poly_x, -1)
    py_next = jnp.roll(poly_y, -1)

    # Vectors from query to current and next vertex: (N, V)
    ax = poly_x[None, :] - qx[:, None]
    ay = poly_y[None, :] - qy[:, None]
    bx = px_next[None, :] - qx[:, None]
    by = py_next[None, :] - qy[:, None]

    cross = ax * by - ay * bx
    dot   = ax * bx + ay * by
    angles = jnp.arctan2(cross, dot)

    return jnp.sum(angles, axis=1) / (2.0 * jnp.pi)


# ── SDF on a 2D grid ────────────────────────────────────────────────────

def compute_sdf_2d(x_1d, y_1d, poly_x, poly_y):
    """Signed distance field on a 2D grid from a closed polygon.

    Parameters
    ----------
    x_1d : (Nx,) grid x-coordinates
    y_1d : (Ny,) grid y-coordinates
    poly_x, poly_y : (V,) polygon vertex coordinates

    Returns
    -------
    sdf : (Nx, Ny) — negative inside, positive outside
    """
    Nx = x_1d.shape[0]
    Ny = y_1d.shape[0]

    xx, yy = jnp.meshgrid(x_1d, y_1d, indexing='ij')
    qx = xx.ravel()
    qy = yy.ravel()

    # Segment endpoints (consecutive vertex pairs, last wraps to first)
    seg_x1 = jnp.roll(poly_x, -1)
    seg_y1 = jnp.roll(poly_y, -1)

    # Minimum unsigned distance
    sq_dist = _point_to_segments_sqdist(qx, qy, poly_x, poly_y, seg_x1, seg_y1)
    unsigned = jnp.sqrt(jnp.min(sq_dist, axis=1) + 1e-14)

    # Smooth sign from winding number: −1 inside, +1 outside
    winding = _winding_number(qx, qy, poly_x, poly_y)
    sign = jnp.tanh(10.0 * (0.5 - jnp.abs(winding)))

    return (unsigned * sign).reshape(Nx, Ny)


# ── Mask & Gradient ──────────────────────────────────────────────────────

def sigmoid_mask(sdf, eps):
    """Smooth body mask: ~1 inside (sdf<0), ~0 outside.  Width ≈ 2·eps."""
    return jax.nn.sigmoid(-sdf / eps)


def _sdf_gradient_normalised(sdf, dx, dy):
    """Outward unit normal from SDF via central differences."""
    gx = (jnp.roll(sdf, -1, axis=0) - jnp.roll(sdf, 1, axis=0)) / (2.0 * dx)
    gy = (jnp.roll(sdf, -1, axis=1) - jnp.roll(sdf, 1, axis=1)) / (2.0 * dy)
    mag = jnp.sqrt(gx ** 2 + gy ** 2 + 1e-14)
    return gx / mag, gy / mag


# ── Image-Point Interpolation ───────────────────────────────────────────

def _ghost_velocity_component(u_field, sdf, n_x, n_y,
                               x_1d, y_1d, U_body_cmp, band):
    """Ghost-cell velocity for one velocity component.

    For cells inside the body (sdf < 0), reflects across the boundary to
    find an image point in the fluid, interpolates the velocity there, and
    sets the ghost value to enforce the Dirichlet no-slip condition.

    Deep-interior cells (sdf < −band) smoothly revert to the body velocity
    to avoid interpolating at distant (possibly outside-domain) image points.
    """
    Nx, Ny = u_field.shape
    xx, yy = jnp.meshgrid(x_1d, y_1d, indexing='ij')

    # Image point: reflected across boundary
    x_img = xx - 2.0 * sdf * n_x
    y_img = yy - 2.0 * sdf * n_y

    # Physical → fractional array indices (assumes uniform grid)
    x0 = x_1d[0]
    y0 = y_1d[0]
    dx_g = x_1d[1] - x_1d[0]
    dy_g = y_1d[1] - y_1d[0]

    ix = jnp.clip((x_img - x0) / dx_g, 0.0, Nx - 1.001)
    iy = jnp.clip((y_img - y0) / dy_g, 0.0, Ny - 1.001)

    # Bilinear interpolation at image points
    u_image = map_coordinates(u_field, [ix, iy], order=1, mode='nearest')

    # Dirichlet no-slip: u_ghost = 2·U_body − u_image
    u_ghost = 2.0 * U_body_cmp - u_image

    # Deep interior: smooth crossover to body velocity
    deep = jax.nn.sigmoid(-(sdf + band) / (0.25 * band + 1e-14))
    u_ghost = deep * U_body_cmp + (1.0 - deep) * u_ghost

    return u_ghost


# ── Factory ──────────────────────────────────────────────────────────────

def make_ghost_cell_ibm_fn(grid, coords_tuple, ds,
                            U_body=None, eps=None, band_width=None,
                            jet_surface_basis=None):
    """Create a ghost-cell IBM closure compatible with ``create_step_fn``.

    Pre-computes SDF, sigmoid mask, and normalised gradients on the
    u-face and v-face grids so they are evaluated once (not per RK3 stage).

    Parameters
    ----------
    grid : MACGrid
    coords_tuple : (xs, ys) closed-polygon body coordinates
    ds : (n_pts,) arc-length weights (unused; kept for API parity)
    U_body : (u, v) body velocity; default (0, 0)
    eps : sigmoid half-width; default 0.5·dx
    band_width : image-point band; default 3·dx
    jet_surface_basis : dict or None
        If provided, ``{'u': array, 'v': array}`` — pre-computed surface
        velocity fields per unit jet action.  When set, ``extra`` arg to
        ``ibm_fn`` is interpreted as the jet action scalar, and the surface
        velocity becomes ``U_body + action * basis``.

    Returns
    -------
    ibm_fn : callable
        ``ibm_fn(velocity, extra, grid, dt_sub) → (vel_corrected, forces)``
    """
    dx = grid.spacings[0]
    dy = grid.spacings[1]
    if eps is None:
        eps = 0.5 * dx
    if band_width is None:
        band_width = 3.0 * dx
    if U_body is None:
        U_body = (0.0, 0.0)

    poly_x, poly_y = coords_tuple

    # ── u-face grid (Nx+1, Ny) ───────────────────────────────────────
    x_u = grid.face_centers(0)[0]
    y_u = grid.face_centers(0)[1]
    sdf_u = compute_sdf_2d(x_u, y_u, poly_x, poly_y)
    mask_u = sigmoid_mask(sdf_u, eps)
    nu_x, nu_y = _sdf_gradient_normalised(sdf_u, dx, dy)

    # ── v-face grid (Nx, Ny+1) ───────────────────────────────────────
    x_v = grid.face_centers(1)[0]
    y_v = grid.face_centers(1)[1]
    sdf_v = compute_sdf_2d(x_v, y_v, poly_x, poly_y)
    mask_v = sigmoid_mask(sdf_v, eps)
    nv_x, nv_y = _sdf_gradient_normalised(sdf_v, dx, dy)

    _has_jets = jet_surface_basis is not None
    _jet_basis_u = jet_surface_basis['u'] if _has_jets else None
    _jet_basis_v = jet_surface_basis['v'] if _has_jets else None

    def ibm_fn(velocity, extra, grid_arg, dt_sub):
        u, v = velocity

        # Surface velocity: static U_body + dynamic jet contribution
        if _has_jets:
            action = extra  # extra carries the jet action scalar
            Ub_u = U_body[0] + action * _jet_basis_u
            Ub_v = U_body[1] + action * _jet_basis_v
        else:
            Ub_u, Ub_v = U_body[0], U_body[1]

        u_ghost = _ghost_velocity_component(
            u, sdf_u, nu_x, nu_y, x_u, y_u, Ub_u, band_width)
        v_ghost = _ghost_velocity_component(
            v, sdf_v, nv_x, nv_y, x_v, y_v, Ub_v, band_width)

        # Blend: fluid velocity outside, ghost velocity inside
        u_new = (1.0 - mask_u) * u + mask_u * u_ghost
        v_new = (1.0 - mask_v) * v + mask_v * v_ghost

        # Momentum-deficit reaction force on the body
        dV = dx * dy
        Fx = -jnp.sum(u_new - u) * dV / dt_sub
        Fy = -jnp.sum(v_new - v) * dV / dt_sub

        return (u_new, v_new), jnp.array([Fx, Fy])

    return ibm_fn
