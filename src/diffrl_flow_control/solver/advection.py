import jax.numpy as jnp
from typing import Tuple, Union
from diffrl_flow_control.solver.grid import MACGrid

# ======================================================================
# Helper primitives
# ======================================================================

def _cell_dx_broadcast(grid: MACGrid, axis: int) -> Union[float, jnp.ndarray]:
    """Cell spacing along *axis*, broadcast-ready for cell-shape arrays (length N)."""
    dx = grid.spacings_array(axis)
    if isinstance(dx, (int, float)):
        return dx
    shape = [1] * grid.ndim
    shape[axis] = grid.shape[axis]
    return jnp.reshape(dx, shape)


def _face_dx_broadcast(grid: MACGrid, axis: int) -> Union[float, jnp.ndarray]:
    """
    Face-to-face spacing along *axis*, broadcast-ready for face-shape arrays (length N+1).

    Used for flux divergence in the longitudinal advection case (i==j),
    where the result has N+1 elements.

    Interior: d[k] = (dx[k-1] + dx[k]) / 2  (distance between adjacent cell centres).
    Boundary: d[0] = dx[0], d[N] = dx[-1]  (exact value irrelevant — overwritten by BC).
    """
    dx = grid.spacings_array(axis)
    if isinstance(dx, (int, float)):
        return dx
    N = grid.shape[axis]
    d_face = jnp.empty(N + 1)
    d_face = d_face.at[1:N].set(0.5 * (dx[:-1] + dx[1:]))
    d_face = d_face.at[0].set(dx[0])
    d_face = d_face.at[N].set(dx[-1])
    shape = [1] * grid.ndim
    shape[axis] = N + 1
    return jnp.reshape(d_face, shape)

def interpolate_to_face(field: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Averages adjacent values along `axis` (1st-order midpoint interpolation)."""
    slices_plus = [slice(None)] * field.ndim
    slices_minus = [slice(None)] * field.ndim
    slices_plus[axis] = slice(1, None)
    slices_minus[axis] = slice(None, -1)
    return 0.5 * (field[tuple(slices_plus)] + field[tuple(slices_minus)])

def diff_to_face(field: jnp.ndarray, axis: int, dx: float) -> jnp.ndarray:
    """Computes finite difference across `axis`."""
    slices_plus = [slice(None)] * field.ndim
    slices_minus = [slice(None)] * field.ndim
    slices_plus[axis] = slice(1, None)
    slices_minus[axis] = slice(None, -1)
    return (field[tuple(slices_plus)] - field[tuple(slices_minus)]) / dx

# ======================================================================
# Van Leer flux limiter kernel
# ======================================================================

def _van_leer(r: jnp.ndarray) -> jnp.ndarray:
    """
    Van Leer (1974) smooth flux limiter.

    φ(r) = (r + |r|) / (1 + |r|)

    Properties:
      - φ(r) = 0 for r < 0   (local extremum: revert to 1st-order upwind)
      - φ(1) = 1              (smooth region: recovers 2nd-order CDS)
      - Symmetric: r·φ(1/r)  = φ(r)
    """
    abs_r = jnp.abs(r)
    return (r + abs_r) / (1.0 + abs_r)

def _muscl_face_value(u_mm: jnp.ndarray, u_m: jnp.ndarray,
                      u_p: jnp.ndarray, u_pp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    MUSCL reconstruction of the left and right states at the i+1/2 face.

    Uses Van Leer limiter to selectively blend upwind and central reconstructions.

    Arguments ordered as: u_{i-1}, u_i, u_{i+1}, u_{i+2}

    Returns:
      u_L: reconstructed state approaching from the left  (u_i side)
      u_R: reconstructed state approaching from the right (u_{i+1} side)
    """
    # Forward differences (upstream of face for each side)
    delta_L = u_p - u_m    # u_{i+1} - u_i
    delta_LL = u_m - u_mm  # u_i     - u_{i-1}
    delta_R  = u_pp - u_p  # u_{i+2} - u_{i+1}

    # Smoothness ratios — safe denominator pattern to avoid NaN in XLA.
    eps = 1e-10
    safe_delta_L = jnp.where(jnp.abs(delta_L) > eps, delta_L, 1.0)
    safe_delta_R = jnp.where(jnp.abs(delta_R) > eps, delta_R, 1.0)
    r_L = jnp.where(jnp.abs(delta_L) > eps, delta_LL / safe_delta_L, 0.0)
    r_R = jnp.where(jnp.abs(delta_R) > eps, delta_L  / safe_delta_R, 0.0)

    phi_L = _van_leer(r_L)
    phi_R = _van_leer(r_R)

    # MUSCL 2nd-order reconstruction
    u_L = u_m + 0.5 * phi_L * delta_L
    u_R = u_p - 0.5 * phi_R * delta_R

    return u_L, u_R

def _godunov_upwind_flux(u_L: jnp.ndarray, u_R: jnp.ndarray,
                         c_L: jnp.ndarray, c_R: jnp.ndarray) -> jnp.ndarray:
    """
    Godunov-type numerical flux for a linear advection term.

    For the convective flux F = c * u at the cell face:
      - If the advecting velocity c > 0, fluid comes from the LEFT (u_L).
      - If c < 0, fluid comes from the RIGHT (u_R).

    c_L, c_R: left/right reconstructed advecting velocities.
    u_L, u_R: left/right reconstructed transported quantities.
    """
    # Use arithmetic mean of advecting velocity at the face
    c_face = 0.5 * (c_L + c_R)
    return jnp.where(c_face > 0.0, c_L * u_L, c_R * u_R)

def _muscl_flux_1d(
    field: jnp.ndarray,
    conv: jnp.ndarray,
    axis: int,
    periodic_axes: Tuple[int, ...] = (),
) -> jnp.ndarray:
    """
    Computes MUSCL Van Leer limited numerical flux F_{i+1/2} = c_{i+1/2} * phi(u)
    along a single `axis`. `field` and `conv` must have the same shape.

    Pads by 2 ghost layers on each side internally (required for the 4-point stencil).
    Uses ``mode='wrap'`` for axes in *periodic_axes*, ``mode='edge'`` otherwise.

    Returns:
        flux array of shape field.shape with field.shape[axis] + 1 along `axis`.
        (i.e. one flux value per inter-cell face, including the two boundary faces)
        Call diff_to_face(result, axis, dx) to obtain the flux divergence of shape
        field.shape.
    """
    ndim = field.ndim
    pad_w = [(0, 0)] * ndim
    pad_w[axis] = (2, 2)
    pad_mode = 'wrap' if axis in periodic_axes else 'edge'

    f_pad = jnp.pad(field, pad_w, mode=pad_mode)
    c_pad = jnp.pad(conv,  pad_w, mode=pad_mode)

    # Extract 4-point stencil shifted by index for each face
    def sl(start, end=None):
        s = [slice(None)] * ndim
        s[axis] = slice(start, end)
        return tuple(s)

    # Stencil: mm=i-1, m=i, p=i+1, pp=i+2  (face between m and p)
    f_mm = f_pad[sl(0, -3)]
    f_m  = f_pad[sl(1, -2)]
    f_p  = f_pad[sl(2, -1)]
    f_pp = f_pad[sl(3, None)]

    c_mm = c_pad[sl(0, -3)]
    c_m  = c_pad[sl(1, -2)]
    c_p  = c_pad[sl(2, -1)]
    c_pp = c_pad[sl(3, None)]

    # MUSCL reconstruction of transported value at face
    f_L, f_R = _muscl_face_value(f_mm, f_m, f_p, f_pp)
    # MUSCL reconstruction of advecting velocity at face
    c_L, c_R = _muscl_face_value(c_mm, c_m, c_p, c_pp)

    # Numerical flux at i+1/2 faces, shape = field.shape + 1 along axis
    flux = _godunov_upwind_flux(f_L, f_R, c_L, c_R)

    return flux

# ======================================================================
# Central Scheme (original, kept for reference / low-Re simulations)
# ======================================================================


# ======================================================================
# MUSCL Van Leer Scheme (new, high-Pe stable)
# ======================================================================

def advect_momentum_muscl(
    velocity: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
    periodic_axes: Tuple[int, ...] = (),
) -> Tuple[jnp.ndarray, ...]:
    """
    MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) momentum
    advection with Van Leer flux limiter.

    Reference: van Leer (1979), "Towards the ultimate conservative difference scheme V",
               J. Computational Physics 32(1):101-136.

    Properties:
      - 2nd-order accurate in smooth regions (like central scheme)
      - Reverts to 1st-order upwind at extrema / discontinuities (no oscillations)
      - Total Variation Diminishing (TVD) property guaranteed by Van Leer limiter
      - Stable for high Peclet numbers (Pe >> 2) unlike pure central scheme

    periodic_axes : axes with periodic BC (use ``mode='wrap'`` padding).

    The divergence form is used:
      ∂u_i/∂t = ... - ∑_j ∂(u_j * u_i) / ∂x_j

    Shape invariant: output[i] has the same shape as velocity[i].
    """
    advection = []

    for i, u_i in enumerate(velocity):
        adv_i = jnp.zeros_like(u_i)

        for j, u_j in enumerate(velocity):

            if i == j:
                # ------------------------------------------------------------------
                # Longitudinal: ∂(u_i²)/∂x_i
                # Result has N+1 elements along axis i → use face-to-face spacing.
                # ------------------------------------------------------------------
                dx_j = _face_dx_broadcast(grid, j)
                flux = _muscl_flux_1d(u_i, u_i, i, periodic_axes)
                adv_i = adv_i + diff_to_face(flux, i, dx_j)

            else:
                # ------------------------------------------------------------------
                # Transverse: ∂(u_j * u_i)/∂x_j
                # Result has N_j elements along axis j → use cell spacing.
                # ------------------------------------------------------------------
                dx_j = _cell_dx_broadcast(grid, j)

                # Step 1: Interpolate u_j to corners (pad 1 ghost along i, then average).
                pad_corner = [(0, 0)] * grid.ndim
                pad_corner[i] = (1, 1)
                pad_mode_i = 'wrap' if i in periodic_axes else 'edge'
                u_j_corner = interpolate_to_face(
                    jnp.pad(u_j, pad_corner, mode=pad_mode_i), i
                )

                # Step 2: MUSCL-reconstruct u_i at j-face positions.
                pad_muscl = [(0, 0)] * grid.ndim
                pad_muscl[j] = (2, 2)
                pad_mode_j = 'wrap' if j in periodic_axes else 'edge'
                u_i_pad = jnp.pad(u_i, pad_muscl, mode=pad_mode_j)

                # 4-point stencil along j
                def _sl(start, end=None, _j=j, _nd=grid.ndim):
                    s = [slice(None)] * _nd
                    s[_j] = slice(start, end)
                    return tuple(s)

                f_mm = u_i_pad[_sl(0, -3)]
                f_m  = u_i_pad[_sl(1, -2)]
                f_p  = u_i_pad[_sl(2, -1)]
                f_pp = u_i_pad[_sl(3, None)]

                u_L, u_R = _muscl_face_value(f_mm, f_m, f_p, f_pp)
                flux = jnp.where(u_j_corner > 0.0, u_j_corner * u_L, u_j_corner * u_R)
                adv_i = adv_i + diff_to_face(flux, j, dx_j)

        advection.append(adv_i)

    return tuple(advection)

# ======================================================================
# Scalar Advection (conservative, central)
# ======================================================================

def advect_scalar(
    scalar: jnp.ndarray,
    velocity: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
    periodic_axes: Tuple[int, ...] = (),
) -> jnp.ndarray:
    """
    Conservative central-scheme staggered advection: div(u * T) at cell centers.
    For scalar fields (temperature, concentration) on a MAC grid.

    periodic_axes : axes with periodic BC (use ``mode='wrap'`` padding).
    """
    div_flux = jnp.zeros_like(scalar)

    for i, u_i in enumerate(velocity):
        dx_i = _cell_dx_broadcast(grid, i)

        pad_width = [(0, 0)] * grid.ndim
        pad_width[i] = (1, 1)
        pad_mode = 'wrap' if i in periodic_axes else 'edge'
        s_pad = jnp.pad(scalar, pad_width, mode=pad_mode)
        s_face = interpolate_to_face(s_pad, i)
        flux = u_i * s_face
        div_flux += diff_to_face(flux, i, dx_i)

    return div_flux

# ======================================================================
# WENO5 (Weighted Essentially Non-Oscillatory, 5th order)
# ======================================================================











def advect_scalar_muscl(
    scalar: jnp.ndarray,
    velocity: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
    periodic_axes: Tuple[int, ...] = (),
) -> jnp.ndarray:
    """
    MUSCL Van Leer flux-limited scalar advection: div(u * T).
    Use for scalars with sharp gradients (e.g., temperature fronts, phase interfaces).

    scalar: cell-centered, shape (N_0, N_1, ...)
    velocity[i]: i-face-centered, shape (..., N_i+1, ...)
    periodic_axes : axes with periodic BC (use ``mode='wrap'`` padding).
    """
    div_flux = jnp.zeros_like(scalar)

    for i, u_i in enumerate(velocity):
        dx_i = _cell_dx_broadcast(grid, i)

        # Interpolate face-centered u_i to cell centers
        u_cc = interpolate_to_face(u_i, i)
        flux = _muscl_flux_1d(scalar, u_cc, i, periodic_axes)
        div_flux = div_flux + diff_to_face(flux, i, dx_i)

    return div_flux
