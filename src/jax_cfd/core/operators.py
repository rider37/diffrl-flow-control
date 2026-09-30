import jax.numpy as jnp
from typing import Tuple, Union
from jax_cfd.core.mac_grid import MACGrid


# ── Helpers for non-uniform spacing ──────────────────────────────────────

def _dx_broadcast(grid: MACGrid, axis: int) -> Union[float, jnp.ndarray]:
    """
    Cell-width spacing for *axis*, broadcast-ready for arrays of shape
    ``cell_shape()``.

    Returns a scalar for uniform axes, or a reshaped 1-D array for the
    stretched axis.
    """
    dx = grid.spacings_array(axis)
    if isinstance(dx, (int, float)):
        return dx
    shape = [1] * grid.ndim
    shape[axis] = grid.shape[axis]
    return jnp.reshape(dx, shape)


def _dx_face_broadcast(grid: MACGrid, axis: int) -> Union[float, jnp.ndarray]:
    """
    Distance between adjacent cell centres along *axis*, broadcast-ready
    for arrays of shape ``face_shape(axis)`` (length N+1).

    Used as the denominator in the MAC pressure gradient.

    For uniform axes returns a scalar.  For the stretched axis returns a
    reshaped 1-D array of length N+1.
    """
    dx = grid.spacings_array(axis)
    if isinstance(dx, (int, float)):
        return dx
    # Interior face i (between cell i-1 and cell i):
    #   d_face[i] = 0.5*(dx[i-1] + dx[i])  for i=1,...,N-1
    # Boundary faces (with Neumann / edge padding, gradient = 0):
    #   d_face[0] = dx[0],  d_face[N] = dx[-1]  (exact value doesn't matter)
    N = grid.shape[axis]
    d_face = jnp.empty(N + 1)
    d_face = d_face.at[1:N].set(0.5 * (dx[:-1] + dx[1:]))
    d_face = d_face.at[0].set(dx[0])
    d_face = d_face.at[N].set(dx[-1])
    shape = [1] * grid.ndim
    shape[axis] = N + 1
    return jnp.reshape(d_face, shape)


def _central_denom_broadcast(grid: MACGrid, axis: int) -> Union[float, jnp.ndarray]:
    """
    Denominator for central differences ``(f[i+1] - f[i-1]) / denom``
    at cell centres (after edge-padding by 1 on each side).

    Uniform → ``2*dx`` (scalar).
    Stretched → 1-D array of length N, reshaped for broadcasting.
    """
    dx = grid.spacings_array(axis)
    if isinstance(dx, (int, float)):
        return 2.0 * dx
    xc = grid.cell_centers_1d(axis)
    xc_pad = jnp.pad(xc, (1, 1), mode='edge')
    denom = xc_pad[2:] - xc_pad[:-2]  # (N,)
    shape = [1] * grid.ndim
    shape[axis] = grid.shape[axis]
    return jnp.reshape(denom, shape)


# ── Core operators ────────────────────────────────────────────────────────

def divergence(velocity: Tuple[jnp.ndarray, ...], grid: MACGrid) -> jnp.ndarray:
    """
    True N-Dimensional MAC Divergence.
    Velocity component `i` exists on the faces of dimension `i`,
    so shape is N_i + 1 down axis `i`.
    Divergence naturally maps these face fluxes onto cell centers (N_i).

    Supports non-uniform spacing via ``grid.spacings_array()``.
    """
    div = jnp.zeros(grid.cell_shape())
    for i in range(grid.ndim):
        vel_cmp = velocity[i]
        dx_bc = _dx_broadcast(grid, i)

        slices_plus = [slice(None)] * grid.ndim
        slices_minus = [slice(None)] * grid.ndim
        slices_plus[i] = slice(1, None)
        slices_minus[i] = slice(None, -1)

        div = div + (vel_cmp[tuple(slices_plus)] - vel_cmp[tuple(slices_minus)]) / dx_bc

    return div

def gradient_p(
    p: jnp.ndarray,
    grid: MACGrid,
    periodic_axes: Tuple[int, ...] = (),
) -> Tuple[jnp.ndarray, ...]:
    """
    True N-Dimensional MAC Pressure Gradient.
    Pressure exists at cell centers. The gradient is evaluated perfectly at the faces.

    Supports non-uniform spacing.  The denominator at each face is the
    distance between the two adjacent cell centres.
    """
    grad = []
    for i in range(grid.ndim):
        pad_width = [(0, 0)] * grid.ndim
        pad_width[i] = (1, 1)
        pad_mode = 'wrap' if i in periodic_axes else 'edge'
        p_pad = jnp.pad(p, pad_width, mode=pad_mode)

        slices_plus = [slice(None)] * grid.ndim
        slices_minus = [slice(None)] * grid.ndim
        slices_plus[i] = slice(1, None)
        slices_minus[i] = slice(None, -1)

        dx_face = _dx_face_broadcast(grid, i)
        grad_cmp = (p_pad[tuple(slices_plus)] - p_pad[tuple(slices_minus)]) / dx_face
        grad.append(grad_cmp)

    return tuple(grad)

def diff_axis_padded(
    field: jnp.ndarray,
    axis: int,
    dx: float,
) -> jnp.ndarray:
    """
    Computes rigorous 2nd derivative on a **uniform** grid.
    Assumes `field` HAS ALREADY BEEN PADDED with 1 ghost layer along `axis`.
    This decouples the math from the boundary application!
    """
    slices_p = [slice(None)] * field.ndim
    slices_c = [slice(None)] * field.ndim
    slices_m = [slice(None)] * field.ndim

    slices_p[axis] = slice(2, None)
    slices_c[axis] = slice(1, -1)
    slices_m[axis] = slice(None, -2)

    return (field[tuple(slices_p)] - 2.0 * field[tuple(slices_c)] + field[tuple(slices_m)]) / (dx**2)


def diff_axis_padded_nu(
    field: jnp.ndarray,
    axis: int,
    grid: MACGrid,
) -> jnp.ndarray:
    """
    Second derivative (Laplacian component) on a **non-uniform** grid.

    Assumes `field` HAS ALREADY BEEN PADDED with 1 ghost layer along `axis`.

    Conservative discretization::

        d²f/dx² at point i ≈ [(f[i+1]-f[i])/d⁺ - (f[i]-f[i-1])/d⁻] / h[i]

    Handles both cell-centered fields (N interior points) and face-centered
    fields (N+1 interior points along the face axis) by inferring the
    interior size from the padded field shape.

    Falls back to the uniform formula when the axis is not stretched.
    """
    dx_arr = grid.spacings_array(axis)
    if isinstance(dx_arr, (int, float)):
        return diff_axis_padded(field, axis, dx_arr)

    ndim = field.ndim
    N_cell = grid.shape[axis]
    N_int = field.shape[axis] - 2  # interior points after removing 2 ghost layers

    slices_p = [slice(None)] * ndim
    slices_c = [slice(None)] * ndim
    slices_m = [slice(None)] * ndim
    slices_p[axis] = slice(2, None)
    slices_c[axis] = slice(1, -1)
    slices_m[axis] = slice(None, -2)

    f_p = field[tuple(slices_p)]
    f_c = field[tuple(slices_c)]
    f_m = field[tuple(slices_m)]

    if N_int == N_cell:
        # ── Cell-centred field: N interior points ──
        # d_plus[i]  = distance from cell centre i to centre i+1
        # d_minus[i] = distance from centre i-1 to centre i
        # h[i]       = cell width dx_arr[i]
        d_plus = jnp.empty(N_cell)
        d_plus = d_plus.at[:N_cell-1].set(0.5 * (dx_arr[:N_cell-1] + dx_arr[1:N_cell]))
        d_plus = d_plus.at[N_cell-1].set(dx_arr[N_cell-1])  # to ghost

        d_minus = jnp.empty(N_cell)
        d_minus = d_minus.at[0].set(dx_arr[0])  # from ghost
        d_minus = d_minus.at[1:N_cell].set(0.5 * (dx_arr[:N_cell-1] + dx_arr[1:N_cell]))

        h = dx_arr
    else:
        # ── Face-centred field: N_cell+1 interior points ──
        # Points sit at face positions y_0, y_1, ..., y_{N_cell}.
        # d_plus[k]  = y_{k+1} - y_k = cell spacing dy[k]  (k < N_cell)
        # d_minus[k] = y_k - y_{k-1} = cell spacing dy[k-1] (k > 0)
        # Boundary ghosts mirror the nearest cell spacing.
        d_plus = jnp.concatenate([dx_arr, dx_arr[-1:]])    # (N_cell+1,)
        d_minus = jnp.concatenate([dx_arr[:1], dx_arr])     # (N_cell+1,)
        h = 0.5 * (d_plus + d_minus)

    # Reshape for broadcasting
    shape_bc = [1] * ndim
    shape_bc[axis] = N_int
    d_plus  = jnp.reshape(d_plus,  shape_bc)
    d_minus = jnp.reshape(d_minus, shape_bc)
    h       = jnp.reshape(h,       shape_bc)

    return ((f_p - f_c) / d_plus - (f_c - f_m) / d_minus) / h


def interp_to_face(field: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    Interpolates a cell-centered field onto the faces of `axis`.
    """
    slices_p = [slice(None)] * field.ndim
    slices_m = [slice(None)] * field.ndim

    slices_p[axis] = slice(1, None)
    slices_m[axis] = slice(None, -1)

    return 0.5 * (field[tuple(slices_p)] + field[tuple(slices_m)])


def velocity_gradient_tensor(
    velocity: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
) -> jnp.ndarray:
    """
    Full velocity gradient tensor g_{ij} = du_i/dx_j at cell centers.

    Returns shape (ndim, ndim, *cell_shape).  Reused by turbulence models
    (Smagorinsky, WALE) and vorticity computation to avoid code duplication.

    Supports non-uniform spacing via ``grid.spacings_array()``.
    """
    ndim = grid.ndim
    # Face -> cell center averages
    u_cc = []
    for i, u_cmp in enumerate(velocity):
        sl_p = [slice(None)] * ndim
        sl_m = [slice(None)] * ndim
        sl_p[i] = slice(1, None)
        sl_m[i] = slice(None, -1)
        u_cc.append(0.5 * (u_cmp[tuple(sl_p)] + u_cmp[tuple(sl_m)]))

    g_list = []
    for i in range(ndim):
        grad_ui = []
        for j in range(ndim):
            pw = [(0, 0)] * ndim
            pw[j] = (1, 1)
            u_pad = jnp.pad(u_cc[i], pw, mode='edge')
            sl_p = [slice(None)] * ndim
            sl_m = [slice(None)] * ndim
            sl_p[j] = slice(2, None)
            sl_m[j] = slice(None, -2)
            denom = _central_denom_broadcast(grid, j)
            grad_ui.append(
                (u_pad[tuple(sl_p)] - u_pad[tuple(sl_m)]) / denom
            )
        g_list.append(jnp.stack(grad_ui, axis=0))
    return jnp.stack(g_list, axis=0)


def vorticity_2d(velocity: Tuple[jnp.ndarray, ...], grid: MACGrid) -> jnp.ndarray:
    """
    z-component of vorticity at cell centres for 2-D flows:
        ω_z = ∂v/∂x − ∂u/∂y   →  shape (N0, N1)

    Face velocities are averaged to cell centres first; central differences
    with edge padding give full-domain coverage without boundary artefacts.
    """
    u, v = velocity[0], velocity[1]
    dx, dy = grid.spacings[0], grid.spacings[1]

    u_cc = 0.5 * (u[:-1, :] + u[1:, :])   # (N0, N1)
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])   # (N0, N1)

    v_pad = jnp.pad(v_cc, ((1, 1), (0, 0)), mode='edge')
    u_pad = jnp.pad(u_cc, ((0, 0), (1, 1)), mode='edge')

    dv_dx = (v_pad[2:, :] - v_pad[:-2, :]) / (2.0 * dx)
    du_dy = (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * dy)

    return dv_dx - du_dy


def vorticity(velocity: Tuple[jnp.ndarray, ...], grid: MACGrid):
    """
    Vorticity at cell centres (N-dimensional).

    2D: returns scalar ω_z = ∂v/∂x − ∂u/∂y   shape (Nx, Ny)
    3D: returns tuple  (ω_x, ω_y, ω_z)        each  (Nx, Ny, Nz)

    Supports non-uniform spacing.
    """
    ndim = grid.ndim

    # Interpolate face velocities to cell centres
    u_cc = []
    for i, u_cmp in enumerate(velocity):
        sl_p = [slice(None)] * ndim
        sl_m = [slice(None)] * ndim
        sl_p[i] = slice(1, None)
        sl_m[i] = slice(None, -1)
        u_cc.append(0.5 * (u_cmp[tuple(sl_p)] + u_cmp[tuple(sl_m)]))

    def _grad(i, j):
        """∂u_i/∂x_j at cell centres via central difference with edge padding."""
        pw = [(0, 0)] * ndim
        pw[j] = (1, 1)
        u_pad = jnp.pad(u_cc[i], pw, mode='edge')
        sl_p = [slice(None)] * ndim
        sl_m = [slice(None)] * ndim
        sl_p[j] = slice(2, None)
        sl_m[j] = slice(None, -2)
        denom = _central_denom_broadcast(grid, j)
        return (u_pad[tuple(sl_p)] - u_pad[tuple(sl_m)]) / denom

    if ndim == 2:
        return _grad(1, 0) - _grad(0, 1)  # dv/dx - du/dy
    elif ndim == 3:
        omega_x = _grad(2, 1) - _grad(1, 2)  # dw/dy - dv/dz
        omega_y = _grad(0, 2) - _grad(2, 0)  # du/dz - dw/dx
        omega_z = _grad(1, 0) - _grad(0, 1)  # dv/dx - du/dy
        return (omega_x, omega_y, omega_z)
    else:
        raise NotImplementedError(f"Vorticity not implemented for {ndim}D")


# ── Shared higher-level operators ────────────────────────────────────────

def laplacian_cell_centered(
    field: jnp.ndarray,
    grid: MACGrid,
) -> jnp.ndarray:
    """Cell-centred Laplacian with zero-gradient (Neumann) BC.

    Pads each axis independently with ``mode='edge'`` and applies
    ``diff_axis_padded`` (uniform) or ``diff_axis_padded_nu`` (non-uniform).

    Reusable by RANS, combustion, and any module that needs ∇²f at cells.
    """
    lap = jnp.zeros_like(field)
    for ax in range(grid.ndim):
        pad_w = [(0, 0)] * grid.ndim
        pad_w[ax] = (1, 1)
        padded = jnp.pad(field, pad_w, mode='edge')
        lap = lap + diff_axis_padded_nu(padded, ax, grid)
    return lap


def strain_rate_magnitude(
    velocity: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
) -> jnp.ndarray:
    """Compute |S| = √(2 S_ij S_ij) at cell centres.

    S is the symmetric part of the velocity gradient tensor.
    Reusable by RANS (production term) and rheology (non-Newtonian viscosity).
    """
    g = velocity_gradient_tensor(velocity, grid)
    ndim = grid.ndim
    axes = tuple(range(2, 2 + ndim))
    S = 0.5 * (g + jnp.transpose(g, (1, 0) + axes))
    S_sq = jnp.sum(S * S, axis=(0, 1))
    return jnp.sqrt(2.0 * S_sq + 1e-20)
