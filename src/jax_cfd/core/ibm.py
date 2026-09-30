import jax.numpy as jnp
from typing import Tuple
from jax_cfd.core.mac_grid import MACGrid

def peskin_phi(r: jnp.ndarray) -> jnp.ndarray:
    """4-point piecewise Peskin kernel for Immersed Boundaries.

    Uses ``jnp.maximum(..., _EPS)`` with a tiny positive epsilon to keep
    ``sqrt`` arguments strictly positive.  This prevents NaN gradients
    when differentiating through the kernel via ``jax.grad``, because
    ``d/dx sqrt(0) = +inf`` and ``inf * 0 = NaN`` in IEEE 754.
    """
    _EPS = 1e-14
    abs_r = jnp.abs(r)

    # Clamp abs_r into each branch's valid range so that the sqrt
    # argument is always positive even when the branch is not selected
    # by the outer jnp.where.  This prevents NaN gradients leaking
    # through the unselected branch.
    ar1 = jnp.clip(abs_r, 0.0, 1.0)
    ar2 = jnp.clip(abs_r, 1.0, 2.0)

    val_1 = 0.125 * (3.0 - 2.0 * abs_r + jnp.sqrt(jnp.maximum(1.0 + 4.0 * ar1 - 4.0 * ar1**2, _EPS)))
    val_2 = 0.125 * (5.0 - 2.0 * abs_r - jnp.sqrt(jnp.maximum(-7.0 + 12.0 * ar2 - 4.0 * ar2**2, _EPS)))

    phi = jnp.where(abs_r <= 1.0, val_1, jnp.where(abs_r <= 2.0, val_2, 0.0))
    return phi

def compute_weights_1d(x_grid: jnp.ndarray, X_lag: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Computes exactly the 1D Dirac delta weights: D[k, i] = (1/dx) * phi((x_grid[i] - X_lag[k]) / dx)"""
    r = (x_grid[None, :] - X_lag[:, None]) / dx
    return peskin_phi(r) / dx

def compute_tensor_weights_mac(grid: MACGrid, coords_tuple: Tuple[jnp.ndarray, ...]):
    """
    Computes Separable Einsum Weights for the MAC Grid in 2D or 3D.
    Returns weight arrays perfectly aligned with the staggered geometries.
    """
    weights_per_dim = []
    for cmp_idx in range(grid.ndim): # loop over U, V, [W] components
        w_cmp = []
        for ax in range(grid.ndim):
            # The coordinate lives at face center for ax == cmp_idx, else cell center
            axis_grid_1d = grid.face_centers(cmp_idx)[ax] if ax == cmp_idx else grid.cell_centers()[ax]
            W = compute_weights_1d(axis_grid_1d, coords_tuple[ax], grid.spacings[ax])
            w_cmp.append(W)
        weights_per_dim.append(tuple(w_cmp))
    return tuple(weights_per_dim)

def interpolate_mac_velocity(velocity: Tuple[jnp.ndarray, ...], weights, grid: MACGrid) -> Tuple[jnp.ndarray, ...]:
    """Interpolate Eulerian velocities to Lagrangian points using Separable Einsum outer-products."""
    dv = jnp.prod(jnp.array(grid.spacings))
    u_lag_list = []
    
    for i, u_cmp in enumerate(velocity):
        W_tuple = weights[i]
        if grid.ndim == 2:
            Wx, Wy = W_tuple
            u_lag = jnp.einsum('ki,ij,kj->k', Wx, u_cmp, Wy) * dv
        else: # 3D: Wx[k, i], u[i,j,m], Wy[k, j], Wz[k, m] -> k
            Wx, Wy, Wz = W_tuple
            u_lag = jnp.einsum('ki,ijm,kj,km->k', Wx, u_cmp, Wy, Wz) * dv
        u_lag_list.append(u_lag)
        
    return tuple(u_lag_list)

def spread_mac_forcing(delta_U: Tuple[jnp.ndarray, ...], weights, grid: MACGrid) -> Tuple[jnp.ndarray, ...]:
    """Spread Lagrangian velocity corrections back to the Eulerian MAC Grid using Separable Einsum."""
    dv = jnp.prod(jnp.array(grid.spacings))
    du_eul_list = []
    
    for i, dU_lag in enumerate(delta_U):
        W_tuple = weights[i]
        # dU_lag * dv is perfectly safe.
        F = dU_lag * dv
        if grid.ndim == 2:
            Wx, Wy = W_tuple
            du_eul = jnp.einsum('ki,k,kj->ij', Wx, F, Wy)
        else:
            Wx, Wy, Wz = W_tuple
            du_eul = jnp.einsum('ki,k,kj,km->ijm', Wx, F, Wy, Wz)
        du_eul_list.append(du_eul)
        
    return tuple(du_eul_list)

def apply_ibm_forcing(velocity: Tuple[jnp.ndarray, ...], U_tgt: Tuple[jnp.ndarray, ...],
                      coords_tuple: Tuple[jnp.ndarray, ...], grid: MACGrid,
                      weights=None) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
    """
    Executes the orthodox Direct Forcing Immersed Boundary Method constraint on a field.

    Parameters
    ----------
    weights : optional
        Pre-computed tensor weights from ``compute_tensor_weights_mac``.
        Pass this for static bodies to avoid recomputing every step.
    """
    if weights is None:
        weights = compute_tensor_weights_mac(grid, coords_tuple)

    # 1. Eulerian-to-Lagrangian: Measure fluid velocity exactly at the body
    U_interp = interpolate_mac_velocity(velocity, weights, grid)

    # 2. Compute Direct Velocity Deficiency
    dU_lag_list = [u_tgt_c - u_int_c for u_tgt_c, u_int_c in zip(U_tgt, U_interp)]

    # 3. Lagrangian-to-Eulerian: Spread precise deficiency reaction back to the grid
    dU_eul_list = spread_mac_forcing(tuple(dU_lag_list), weights, grid)

    # 4. Integrate constraints
    vel_corrected = tuple(u + du for u, du in zip(velocity, dU_eul_list))
    return vel_corrected, tuple(dU_lag_list)


# ── IBM Scalar (Temperature) Forcing ─────────────────────────────────────

def compute_weights_cell_center(grid: MACGrid, coords_tuple: Tuple[jnp.ndarray, ...]):
    """Peskin delta weights at cell centers for scalar interpolation/spreading."""
    cc = grid.cell_centers()
    return tuple(
        compute_weights_1d(cc[ax], coords_tuple[ax], grid.spacings[ax])
        for ax in range(grid.ndim)
    )


def interpolate_scalar(scalar: jnp.ndarray, weights, grid: MACGrid) -> jnp.ndarray:
    """Interpolate a cell-centered scalar to Lagrangian marker locations."""
    dv = jnp.prod(jnp.array(grid.spacings))
    if grid.ndim == 2:
        Wx, Wy = weights
        return jnp.einsum('ki,ij,kj->k', Wx, scalar, Wy) * dv
    else:
        Wx, Wy, Wz = weights
        return jnp.einsum('ki,ijm,kj,km->k', Wx, scalar, Wy, Wz) * dv


def spread_scalar(dT_lag: jnp.ndarray, weights, grid: MACGrid) -> jnp.ndarray:
    """Spread Lagrangian scalar correction back to the Eulerian cell-center grid."""
    dv = jnp.prod(jnp.array(grid.spacings))
    F = dT_lag * dv
    if grid.ndim == 2:
        Wx, Wy = weights
        return jnp.einsum('ki,k,kj->ij', Wx, F, Wy)
    else:
        Wx, Wy, Wz = weights
        return jnp.einsum('ki,k,kj,km->ijm', Wx, F, Wy, Wz)


def apply_ibm_scalar_forcing(scalar: jnp.ndarray, T_tgt: float,
                              coords_tuple: Tuple[jnp.ndarray, ...],
                              grid: MACGrid, weights=None) -> jnp.ndarray:
    """Enforce Dirichlet temperature T_tgt on an IBM body surface via direct forcing."""
    if weights is None:
        weights = compute_weights_cell_center(grid, coords_tuple)
    T_interp = interpolate_scalar(scalar, weights, grid)
    dT_lag = T_tgt - T_interp
    dT_eul = spread_scalar(dT_lag, weights, grid)
    return scalar + dT_eul


# ── Wall-Modeled IBM ─────────────────────────────────────────────────────

# Log-law constants (same as boundaries.py)
_KAPPA = 0.41
_E = 9.793
_YPLUS_SWITCH = 11.6


def compute_probe_points(coords_tuple: Tuple[jnp.ndarray, ...],
                         normals: Tuple[jnp.ndarray, ...],
                         probe_distance: float) -> Tuple[jnp.ndarray, ...]:
    """Offset Lagrangian markers along outward normals to get probe locations.

    Parameters
    ----------
    coords_tuple : (xs, ys) marker coordinates
    normals : (nx, ny) outward unit normals at each marker
    probe_distance : distance along normal (typically 2*dx)

    Returns
    -------
    probe_coords : (xs_probe, ys_probe)
    """
    return tuple(c + probe_distance * n for c, n in zip(coords_tuple, normals))


def wall_shear_from_log_law(u_tan_mag: jnp.ndarray, y1: float,
                            nu: float, n_iter: int = 10):
    """Compute wall shear stress from tangential velocity via log-law.

    Uses the same Newton iteration as ``boundaries._wall_function_ghost``
    but returns (tau_w, u_tau) instead of ghost-cell values.

    Parameters
    ----------
    u_tan_mag : (n_pts,) magnitude of tangential velocity at probe points
    y1 : wall-normal distance of probe points
    nu : molecular kinematic viscosity

    Returns
    -------
    tau_w : (n_pts,) wall shear stress (= u_tau^2)
    u_tau : (n_pts,) friction velocity
    """
    _EPS = 1e-14
    u_mag = jnp.maximum(u_tan_mag, 1e-12)
    nu_safe = jnp.maximum(nu, _EPS)

    # Viscous sublayer formula (gradient-safe sqrt)
    u_tau_visc = jnp.sqrt(jnp.maximum(nu_safe * u_mag / y1, _EPS))

    # Predict regime from viscous estimate (before Newton)
    yplus_visc = y1 * u_tau_visc / nu_safe
    is_viscous = yplus_visc < _YPLUS_SWITCH

    # Feed safe u_mag to Newton in viscous regime (where Newton result
    # is discarded). This prevents NaN gradients from unconverged Newton
    # leaking through jnp.where via IEEE 754 NaN*0=NaN.
    # In log-law regime u_mag is used as-is.
    u_mag_safe = jnp.where(is_viscous, 1.0, u_mag)

    # Newton iteration for log-law u_tau (always converges with safe input)
    u_tau = jnp.sqrt(jnp.maximum(nu_safe * u_mag_safe / y1, _EPS))
    for _ in range(n_iter):
        u_tau_s = jnp.maximum(u_tau, 1e-12)
        yplus = y1 * u_tau_s / nu_safe
        f = u_mag_safe / u_tau_s - (1.0 / _KAPPA) * jnp.log(jnp.maximum(_E * yplus, 1.0))
        fp = -u_mag_safe / (u_tau_s ** 2) - 1.0 / (_KAPPA * u_tau_s)
        delta = f / jnp.minimum(fp, -_EPS)
        u_tau = jnp.maximum(u_tau_s - delta, 1e-12)

    # Select: viscous sublayer (y+ < 11.6) or log-law
    u_tau = jnp.where(is_viscous, u_tau_visc, u_tau)

    tau_w = u_tau ** 2
    return tau_w, u_tau


def apply_ibm_wall_model(velocity: Tuple[jnp.ndarray, ...],
                         coords_tuple: Tuple[jnp.ndarray, ...],
                         normals: Tuple[jnp.ndarray, ...],
                         ds: jnp.ndarray,
                         grid: MACGrid,
                         nu: float,
                         dt: float,
                         probe_distance: float,
                         penalty_factor: float = 0.1,
                         weights=None,
                         probe_weights=None):
    """Wall-modeled IBM: apply log-law friction + weak no-penetration penalty.

    Instead of direct-forcing velocity to zero, this:
    1. Interpolates velocity at probe points (offset along normals)
    2. Decomposes into tangential and normal components
    3. Computes wall shear stress from the log-law
    4. Converts to velocity correction (dU = tau_w * dt) for spreading
    5. Applies a weak direct-forcing penalty for the normal component

    Parameters
    ----------
    velocity : (u, v) Eulerian velocity fields
    coords_tuple : (xs, ys) Lagrangian marker coordinates on the surface
    normals : (nx, ny) outward unit normals
    ds : (n_pts,) arc-length per marker
    grid : MACGrid
    nu : molecular kinematic viscosity
    dt : sub-step time increment (for tau_w → velocity conversion)
    probe_distance : distance of probe points from surface
    penalty_factor : strength of no-penetration penalty (default 0.1)
    weights : pre-computed weights at surface points (for penalty)
    probe_weights : pre-computed weights at probe points (for interpolation)

    Returns
    -------
    vel_corrected : (u, v) corrected velocity
    body_forces : (dU_x, dU_y) Lagrangian velocity deficiency per marker
                  (same convention as direct-forcing for extract_ibm_forces)
    """
    ndim = grid.ndim

    # Pre-compute weights if not provided
    if weights is None:
        weights = compute_tensor_weights_mac(grid, coords_tuple)

    # Probe point coordinates and weights
    probe_coords = compute_probe_points(coords_tuple, normals, probe_distance)
    if probe_weights is None:
        probe_weights = compute_tensor_weights_mac(grid, probe_coords)

    # 1. Interpolate velocity at probe points
    U_probe = interpolate_mac_velocity(velocity, probe_weights, grid)

    # 2. Decompose into normal and tangential components
    # u_n = dot(U, n)
    u_normal = sum(U_probe[i] * normals[i] for i in range(ndim))
    # u_tan = U - u_n * n
    u_tan = tuple(U_probe[i] - u_normal * normals[i] for i in range(ndim))
    u_tan_mag = jnp.sqrt(sum(u_tan[i] ** 2 for i in range(ndim)) + 1e-14)

    # Tangential direction (unit vector)
    u_tan_dir = tuple(u_tan[i] / jnp.maximum(u_tan_mag, 1e-12)
                       for i in range(ndim))

    # 3. Log-law wall shear stress
    tau_w, _ = wall_shear_from_log_law(u_tan_mag, probe_distance, nu)

    # 4. Spread friction to Euler grid
    #    Physical: body force f(x) = integral tau_w * t_hat * delta(x-X) ds
    #    Discrete velocity update: du = f * dt
    #      = sum_k tau_w_k * t_hat_k * delta_h * ds_k * dt
    #    spread_mac_forcing(dU_lag) gives sum_k dU_lag_k * phi_x * phi_y,
    #    so dU_lag_k = -tau_w_k * u_tan_dir_k * dt * ds_k / dv
    dv = jnp.prod(jnp.array(grid.spacings))
    dU_friction_spread = tuple(-tau_w * u_tan_dir[i] * dt * ds / dv
                               for i in range(ndim))
    friction_eul = spread_mac_forcing(dU_friction_spread, weights, grid)

    # 5. Weak no-penetration penalty (velocity constraint, like direct-forcing)
    U_surf = interpolate_mac_velocity(velocity, weights, grid)
    u_normal_surf = sum(U_surf[i] * normals[i] for i in range(ndim))
    dU_penalty = tuple(-penalty_factor * u_normal_surf * normals[i]
                       for i in range(ndim))
    penalty_eul = spread_mac_forcing(dU_penalty, weights, grid)

    # 6. Combine
    vel_corrected = tuple(
        velocity[i] + friction_eul[i] + penalty_eul[i]
        for i in range(ndim)
    )

    # Body forces for diagnostics (per-marker velocity deficiency).
    # extract_ibm_forces does: F = -sum(dU * ds) / dt
    # Friction: dU = -tau_w * u_tan_dir * dt → F = sum(tau_w * u_tan_dir * ds) ✓
    # Penalty: dU = -penalty * u_n * n → F = sum(penalty * u_n * n * ds) / dt
    dU_friction_diag = tuple(-tau_w * u_tan_dir[i] * dt for i in range(ndim))
    body_forces = tuple(
        dU_friction_diag[i] + dU_penalty[i]
        for i in range(ndim)
    )

    return vel_corrected, body_forces


def compute_ibm_heat_source_field(
    q_flux: float,
    coords_tuple: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
    ds: jnp.ndarray,
    weights=None,
) -> jnp.ndarray:
    """Pre-compute volumetric heat source from constant surface heat flux.

    Parameters
    ----------
    q_flux : float
        Non-dimensional surface heat flux (power per unit area).
    coords_tuple : tuple of arrays
        Lagrangian marker coordinates.
    grid : MACGrid
    ds : array (n_markers,)
        Arc-length element per marker.
    weights : optional
        Pre-computed Peskin weights at cell centers.

    Returns
    -------
    Q_euler : array (Nx, Ny)
        Volumetric source field to add to scalar RHS.
    """
    if weights is None:
        weights = compute_weights_cell_center(grid, coords_tuple)
    q_lag = q_flux * ds          # heat per marker [T·length / time]
    return spread_scalar(q_lag, weights, grid)
