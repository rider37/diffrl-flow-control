"""k-ω SST (Shear Stress Transport) RANS turbulence model.

Menter (1994) two-equation model combining k-ω near walls and k-ε in
the freestream via blending functions F1, F2.

Transport equations:
    ∂k/∂t + u·∇k = P_k - β* k ω + ∇·((ν + σ_k ν_t) ∇k)
    ∂ω/∂t + u·∇ω = α P_k/ν_t - β ω² + ∇·((ν + σ_ω ν_t) ∇ω) + CD_kω

Eddy viscosity:
    ν_t = a1 k / max(a1 ω, |S| F2)

Usage (via runner CONFIG):
    'turbulence': {'model': 'k_omega_sst'}
"""
from __future__ import annotations

import jax.numpy as jnp
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.operators import laplacian_cell_centered, strain_rate_magnitude

# ── SST Constants (Menter 1994) ──────────────────────────────

# Inner layer (k-ω)
ALPHA1 = 5.0 / 9.0
BETA1 = 3.0 / 40.0
SIGMA_K1 = 0.85
SIGMA_W1 = 0.5

# Outer layer (k-ε transformed)
ALPHA2 = 0.44
BETA2 = 0.0828
SIGMA_K2 = 1.0
SIGMA_W2 = 0.856

# Common
BETA_STAR = 0.09
A1 = 0.31
KAPPA = 0.41


def _wall_distance(grid: MACGrid, bc_config: dict = None) -> jnp.ndarray:
    """Approximate wall-normal distance for each cell.

    For structured grids: distance to nearest wall boundary.
    Uses y-coordinate distance to y_min / y_max walls by default.
    """
    coords = grid.cell_centers()
    ndim = grid.ndim
    d = jnp.full(grid.cell_shape(), 1e10)

    # Simple approach: minimum distance to domain boundaries that are walls
    for axis in range(ndim):
        x = coords[axis]
        shape = [1] * ndim
        shape[axis] = len(x)
        x_bc = jnp.reshape(x, shape)
        L = grid.domain_size[axis]
        d_min = x_bc
        d_max = L - x_bc
        d = jnp.minimum(d, jnp.minimum(d_min, d_max))

    return d


def _strain_rate_mag(velocity, grid: MACGrid) -> jnp.ndarray:
    """Compute |S| = √(2 S_ij S_ij) at cell centres."""
    return strain_rate_magnitude(velocity, grid)


def make_k_omega_sst(grid: MACGrid, nu: float):
    """Create k-ω SST model components.

    Returns
    -------
    scalar_rhs_fn : (scalars, vel, state, nu_base) -> dict
        RHS for k and ω transport equations.
    get_nu_t_fn : (state, grid, params) -> ndarray
        Eddy viscosity closure.
    init_scalars : dict
        Initial k and ω fields.
    """
    wall_d = _wall_distance(grid)

    def _blending_F1(k, omega, d, nu_val):
        """Menter's F1 blending function (1 near wall, 0 far)."""
        d_safe = jnp.maximum(d, 1e-10)
        k_safe = jnp.maximum(k, 1e-20)
        omega_safe = jnp.maximum(omega, 1e-20)

        arg1_a = jnp.sqrt(k_safe) / (BETA_STAR * omega_safe * d_safe)
        arg1_b = 500.0 * nu_val / (d_safe ** 2 * omega_safe)
        arg1 = jnp.minimum(jnp.maximum(arg1_a, arg1_b), 1e10)

        # Cross-diffusion term for F1
        dk = jnp.zeros_like(k)  # simplified: omit gradient-based CD for now
        CD_kw = jnp.maximum(2.0 * SIGMA_W2 / omega_safe * dk, 1e-10)
        arg1_c = 4.0 * SIGMA_W2 * k_safe / (CD_kw * d_safe ** 2)
        arg1 = jnp.minimum(arg1, arg1_c)

        return jnp.tanh(arg1 ** 4)

    def _blending_F2(k, omega, d, nu_val):
        """Menter's F2 blending function for eddy viscosity limiter."""
        d_safe = jnp.maximum(d, 1e-10)
        k_safe = jnp.maximum(k, 1e-20)
        omega_safe = jnp.maximum(omega, 1e-20)

        arg2_a = 2.0 * jnp.sqrt(k_safe) / (BETA_STAR * omega_safe * d_safe)
        arg2_b = 500.0 * nu_val / (d_safe ** 2 * omega_safe)
        arg2 = jnp.maximum(arg2_a, arg2_b)
        return jnp.tanh(arg2 ** 2)

    def _blend(phi1, phi2, F1):
        """Blend inner/outer constants: φ = F1·φ1 + (1-F1)·φ2."""
        return F1 * phi1 + (1.0 - F1) * phi2

    def _laplacian(field, grid_arg):
        """Cell-centred Laplacian with zero-gradient BC."""
        return laplacian_cell_centered(field, grid_arg)

    def _grad_dot(f1, f2, grid_arg):
        """∇f1 · ∇f2 at cell centres (for cross-diffusion)."""
        ndim = grid_arg.ndim
        result = jnp.zeros_like(f1)
        for ax in range(ndim):
            dx = grid_arg.spacings[ax]
            pad_w = [(0, 0)] * ndim
            pad_w[ax] = (1, 1)
            f1p = jnp.pad(f1, pad_w, mode='edge')
            f2p = jnp.pad(f2, pad_w, mode='edge')
            s = [slice(None)] * ndim
            sp = [slice(None)] * ndim
            sm = [slice(None)] * ndim
            sp[ax] = slice(2, None)
            sm[ax] = slice(None, -2)
            s[ax] = slice(1, -1)
            df1 = (f1p[tuple(sp)] - f1p[tuple(sm)]) / (2.0 * dx)
            df2 = (f2p[tuple(sp)] - f2p[tuple(sm)]) / (2.0 * dx)
            result = result + df1 * df2
        return result

    def get_nu_t_fn(state, grid_arg, params):
        """Compute ν_t = a1·k / max(a1·ω, |S|·F2)."""
        k = state.scalars.get('k', jnp.zeros(grid_arg.cell_shape()))
        omega = state.scalars.get('omega', jnp.ones(grid_arg.cell_shape()))

        k_safe = jnp.maximum(k, 0.0)
        omega_safe = jnp.maximum(omega, 1e-20)

        S_mag = _strain_rate_mag(state.velocity, grid_arg)
        F2 = _blending_F2(k_safe, omega_safe, wall_d, nu)

        nu_t = A1 * k_safe / jnp.maximum(A1 * omega_safe, S_mag * F2 + 1e-20)
        return jnp.clip(nu_t, 0.0, 1000.0 * nu)

    def scalar_rhs_fn(scalars, vel, state, nu_base_val):
        """RHS for k and ω transport."""
        k = jnp.maximum(scalars.get('k', jnp.zeros(grid.cell_shape())), 0.0)
        omega = jnp.maximum(scalars.get('omega', jnp.ones(grid.cell_shape())), 1e-20)

        # Strain rate
        S_mag = _strain_rate_mag(vel, grid)

        # Blending
        F1 = _blending_F1(k, omega, wall_d, nu)

        # Blended constants
        alpha = _blend(ALPHA1, ALPHA2, F1)
        beta = _blend(BETA1, BETA2, F1)
        sigma_k = _blend(SIGMA_K1, SIGMA_K2, F1)
        sigma_w = _blend(SIGMA_W1, SIGMA_W2, F1)

        # Eddy viscosity (for production and diffusion)
        F2 = _blending_F2(k, omega, wall_d, nu)
        nu_t = A1 * k / jnp.maximum(A1 * omega, S_mag * F2 + 1e-20)
        nu_t = jnp.clip(nu_t, 0.0, 1000.0 * nu)

        # Production
        P_k = nu_t * S_mag ** 2
        P_k = jnp.minimum(P_k, 10.0 * BETA_STAR * k * omega)  # clip

        # k equation: P_k - β*kω + ∇·((ν + σ_k ν_t) ∇k)
        k_diff = (nu + sigma_k * nu_t) * _laplacian(k, grid)
        dk_dt = P_k - BETA_STAR * k * omega + k_diff

        # ω equation: α P_k / ν_t - β ω² + ∇·((ν + σ_ω ν_t) ∇ω) + CD_kω
        P_omega = alpha * S_mag ** 2
        omega_diff = (nu + sigma_w * nu_t) * _laplacian(omega, grid)

        # Cross-diffusion (only active in outer layer where F1 < 1)
        CD_kw = 2.0 * (1.0 - F1) * SIGMA_W2 / omega * _grad_dot(k, omega, grid)
        CD_kw = jnp.maximum(CD_kw, 0.0)

        domega_dt = P_omega - beta * omega ** 2 + omega_diff + CD_kw

        # Pass through other scalars unchanged
        rhs = {}
        for name, val in scalars.items():
            if name == 'k':
                rhs[name] = dk_dt
            elif name == 'omega':
                rhs[name] = domega_dt
            else:
                rhs[name] = jnp.zeros_like(val)
        return rhs

    # Initial fields
    k0 = 1.5 * (0.01) ** 2  # TI = 1%
    omega0 = k0 / (nu * 10.0)
    init_scalars = {
        'k': jnp.full(grid.cell_shape(), k0),
        'omega': jnp.full(grid.cell_shape(), omega0),
    }

    return scalar_rhs_fn, get_nu_t_fn, init_scalars
