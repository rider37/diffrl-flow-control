"""
Design parameterisation utilities for topology optimisation.

Provides mappings from raw design variables to physical fields
(Brinkman alpha, thermal conductivity) and regularisation filters
(density filter, Sobolev gradient smoothing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ══════════════════════════════════════════════════════════════════════
#  Brinkman penalisation
# ══════════════════════════════════════════════════════════════════════

def brinkman_alpha(
    rho: jnp.ndarray,
    alpha_max: float = 1e4,
    q: float = 0.01,
) -> jnp.ndarray:
    """RAMP interpolation from design density to Brinkman impermeability.

    rho=1 -> fluid (alpha=0),  rho=0 -> solid (alpha ~ alpha_max).
    """
    return alpha_max * q * (1.0 - rho) / (q + rho)


def make_kappa_field(
    rho: jnp.ndarray,
    kappa_base: float,
    k_ratio: float = 50.0,
) -> jnp.ndarray:
    """Map design density to spatially-varying thermal diffusivity.

    rho=1 -> fluid (kappa_base),  rho=0 -> solid (kappa_base * k_ratio).
    Returns the *extra* diffusivity to add to the base value.
    """
    return kappa_base * (k_ratio - 1.0) * (1.0 - rho)


# ══════════════════════════════════════════════════════════════════════
#  Smooth max
# ══════════════════════════════════════════════════════════════════════

def smooth_max(
    field: jnp.ndarray,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable approximation of max(field) via LogSumExp."""
    return (1.0 / beta) * jax.nn.logsumexp(beta * field.ravel())


# ══════════════════════════════════════════════════════════════════════
#  Density filter (cone-weighted convolution)
# ══════════════════════════════════════════════════════════════════════

def density_filter(
    rho: jnp.ndarray,
    radius: float,
    grid,
) -> jnp.ndarray:
    """Cone-weighted density filter to suppress checkerboard patterns."""
    from jax.scipy.signal import convolve
    ndim = grid.ndim
    dx = grid.spacings

    half = tuple(max(int(radius / dx[i] + 0.5), 1) for i in range(ndim))
    ranges = [jnp.arange(-h, h + 1, dtype=jnp.float32) * dx[i]
              for i, h in enumerate(half)]
    meshes = jnp.meshgrid(*ranges, indexing='ij')
    dist = jnp.sqrt(sum(m ** 2 for m in meshes))

    kernel = jnp.maximum(radius - dist, 0.0)
    kernel = kernel / jnp.sum(kernel)

    return convolve(rho, kernel, mode='same')


# ══════════════════════════════════════════════════════════════════════
#  Sobolev gradient (Helmholtz filter via FFT)                  [Method 4]
# ══════════════════════════════════════════════════════════════════════

def sobolev_filter(
    gradient: jnp.ndarray,
    epsilon: float,
    grid,
) -> jnp.ndarray:
    """Helmholtz-filtered (Sobolev) gradient smoothing.

    Solves ``(I - epsilon^2 * nabla^2) g_smooth = g_raw`` spectrally
    using DCT-II (Neumann BCs), reusing the same eigenvalue structure
    as the pressure Poisson solver.

    Parameters
    ----------
    gradient : jnp.ndarray
        Raw gradient field (cell-centred), same shape as design variable.
    epsilon : float
        Smoothing length scale.  Typical: 2-4 cell widths.
        ``epsilon = 0`` returns the raw gradient unchanged.
    grid : MACGrid
        Grid object (used for spacings and shape).

    Returns
    -------
    g_smooth : jnp.ndarray  (same shape)

    Notes
    -----
    This replaces the cone-weighted density filter for gradient
    regularisation.  Advantages over convolution-based filters:

    * O(N log N) cost via FFT (vs O(N * r^d) for convolution)
    * Intrinsic smoothness (H^1 norm minimisation)
    * No kernel truncation artefacts at domain boundaries
    """
    if epsilon <= 0.0:
        return gradient

    import jax.scipy.fft as jfft

    ndim = grid.ndim

    # Forward DCT-II along each axis
    g_hat = gradient
    for ax in range(ndim):
        g_hat = jfft.dct(g_hat, axis=ax, type=2)

    # Build Helmholtz denominator: 1 - epsilon^2 * sum_i lambda_i
    # where lambda_i = 2*(cos(pi*k/N) - 1) / dx^2  (DCT eigenvalues)
    denom = jnp.ones(grid.cell_shape())
    for ax in range(ndim):
        n_ax = grid.shape[ax]
        dx_ax = grid.spacings[ax]
        k = jnp.arange(n_ax)
        lam = 2.0 * (jnp.cos(jnp.pi * k / n_ax) - 1.0) / (dx_ax ** 2)
        shape = [1] * ndim
        shape[ax] = n_ax
        denom = denom - epsilon ** 2 * lam.reshape(shape)

    g_hat = g_hat / denom

    # Inverse DCT-II
    g_smooth = g_hat
    for ax in range(ndim):
        g_smooth = jfft.idct(g_smooth, axis=ax, type=2)

    return g_smooth


# ══════════════════════════════════════════════════════════════════════
#  Upscale / downscale for multi-resolution        [Method 1 helper]
# ══════════════════════════════════════════════════════════════════════

def upscale_field(rho: jnp.ndarray, target_shape: tuple) -> jnp.ndarray:
    """Bilinear upscale of design field from coarse to fine grid.

    Works for 2D (Nx, Ny) and 3D (Nx, Ny, Nz) fields by adding a
    temporary channel dimension that ``jax.image.resize`` requires.
    """
    import jax.image
    rho_ch = rho[..., jnp.newaxis]              # (..., 1) channel dim
    resized = jax.image.resize(rho_ch, target_shape + (1,), method='bilinear')
    return resized[..., 0]


def downscale_field(rho: jnp.ndarray, target_shape: tuple) -> jnp.ndarray:
    """Average-pool downscale of design field from fine to coarse grid.

    Uses reshape + mean to perform exact area-averaging (no aliasing).
    Falls back to bilinear resize when shapes are not integer multiples.
    """
    ndim = rho.ndim
    # Check if each axis is an integer multiple
    ratios = tuple(s // t for s, t in zip(rho.shape, target_shape))
    exact = all(s == t * r for s, t, r in zip(rho.shape, target_shape, ratios))

    if exact and all(r >= 1 for r in ratios):
        # Reshape to (t0, r0, t1, r1, ...) then mean over ratio axes
        new_shape = []
        mean_axes = []
        for i in range(ndim):
            new_shape.append(target_shape[i])
            new_shape.append(ratios[i])
            mean_axes.append(2 * i + 1)
        return rho.reshape(new_shape).mean(axis=tuple(mean_axes))
    else:
        import jax.image
        rho_ch = rho[..., jnp.newaxis]
        resized = jax.image.resize(rho_ch, target_shape + (1,), method='bilinear')
        return resized[..., 0]
