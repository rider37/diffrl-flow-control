"""
Cahn-Hilliard Phase-Field model for multiphase flow.

Solves the coupled system:

    ∂φ/∂t + u·∇φ = M ∇²μ               (phase-field advection-diffusion)
    μ = f'(φ) - ε² ∇²φ                  (chemical potential)
    f(φ) = ¼(φ²-1)²  →  f'(φ) = φ³ - φ  (double-well potential)

Surface tension enters the momentum equation as:

    F_σ,i = (σ/ε) μ (∂φ/∂x_i)          (capillary force at faces)

References
----------
- Jacqmin (1999), J. Comput. Phys. 155, 96–127
- Kim (2012), Commun. Comput. Phys. 12, 613–661
"""

from __future__ import annotations

import jax.numpy as jnp
from typing import Callable, Dict, Tuple

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.operators import diff_axis_padded_nu


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def _extract_axis_pad(padded: jnp.ndarray, axis: int, ndim: int) -> jnp.ndarray:
    """Keep ghost padding on *axis* only; strip ghosts on all other axes."""
    sl = [slice(None)] * ndim
    for k in range(ndim):
        if k != axis:
            sl[k] = slice(1, -1)
    return padded[tuple(sl)]


def _pad_neumann(field: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """Pad a cell-centred field with 1 ghost layer (Neumann ∂f/∂n=0)."""
    return jnp.pad(field, [(1, 1)] * ndim, mode='edge')


def _laplacian(field: jnp.ndarray, grid: MACGrid, pad_fn) -> jnp.ndarray:
    """Compute ∇²f using ghost-cell padding and existing Laplacian operator."""
    padded = pad_fn(field, grid.ndim)
    lap = jnp.zeros(grid.cell_shape())
    for j in range(grid.ndim):
        lap = lap + diff_axis_padded_nu(
            _extract_axis_pad(padded, j, grid.ndim), j, grid
        )
    return lap


# ──────────────────────────────────────────────────────────────────────
#  Chemical potential
# ──────────────────────────────────────────────────────────────────────

def compute_chemical_potential(
    phi: jnp.ndarray,
    grid: MACGrid,
    epsilon: float,
    pad_fn,
) -> jnp.ndarray:
    """Chemical potential  μ = f'(φ) - ε² ∇²φ  (cell-centred).

    Parameters
    ----------
    phi : cell-centred phase-field array.
    grid : MACGrid.
    epsilon : interface half-width parameter.
    pad_fn : callable  ``(field, ndim) -> padded_field``  for ghost cells.
    """
    lap_phi = _laplacian(phi, grid, pad_fn)
    return phi ** 3 - phi - epsilon ** 2 * lap_phi


# ──────────────────────────────────────────────────────────────────────
#  Cahn-Hilliard scalar RHS
# ──────────────────────────────────────────────────────────────────────

def make_ch_scalar_rhs_fn(
    grid: MACGrid,
    mobility: float,
    epsilon: float,
    advect_scalar_fn: Callable,
    pad_phi_fn: Callable,
) -> Callable:
    """Build the Cahn-Hilliard RHS function for the ``'phi'`` scalar.

    Returns a callable with signature::

        (scalars, vel, state, nu_base_val) -> Dict[str, jnp.ndarray]

    matching ``compute_scalar_rhs`` in ``solver.py``.

    Parameters
    ----------
    grid : MACGrid
    mobility : M  (CH mobility coefficient).
    epsilon : ε  (interface half-width).
    advect_scalar_fn : scalar advection function (e.g. ``advect_scalar_muscl``).
    pad_phi_fn : ``(phi, 'phi', grid) -> padded``  scalar BC padding function.
    """

    def _pad_phi(field, _ndim):
        """Pad φ using BC-aware padding (handles periodic, Dirichlet, Neumann)."""
        return pad_phi_fn(field, 'phi', grid)

    def _pad_mu(field, _ndim):
        """Pad μ using BC-aware padding (periodic → wrap, wall → Neumann).

        Uses scalar name ``'mu_ch'`` which won't match any Dirichlet BC value,
        so walls default to Neumann (∂μ/∂n = 0 → no-flux), while periodic
        axes are correctly wrapped.
        """
        return pad_phi_fn(field, 'mu_ch', grid)

    def ch_rhs(
        scalars: Dict[str, jnp.ndarray],
        vel: Tuple[jnp.ndarray, ...],
        state,
        nu_base_val,
    ) -> Dict[str, jnp.ndarray]:
        phi = scalars['phi']

        # 1. Chemical potential: μ = φ³ - φ - ε²∇²φ
        mu = compute_chemical_potential(phi, grid, epsilon, _pad_phi)

        # 2. Laplacian of μ  (periodic → wrap, wall → Neumann ∂μ/∂n = 0)
        lap_mu = _laplacian(mu, grid, _pad_mu)

        # 3. Advection of φ: u·∇φ  (MUSCL)
        adv_phi = advect_scalar_fn(phi, vel, grid)

        # 4. Full CH RHS:  dφ/dt = -u·∇φ + M ∇²μ
        return {'phi': -adv_phi + mobility * lap_mu}

    return ch_rhs


# ──────────────────────────────────────────────────────────────────────
#  Surface tension force for momentum
# ──────────────────────────────────────────────────────────────────────

def make_surface_tension_fn(
    grid: MACGrid,
    sigma: float,
    epsilon: float,
    pad_phi_fn: Callable,
    delta_rho: float = 0.0,
    g_vector: Tuple[float, ...] = (),
) -> Callable:
    """Build the surface tension + density-buoyancy momentum source.

    Returns a callable with signature::

        (scalars, velocity, grid) -> Tuple[jnp.ndarray, ...]

    Each element is a face-centred force component added to the momentum RHS.

    Parameters
    ----------
    grid : MACGrid
    sigma : surface tension coefficient σ.
    epsilon : interface half-width ε.
    pad_phi_fn : scalar BC padding function.
    delta_rho : density difference  (ρ₁ - ρ₂).  0 disables buoyancy.
    g_vector : gravity direction/magnitude per axis.
    """
    coeff_st = sigma / max(epsilon, 1e-30)
    has_buoyancy = abs(delta_rho) > 1e-12 and len(g_vector) > 0

    def _pad_phi(field, _ndim):
        return pad_phi_fn(field, 'phi', grid)

    def _pad_mu(field, _ndim):
        # 'mu_ch' won't match any Dirichlet BC → walls get Neumann (∂μ/∂n=0),
        # periodic axes get wrap.  Same pattern as in make_ch_scalar_rhs_fn.
        return pad_phi_fn(field, 'mu_ch', grid)

    def source_fn(
        scalars: Dict[str, jnp.ndarray],
        velocity: Tuple[jnp.ndarray, ...],
        _grid: MACGrid,
    ) -> Tuple[jnp.ndarray, ...]:
        phi = scalars['phi']
        mu = compute_chemical_potential(phi, _grid, epsilon, _pad_phi)

        # BC-aware padding once (handles periodic wrap + wall Neumann correctly)
        phi_padded = _pad_phi(phi, _grid.ndim)
        mu_padded = _pad_mu(mu, _grid.ndim)

        forces = []
        for i in range(_grid.ndim):
            # Single-axis padded view for face gradient
            phi_i = _extract_axis_pad(phi_padded, i, _grid.ndim)
            mu_i = _extract_axis_pad(mu_padded, i, _grid.ndim)

            sl_p = [slice(None)] * _grid.ndim
            sl_m = [slice(None)] * _grid.ndim
            sl_p[i] = slice(1, None)
            sl_m[i] = slice(None, -1)

            # ∂φ/∂x_i at i-faces  (cell-to-face finite difference)
            dx_i = _grid.spacings[i] if isinstance(_grid.spacings, tuple) else _grid.spacings
            dphi_face = (phi_i[tuple(sl_p)] - phi_i[tuple(sl_m)]) / dx_i

            # μ interpolated to i-faces
            mu_face = 0.5 * (mu_i[tuple(sl_p)] + mu_i[tuple(sl_m)])

            # Capillary force:  F_σ = (σ/ε) μ ∂φ/∂x_i
            f_i = coeff_st * mu_face * dphi_face

            # Density buoyancy:  (Δρ/2) φ g_i  at faces
            if has_buoyancy and i < len(g_vector) and abs(g_vector[i]) > 1e-12:
                phi_face = 0.5 * (phi_i[tuple(sl_p)] + phi_i[tuple(sl_m)])
                f_i = f_i + (delta_rho / 2.0) * phi_face * g_vector[i]

            forces.append(f_i)
        return tuple(forces)

    return source_fn


# ──────────────────────────────────────────────────────────────────────
#  Boiling: combined CH + heat transport with phase-change coupling
# ──────────────────────────────────────────────────────────────────────

def make_boiling_scalar_rhs_fn(
    grid: MACGrid,
    mobility: float,
    epsilon: float,
    advect_scalar_fn: Callable,
    pad_scalar_fn: Callable,
    Pr: float,
    K_evap: float,
    T_sat: float,
    latent_heat: float,
    get_nu_t_fn: Callable,
    thermal_bodies: list = None,
) -> Callable:
    """Build combined Cahn-Hilliard + heat transport RHS with phase-change coupling.

    Returns a callable with the same signature as ``compute_scalar_rhs``
    that advances both ``'phi'`` and ``'T'`` simultaneously.

    Phase-change model
    ------------------
    * Source in φ equation: ``S_φ = K (T - T_sat)(1 - φ²)``
      Evaporation (T > T_sat) drives φ toward +1 (vapor) at the interface.
    * Sink in T equation: ``S_T = -L · S_φ``
      Latent heat absorption cools the interface during evaporation.

    Thermal body model (conjugate heat transfer)
    ---------------------------------------------
    When ``thermal_bodies`` is provided, bodies have finite thermal mass.
    A penalty source ``γ · w(x) · (T_body - T)`` keeps T ≈ T_body inside the
    body, while the body temperature evolves via an energy balance:

        C · dT_body/dt = Q_gen - γ · ∫ w(x)(T_body - T) dV

    Parameters
    ----------
    grid : MACGrid
    mobility : M  (CH mobility coefficient).
    epsilon : ε  (interface half-width).
    advect_scalar_fn : scalar advection function (e.g. ``advect_scalar_muscl``).
    pad_scalar_fn : ``(field, name, grid) -> padded``  scalar BC padding function.
    Pr : Prandtl number for temperature diffusion.
    K_evap : evaporation/condensation rate coefficient.
    T_sat : saturation temperature.
    latent_heat : latent heat coefficient L.
    get_nu_t_fn : turbulence model closure ``(state, grid, {}) -> nu_t``.
    thermal_bodies : list of dicts, optional
        Each dict: ``{'indicator': array, 'gamma': float, 'C': float, 'Q_gen': float}``.
        ``indicator`` is a cell-centred field (1 inside body, 0 outside).
    """
    Pr_t = 0.85
    _has_thermal = thermal_bodies is not None and len(thermal_bodies) > 0
    n_bodies = len(thermal_bodies) if _has_thermal else 0
    V_cell = float(jnp.prod(jnp.array(grid.spacings)))

    def _pad_phi(field, _ndim):
        return pad_scalar_fn(field, 'phi', grid)

    def _pad_mu(field, _ndim):
        return pad_scalar_fn(field, 'mu_ch', grid)

    def _pad_T(field, _ndim):
        return pad_scalar_fn(field, 'T', grid)

    def boiling_rhs(
        scalars: Dict[str, jnp.ndarray],
        vel: Tuple[jnp.ndarray, ...],
        state,
        nu_base_val,
    ) -> Dict[str, jnp.ndarray]:
        phi = scalars['phi']
        T = scalars['T']

        # ── φ: Cahn-Hilliard + phase change ──────────────────────────
        mu = compute_chemical_potential(phi, grid, epsilon, _pad_phi)
        lap_mu = _laplacian(mu, grid, _pad_mu)
        adv_phi = advect_scalar_fn(phi, vel, grid)

        # Phase-change source: concentrated at interface (1 - φ²)
        interface_fn = 1.0 - phi ** 2
        S_phase = K_evap * (T - T_sat) * interface_fn

        phi_rhs = -adv_phi + mobility * lap_mu + S_phase

        # ── T: advection-diffusion + latent heat ─────────────────────
        nu_t = get_nu_t_fn(state, grid, {})
        kappa = nu_base_val / Pr + nu_t / Pr_t

        adv_T = advect_scalar_fn(T, vel, grid)

        T_padded = _pad_T(T, grid.ndim)
        diff_T = jnp.zeros(grid.cell_shape())
        for j in range(grid.ndim):
            diff_T = diff_T + diff_axis_padded_nu(
                _extract_axis_pad(T_padded, j, grid.ndim), j, grid
            ) * kappa

        T_rhs = -adv_T + diff_T - latent_heat * S_phase

        # ── Thermal bodies: penalty source + body_temps RHS ──────────
        rhs = {'phi': phi_rhs, 'T': T_rhs}
        if _has_thermal:
            body_temps = scalars['body_temps']
            bt_rhs = jnp.zeros(n_bodies)
            for i in range(n_bodies):
                tb = thermal_bodies[i]
                w_i = tb['indicator']
                gamma_i = tb['gamma']
                C_i = tb['C']
                Q_gen_i = tb['Q_gen']
                T_body_i = body_temps[i]

                # Penalty: drive T toward T_body inside body
                penalty_i = gamma_i * w_i * (T_body_i - T)
                rhs['T'] = rhs['T'] + penalty_i

                # Heat removal rate = gamma * integral(w * (T_body - T) dV)
                Q_rem = gamma_i * jnp.sum(w_i * (T_body_i - T)) * V_cell
                bt_rhs = bt_rhs.at[i].set((Q_gen_i - Q_rem) / C_i)

            rhs['body_temps'] = bt_rhs

        return rhs

    return boiling_rhs
