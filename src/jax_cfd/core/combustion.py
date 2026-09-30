"""Species transport with Arrhenius combustion kinetics.

Single-step global reaction model:
    ν_F Fuel + ν_O Oxidiser → ν_P Product + Q_heat

Species mass fractions Y_F, Y_O, Y_P are transported as scalars with
source terms from an Arrhenius reaction rate:

    ω̇ = A · ρ · Y_F^a · Y_O^b · exp(-E_a / (R T))

Usage (via runner CONFIG):
    'combustion': {
        'A':    1e8,              # pre-exponential factor
        'Ea':   1e4,              # activation energy / R_gas
        'Q':    5.0,              # heat release per unit fuel mass
        'nu_F': 1.0,              # stoichiometric coeff fuel
        'nu_O': 4.0,              # stoichiometric coeff oxidiser
        'D':    0.001,            # species diffusivity (all species)
    }
"""
from __future__ import annotations

import jax.numpy as jnp
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.advection import advect_scalar
from jax_cfd.core.operators import laplacian_cell_centered


def arrhenius_rate(T, Y_fuel, Y_ox, A, Ea_over_R, a=1.0, b=1.0):
    """Arrhenius reaction rate (non-dimensional).

    Parameters
    ----------
    T        : temperature field (cell-centred)
    Y_fuel   : fuel mass fraction
    Y_ox     : oxidiser mass fraction
    A        : pre-exponential factor
    Ea_over_R: activation energy / gas constant
    a, b     : reaction orders

    Returns
    -------
    omega_dot : reaction rate (mass of fuel consumed per unit time)
    """
    T_safe = jnp.maximum(T, 0.01)
    Y_f = jnp.maximum(Y_fuel, 0.0)
    Y_o = jnp.maximum(Y_ox, 0.0)
    return A * Y_f ** a * Y_o ** b * jnp.exp(-Ea_over_R / T_safe)


def make_combustion_rhs_fn(grid: MACGrid, config: dict, nu: float):
    """Build scalar_rhs_fn for species transport + single-step combustion.

    Expected scalars in state: 'Y_fuel', 'Y_ox', 'Y_prod', 'T'

    Returns
    -------
    rhs_fn : (scalars, vel, state, nu_base) -> dict of RHS arrays
    init_scalars : dict of initial fields
    """
    A = config.get('A', 1e8)
    Ea_over_R = config.get('Ea', 1e4)
    Q = config.get('Q', 5.0)
    nu_F = config.get('nu_F', 1.0)
    nu_O = config.get('nu_O', 4.0)
    D = config.get('D', 0.001)
    a_order = config.get('a', 1.0)
    b_order = config.get('b', 1.0)

    def _laplacian(field):
        return laplacian_cell_centered(field, grid)

    def rhs_fn(scalars, vel, state, nu_base_val):
        Y_fuel = jnp.maximum(scalars.get('Y_fuel', jnp.zeros(grid.cell_shape())), 0.0)
        Y_ox = jnp.maximum(scalars.get('Y_ox', jnp.zeros(grid.cell_shape())), 0.0)
        Y_prod = scalars.get('Y_prod', jnp.zeros(grid.cell_shape()))
        T = jnp.maximum(scalars.get('T', jnp.ones(grid.cell_shape())), 0.01)

        # Reaction rate
        omega = arrhenius_rate(T, Y_fuel, Y_ox, A, Ea_over_R, a_order, b_order)

        # Advection (central scheme for simplicity)
        adv_Yf = advect_scalar(Y_fuel, vel, grid)
        adv_Yo = advect_scalar(Y_ox, vel, grid)
        adv_Yp = advect_scalar(Y_prod, vel, grid)
        adv_T = advect_scalar(T, vel, grid)

        # Diffusion
        diff_Yf = D * _laplacian(Y_fuel)
        diff_Yo = D * _laplacian(Y_ox)
        diff_Yp = D * _laplacian(Y_prod)
        alpha = nu_base_val / 0.71  # Pr ≈ 0.71
        diff_T = alpha * _laplacian(T)

        result = {}
        # Species equations
        result['Y_fuel'] = -adv_Yf + diff_Yf - nu_F * omega
        result['Y_ox'] = -adv_Yo + diff_Yo - nu_O * omega
        result['Y_prod'] = -adv_Yp + diff_Yp + (nu_F + nu_O) * omega

        # Energy equation with heat release
        result['T'] = -adv_T + diff_T + Q * omega

        # Pass through other scalars
        for name, val in scalars.items():
            if name not in result:
                result[name] = jnp.zeros_like(val)

        return result

    # Initial fields (fuel on left, oxidiser on right)
    init_scalars = {
        'Y_fuel': jnp.zeros(grid.cell_shape()),
        'Y_ox': jnp.zeros(grid.cell_shape()),
        'Y_prod': jnp.zeros(grid.cell_shape()),
    }

    return rhs_fn, init_scalars
