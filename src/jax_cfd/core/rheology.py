"""Non-Newtonian rheology models.

Provides effective viscosity as a function of the local strain-rate magnitude
|γ̇| = √(2 S_ij S_ij), where S is the symmetric part of the velocity gradient.

Supported models:
  - newtonian:  constant ν (identity — no change)
  - power_law:  ν = K |γ̇|^(n-1)        (Ostwald-de Waele)
  - carreau:    ν = ν_∞ + (ν_0 - ν_∞)(1 + (λ|γ̇|)²)^((n-1)/2)

Usage (via runner CONFIG):
    'rheology': {
        'model': 'power_law',
        'K': 0.1,       # consistency index
        'n': 0.5,       # flow behaviour index (n<1: shear-thinning)
    }
"""
from __future__ import annotations

from typing import Callable
import jax.numpy as jnp


# Type alias: fn(strain_rate_magnitude) -> effective_viscosity (cell-centred)
RheologyFn = Callable[[jnp.ndarray], jnp.ndarray]


def newtonian(nu: float) -> RheologyFn:
    """Constant (Newtonian) viscosity — identity wrapper."""
    def _fn(gamma_dot: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(gamma_dot, nu)
    return _fn


def power_law(K: float, n: float, gamma_min: float = 1e-8) -> RheologyFn:
    """Ostwald–de Waele power-law model.

    ν(|γ̇|) = K |γ̇|^(n-1)

    Parameters
    ----------
    K : consistency index (Pa·s^n for dimensional, or non-dim)
    n : flow behaviour index.  n < 1 → shear-thinning,  n > 1 → shear-thickening
    gamma_min : floor on |γ̇| to avoid singularity at n < 1
    """
    def _fn(gamma_dot: jnp.ndarray) -> jnp.ndarray:
        gd = jnp.maximum(gamma_dot, gamma_min)
        return K * gd ** (n - 1.0)
    return _fn


def carreau(nu_0: float, nu_inf: float, lam: float, n: float,
            gamma_min: float = 1e-8) -> RheologyFn:
    """Carreau–Yasuda shear-thinning model.

    ν(|γ̇|) = ν_∞ + (ν_0 - ν_∞) (1 + (λ |γ̇|)²)^((n-1)/2)

    Parameters
    ----------
    nu_0   : zero-shear-rate viscosity
    nu_inf : infinite-shear-rate viscosity
    lam    : time constant (relaxation time)
    n      : power-law index
    """
    def _fn(gamma_dot: jnp.ndarray) -> jnp.ndarray:
        gd = jnp.maximum(gamma_dot, gamma_min)
        return nu_inf + (nu_0 - nu_inf) * (1.0 + (lam * gd) ** 2) ** ((n - 1.0) / 2.0)
    return _fn


def get_rheology_model(name: str, **params) -> RheologyFn:
    """Factory: return a RheologyFn by name.

    >>> rh = get_rheology_model('power_law', K=0.1, n=0.5)
    >>> nu_eff = rh(strain_rate_magnitude)
    """
    if name == 'newtonian':
        return newtonian(params.get('nu', 1.0))
    elif name == 'power_law':
        return power_law(K=params['K'], n=params['n'],
                         gamma_min=params.get('gamma_min', 1e-8))
    elif name == 'carreau':
        return carreau(nu_0=params['nu_0'], nu_inf=params['nu_inf'],
                       lam=params['lam'], n=params['n'],
                       gamma_min=params.get('gamma_min', 1e-8))
    else:
        raise ValueError(f"Unknown rheology model: {name!r}.  "
                         f"Choose from: newtonian, power_law, carreau")
