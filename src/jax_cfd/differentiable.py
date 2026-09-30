"""
Differentiable simulation utilities for JAX-CFD.

This module re-exports the core functions from ``jax_cfd.optim`` for
backward compatibility.  New code should import from ``jax_cfd.optim``
directly.

Example — sensitivity of drag to Reynolds number::

    from jax_cfd.optim import differentiable_simulate

    def loss(nu):
        state = differentiable_simulate(
            step_fn, state0, n_steps=100, dt=0.005,
            nu_override=nu,
        )
        return pressure_cd(state.pressure)

    d_Cd_d_nu = jax.grad(loss)(0.005)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_cfd.core.state import FluidState

# Re-export from optim module
from jax_cfd.optim.simulate import (
    differentiable_simulate,
    differentiable_simulate_segmented,
    steady_state_solve,
)
from jax_cfd.optim.parameterize import (
    brinkman_alpha,
    make_kappa_field,
    smooth_max,
    density_filter,
    sobolev_filter,
)
from jax_cfd.optim.optimizers import (
    topology_optimize as _topology_optimize_new,
    multi_resolution_optimize,
    mma_update,
    adam_init,
    adam_update,
)
from jax_cfd.optim.continuation import (
    make_continuation_schedule,
    ContinuationState,
)


# ══════════════════════════════════════════════════════════════════════
#  Shape Optimisation (kept here — not part of topology optim)
# ══════════════════════════════════════════════════════════════════════

def shape_optimize(
    loss_fn,
    params_init: dict,
    n_steps: int = 50,
    lr: float = 1e-3,
    clip_fn=None,
    verbose: bool = True,
):
    """Gradient-descent shape optimisation loop.

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(params) -> scalar``.  Must be differentiable via
        ``jax.value_and_grad``.
    params_init : dict
        Initial parameter dict, e.g. ``{'radius': jnp.float32(0.35)}``.
    n_steps : int
        Number of gradient-descent iterations.
    lr : float
        Learning rate (constant).
    clip_fn : callable, optional
        ``clip_fn(params) -> params`` applied after each update.
    verbose : bool
        Print loss and params every 5 steps.

    Returns
    -------
    params : dict, history : list of (loss, params) tuples.
    """
    params = jax.tree.map(lambda x: x, params_init)
    history = []
    val_grad_fn = jax.value_and_grad(loss_fn)

    for i in range(n_steps):
        loss_val, grads = val_grad_fn(params)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        if clip_fn is not None:
            params = clip_fn(params)
        loss_f = float(loss_val)
        history.append((loss_f, jax.tree.map(float, params)))
        if verbose and (i % 5 == 0 or i == n_steps - 1):
            p_str = ', '.join(f'{k}={float(v):.6f}' for k, v in params.items())
            print(f'  step {i:3d}  loss={loss_f:.6f}  {p_str}')

    return params, history


# ══════════════════════════════════════════════════════════════════════
#  vmap Ensembles
# ══════════════════════════════════════════════════════════════════════

def vmap_re_sweep(
    step_fn,
    state0: FluidState,
    n_steps: int,
    dt: float,
    re_values,
    objective_fn=None,
    t0: float = 0.0,
):
    """Run simulations at multiple Re in parallel via ``jax.vmap``."""
    nu_values = 1.0 / jnp.asarray(re_values, dtype=jnp.float32)

    def run_single(nu):
        state_f = differentiable_simulate(
            step_fn, state0, n_steps, dt, t0=t0, nu_override=nu,
        )
        if objective_fn is not None:
            return objective_fn(state_f)
        return state_f

    return jax.vmap(run_single)(nu_values)


def vmap_ic_sweep(
    step_fn,
    states0_batch: FluidState,
    n_steps: int,
    dt: float,
    objective_fn=None,
    t0: float = 0.0,
    **step_kwargs,
):
    """Run simulations from multiple initial conditions via ``jax.vmap``."""
    def run_single(s0):
        s_f = differentiable_simulate(
            step_fn, s0, n_steps, dt, t0=t0, **step_kwargs,
        )
        if objective_fn is not None:
            return objective_fn(s_f)
        return s_f

    return jax.vmap(run_single)(states0_batch)


# ══════════════════════════════════════════════════════════════════════
#  Backward-compatible topology_optimize (old API)
# ══════════════════════════════════════════════════════════════════════

def topology_optimize(
    loss_fn,
    rho_init: jnp.ndarray,
    n_steps: int = 100,
    lr: float = 0.01,
    vol_frac: float = 0.5,
    vol_penalty: float = 10.0,
    filter_radius: float = 0.0,
    grid=None,
    verbose: bool = True,
):
    """Projected gradient descent for topology optimisation (legacy API).

    For the new API with MMA, Sobolev, and continuation support, use
    ``jax_cfd.optim.topology_optimize``.
    """
    rho = jax.tree.map(lambda x: x, rho_init)

    def augmented_loss(rho_raw):
        if filter_radius > 0.0 and grid is not None:
            rho_f = density_filter(rho_raw, filter_radius, grid)
        else:
            rho_f = rho_raw
        rho_f = jnp.clip(rho_f, 0.0, 1.0)
        J = loss_fn(rho_f)
        vol = jnp.mean(rho_f)
        J = J + vol_penalty * jnp.maximum(vol - vol_frac, 0.0) ** 2
        return J

    val_grad_fn = jax.value_and_grad(augmented_loss)
    history = []

    for i in range(n_steps):
        loss_val, grads = val_grad_fn(rho)
        rho = rho - lr * grads
        rho = jnp.clip(rho, 0.0, 1.0)
        vol_actual = float(jnp.mean(rho))
        loss_f = float(loss_val)
        history.append((loss_f, vol_actual))
        if verbose and (i % 5 == 0 or i == n_steps - 1):
            print(f'  step {i:3d}  loss={loss_f:.6f}  vol_frac={vol_actual:.4f}')

    return rho, history
