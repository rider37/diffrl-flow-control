"""
Gradient-based actuator placement optimization.

Optimizes jet actuator position, direction, and amplitude to minimize
a flow objective (e.g. drag) using end-to-end automatic differentiation
through the differentiable CFD solver.

Usage
-----
>>> from jax_cfd.optim.actuator_optimize import actuator_optimize
>>> params_opt, history = actuator_optimize(
...     loss_fn=loss,
...     params_init={'thetas': jnp.array([1.0, -1.0]), ...},
...     n_iters=100,
...     lr=0.01,
... )
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Optional, Tuple, List


def actuator_optimize(
    loss_fn: Callable,
    params_init: Dict[str, jnp.ndarray],
    n_iters: int = 100,
    lr: float = 0.01,
    clip_fn: Optional[Callable] = None,
    grad_clip: float = 1.0,
    verbose: bool = True,
    print_every: int = 5,
) -> Tuple[Dict[str, jnp.ndarray], List[dict]]:
    """Gradient-based actuator placement optimization using Adam.

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(params) -> scalar``.  Must be JAX-differentiable.
    params_init : dict of jnp arrays
        Initial parameters. Supported keys:
        - 'thetas': (N,) surface angles
        - 'offsets': (N,) direction offsets from surface normal
        - 'amplitude_raw': (N,) raw amplitude (softplus applied internally)
        - 'positions': (N, 2) free positions
        - 'directions': (N,) angles
        - 'amplitudes': (N,) force magnitudes
    n_iters : int
        Number of optimization iterations.
    lr : float
        Adam learning rate.
    clip_fn : callable, optional
        ``clip_fn(params) -> params`` applied after each update.
    grad_clip : float
        Maximum gradient norm (per-parameter clipping).
    verbose : bool
        Print progress.
    print_every : int
        Print frequency.

    Returns
    -------
    params_opt : dict
        Optimized parameters.
    history : list of dicts
        Per-iteration records: {'iter', 'loss', 'params', 'grad_norm'}.
    """
    # Initialize Adam state for each parameter
    params = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32), params_init)
    adam_m = jax.tree.map(jnp.zeros_like, params)
    adam_v = jax.tree.map(jnp.zeros_like, params)
    adam_t = 0

    val_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    history = []

    for i in range(n_iters):
        loss_val, grads = val_grad_fn(params)

        # Gradient clipping (per-parameter)
        if grad_clip > 0:
            grads = jax.tree.map(
                lambda g: jnp.clip(g, -grad_clip, grad_clip), grads
            )

        # Compute gradient norm for monitoring
        grad_norm = jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree.leaves(grads)
        ))

        # Adam update
        adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        adam_m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, adam_m, grads)
        adam_v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2, adam_v, grads)

        m_hat = jax.tree.map(lambda m: m / (1 - b1 ** adam_t), adam_m)
        v_hat = jax.tree.map(lambda v: v / (1 - b2 ** adam_t), adam_v)

        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
            params, m_hat, v_hat,
        )

        # Apply constraints
        if clip_fn is not None:
            params = clip_fn(params)

        loss_f = float(loss_val)
        grad_f = float(grad_norm)
        history.append({
            'iter': i,
            'loss': loss_f,
            'grad_norm': grad_f,
            'params': jax.tree.map(lambda x: x.copy(), params),
        })

        if verbose and (i % print_every == 0 or i == n_iters - 1):
            param_str = _format_params(params)
            print(f'  iter {i:4d}  loss={loss_f:.6f}  |g|={grad_f:.4f}  {param_str}')

    return params, history


def make_clip_fn(theta_range=(0, 2 * jnp.pi),
                 amplitude_range=(0.0, 10.0)):
    """Create a parameter clipping function for cylinder surface actuators.

    Parameters
    ----------
    theta_range : tuple — (min, max) for surface angles.
    amplitude_range : tuple — (min, max) for amplitudes.
    """
    def clip_fn(params):
        out = dict(params)
        if 'thetas' in out:
            # Wrap angles to [0, 2*pi)
            out['thetas'] = out['thetas'] % (2.0 * jnp.pi)
        if 'amplitudes' in out:
            out['amplitudes'] = jnp.clip(
                out['amplitudes'], amplitude_range[0], amplitude_range[1]
            )
        if 'amplitude_raw' in out:
            # Softplus applied internally; just clip raw values
            out['amplitude_raw'] = jnp.clip(out['amplitude_raw'], -5.0, 5.0)
        return out
    return clip_fn


def ensemble_grad(loss_fn, params, states_batch, n_ensemble=8):
    """Compute ensemble-averaged gradient over multiple initial conditions.

    Reduces gradient variance in chaotic flows.

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(params, state0) -> scalar``
    params : dict
    states_batch : list of FluidState — different initial conditions.
    n_ensemble : int — number of samples.

    Returns
    -------
    grad_mean : dict — averaged gradient.
    grad_std : float — gradient standard deviation (for monitoring).
    """
    def single_grad(state0):
        return jax.grad(lambda p: loss_fn(p, state0))(params)

    # vmap over initial conditions
    grads = jax.vmap(single_grad)(states_batch)
    grad_mean = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)
    grad_var = sum(
        jnp.sum(jnp.var(g, axis=0))
        for g in jax.tree.leaves(grads)
    )
    return grad_mean, float(jnp.sqrt(grad_var))


def _format_params(params, max_items=4):
    """Format parameters for concise printing."""
    parts = []
    for key, val in sorted(params.items()):
        if val.ndim == 0:
            parts.append(f'{key}={float(val):.4f}')
        elif val.size <= max_items:
            vals = ', '.join(f'{float(v):.3f}' for v in val.ravel())
            parts.append(f'{key}=[{vals}]')
        else:
            parts.append(f'{key}=({val.shape})')
    return '  '.join(parts)
