"""
Multi-Condition Multi-Objective (MCMO) optimization utilities.

Provides Chebyshev scalarization, weight vector sampling, Pareto front
extraction, and lightweight MLP for policy-based optimization.

Upgrades Kim, Kim & You (JCP 2022) by replacing black-box XFOIL with
a differentiable NS solver, enabling exact policy gradients.

Usage
-----
>>> from jax_cfd.optim.mcmo import chebyshev_scalarization, sample_weight_vectors
>>> w = sample_weight_vectors(11, n_obj=2)
>>> loss = chebyshev_scalarization(objectives, w[0], utopia)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, List, Tuple, Optional, Any


# ══════════════════════════════════════════════════════════════════════
#  Chebyshev scalarization
# ══════════════════════════════════════════════════════════════════════

def chebyshev_scalarization(
    objectives: jnp.ndarray,
    weights: jnp.ndarray,
    utopia: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted Chebyshev scalarization for multi-objective optimization.

    J = max_i { w_i * |f_i - f_i*| }

    Guarantees finding all Pareto-optimal solutions (convex and non-convex).

    Parameters
    ----------
    objectives : (n_obj,) array — current objective values f.
    weights : (n_obj,) array — weight vector (|w|=1, w_i >= 0).
    utopia : (n_obj,) array — utopia point f* (best-known per objective).

    Returns
    -------
    scalar — scalarized loss value.
    """
    return jnp.max(weights * jnp.abs(objectives - utopia))


def chebyshev_augmented(
    objectives: jnp.ndarray,
    weights: jnp.ndarray,
    utopia: jnp.ndarray,
    rho: float = 0.05,
) -> jnp.ndarray:
    """Augmented Chebyshev scalarization with weighted-sum tie-breaker.

    J = max_i { w_i * |f_i - f_i*| } + rho * sum_i { w_i * |f_i - f_i*| }

    The augmented term breaks ties among solutions with identical
    Chebyshev values, ensuring unique solutions.

    Parameters
    ----------
    objectives, weights, utopia : same as ``chebyshev_scalarization``.
    rho : float — augmentation coefficient (small, e.g. 0.05).
    """
    diff = weights * jnp.abs(objectives - utopia)
    return jnp.max(diff) + rho * jnp.sum(diff)


# ══════════════════════════════════════════════════════════════════════
#  Weight vector sampling
# ══════════════════════════════════════════════════════════════════════

def sample_weight_vectors(n_weights: int, n_obj: int = 2) -> jnp.ndarray:
    """Generate uniformly spaced weight vectors on the (n_obj-1) simplex.

    For n_obj=2, produces n_weights vectors in [0,1]^2 with w1+w2=1.

    Parameters
    ----------
    n_weights : int — number of weight vectors.
    n_obj : int — number of objectives (default 2).

    Returns
    -------
    weights : (n_weights, n_obj) array — each row sums to 1.
    """
    if n_obj == 2:
        w1 = jnp.linspace(0.01, 0.99, n_weights)
        return jnp.stack([w1, 1.0 - w1], axis=1)
    elif n_obj == 3:
        # Grid on simplex for 3 objectives
        pts = []
        h = n_weights - 1
        for i in range(n_weights):
            for j in range(n_weights - i):
                k = h - i - j
                pts.append([i / h, j / h, k / h])
        pts = jnp.array(pts)
        # Avoid exact zeros
        pts = jnp.clip(pts, 0.01, None)
        return pts / pts.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"n_obj={n_obj} not supported; use 2 or 3.")


def sample_weight_random(key, n_samples: int, n_obj: int = 2) -> jnp.ndarray:
    """Sample random weight vectors from Dirichlet distribution.

    Parameters
    ----------
    key : JAX random key.
    n_samples : int — number of weight vectors.
    n_obj : int — number of objectives.

    Returns
    -------
    weights : (n_samples, n_obj) array — each row sums to 1.
    """
    alpha = jnp.ones(n_obj)
    samples = jr.dirichlet(key, alpha, shape=(n_samples,))
    return jnp.clip(samples, 0.01, None)


# ══════════════════════════════════════════════════════════════════════
#  Utopia point tracking
# ══════════════════════════════════════════════════════════════════════

def update_utopia(
    utopia: jnp.ndarray,
    objectives: jnp.ndarray,
) -> jnp.ndarray:
    """Update utopia point with element-wise minimum.

    Parameters
    ----------
    utopia : (n_obj,) array — current best-known per objective.
    objectives : (n_obj,) array — new objective values.

    Returns
    -------
    updated utopia : (n_obj,) array.
    """
    return jnp.minimum(utopia, objectives)


# ══════════════════════════════════════════════════════════════════════
#  Pareto front utilities
# ══════════════════════════════════════════════════════════════════════

def is_dominated(a: jnp.ndarray, b: jnp.ndarray) -> bool:
    """Check if solution a is dominated by solution b (all objectives minimized).

    b dominates a if b_i <= a_i for all i and b_j < a_j for at least one j.
    """
    return bool(jnp.all(b <= a) and jnp.any(b < a))


def extract_pareto_front(
    objectives_list: List[jnp.ndarray],
) -> List[int]:
    """Extract indices of non-dominated solutions.

    Parameters
    ----------
    objectives_list : list of (n_obj,) arrays.

    Returns
    -------
    indices : list of int — indices of Pareto-optimal solutions.
    """
    n = len(objectives_list)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if is_dominated(objectives_list[i], objectives_list[j]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def hypervolume_2d(
    pareto_points: jnp.ndarray,
    ref_point: jnp.ndarray,
) -> float:
    """Compute 2D hypervolume indicator.

    Parameters
    ----------
    pareto_points : (N, 2) array — Pareto front points (minimization).
    ref_point : (2,) array — reference point (must dominate all points).

    Returns
    -------
    hv : float — hypervolume.
    """
    pts = jnp.array(pareto_points)
    # Sort by first objective
    order = jnp.argsort(pts[:, 0])
    pts = pts[order]

    hv = 0.0
    ref_x = float(ref_point[0])
    prev_y = float(ref_point[1])
    for i in range(len(pts)):
        x_i = float(pts[i, 0])
        y_i = float(pts[i, 1])
        if y_i < prev_y:
            hv += (ref_x - x_i) * (prev_y - y_i)
            prev_y = y_i
    return hv


# ══════════════════════════════════════════════════════════════════════
#  Lightweight MLP (manual params, no framework)
# ══════════════════════════════════════════════════════════════════════

def init_mlp_params(
    layer_dims: List[int],
    key,
) -> Dict[str, List[jnp.ndarray]]:
    """Initialize MLP parameters with Kaiming uniform.

    Parameters
    ----------
    layer_dims : list of int — [input_dim, hidden1, ..., output_dim].
    key : JAX random key.

    Returns
    -------
    params : dict with 'W' and 'b' lists.
    """
    W_list, b_list = [], []
    for i in range(len(layer_dims) - 1):
        key, subkey = jr.split(key)
        fan_in = layer_dims[i]
        fan_out = layer_dims[i + 1]
        scale = jnp.sqrt(2.0 / fan_in)
        W_list.append(jr.normal(subkey, (fan_in, fan_out)) * scale)
        b_list.append(jnp.zeros(fan_out))
    return {'W': W_list, 'b': b_list}


def mlp_forward(
    params: Dict[str, List[jnp.ndarray]],
    x: jnp.ndarray,
) -> jnp.ndarray:
    """MLP forward pass with ReLU activations (no output activation).

    Parameters
    ----------
    params : dict with 'W' and 'b' lists.
    x : (input_dim,) or (batch, input_dim) array.

    Returns
    -------
    output : (output_dim,) or (batch, output_dim) array.
    """
    n_layers = len(params['W'])
    for i in range(n_layers):
        x = x @ params['W'][i] + params['b'][i]
        if i < n_layers - 1:
            x = jax.nn.relu(x)
    return x


# ══════════════════════════════════════════════════════════════════════
#  MCMO optimization loop
# ══════════════════════════════════════════════════════════════════════

def mcmo_optimize_direct(
    loss_fn,
    x_init: jnp.ndarray,
    weight_vectors: jnp.ndarray,
    n_iters: int = 80,
    lr: float = 0.01,
    clip_fn=None,
    grad_clip: float = 1.0,
    verbose: bool = True,
    print_every: int = 10,
) -> Tuple[List[jnp.ndarray], List[dict]]:
    """Direct gradient MCMO: optimize design vars for each weight vector.

    For each weight vector w, runs Adam to minimize
    ``loss_fn(x, w) -> scalar`` (Chebyshev-scalarized loss).

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(x, w) -> scalar``. Must be JAX-differentiable w.r.t. x.
    x_init : (n_vars,) array — initial design variables.
    weight_vectors : (n_weights, n_obj) array.
    n_iters : int — iterations per weight vector.
    lr : float — Adam learning rate.
    clip_fn : callable, optional — ``clip_fn(x) -> x``.
    grad_clip : float — max gradient norm.
    verbose, print_every : printing control.

    Returns
    -------
    solutions : list of (n_vars,) arrays — optimized x per weight.
    history : list of dicts with optimization records.
    """
    solutions = []
    history = []
    val_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for wi, w in enumerate(weight_vectors):
        x = jnp.array(x_init, dtype=jnp.float32)
        # Adam state
        m = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8

        w_records = []
        for i in range(n_iters):
            loss_val, grads = val_grad_fn(x, w)

            if grad_clip > 0:
                grads = jnp.clip(grads, -grad_clip, grad_clip)

            t += 1
            m = b1 * m + (1 - b1) * grads
            v = b2 * v + (1 - b2) * grads ** 2
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
            x = x - lr * m_hat / (jnp.sqrt(v_hat) + eps)

            if clip_fn is not None:
                x = clip_fn(x)

            w_records.append({
                'weight_idx': wi, 'iter': i,
                'loss': float(loss_val),
                'x': x.copy(),
            })

            if verbose and (i % print_every == 0 or i == n_iters - 1):
                print(f'  w[{wi:2d}] iter {i:3d}  loss={float(loss_val):.6f}')

        solutions.append(x)
        history.extend(w_records)

    return solutions, history
