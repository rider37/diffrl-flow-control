"""
Optimisation algorithms for topology optimisation.

Provides:
- ``mma_update``: Method of Moving Asymptotes (MMA)        [Method 2]
- ``adam_init / adam_update``: Adam optimizer
- ``topology_optimize``: Unified loop with all 5 methods
- ``multi_resolution_optimize``: Coarse-to-fine cascade    [Method 1]
"""

from __future__ import annotations

from typing import Callable, List, Optional

import jax
import jax.numpy as jnp

from jax_cfd.optim.parameterize import sobolev_filter, upscale_field
from jax_cfd.optim.continuation import ContinuationState


# ══════════════════════════════════════════════════════════════════════
#  Adam optimizer
# ══════════════════════════════════════════════════════════════════════

def adam_init(params):
    """Initialize Adam state (m, v, t)."""
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    return m, v, 0


def adam_update(grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    """Single Adam update. Returns (delta, m_new, v_new, t_new)."""
    t = t + 1
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads ** 2
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    delta = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return delta, m, v, t


# ══════════════════════════════════════════════════════════════════════
#  MMA (Method of Moving Asymptotes)                        [Method 2]
# ══════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class MMAState:
    """Immutable state for the MMA optimizer (no in-place mutation)."""
    x_1: jnp.ndarray     # x at iteration k-1 (flattened)
    x_2: jnp.ndarray     # x at iteration k-2 (flattened)
    low: jnp.ndarray     # lower asymptotes
    upp: jnp.ndarray     # upper asymptotes
    n: int
    x_min: float
    x_max: float
    iter: int


def mma_init(x: jnp.ndarray, x_min: float = 0.0, x_max: float = 1.0) -> MMAState:
    """Initialize MMA state."""
    n = x.size
    x_flat = x.ravel()
    return MMAState(
        x_1=x_flat, x_2=x_flat,
        low=x_min * jnp.ones(n), upp=x_max * jnp.ones(n),
        n=n, x_min=x_min, x_max=x_max, iter=0,
    )


def mma_update(
    x: jnp.ndarray,
    grad: jnp.ndarray,
    mma_state: MMAState,
    constraint_vals: Optional[jnp.ndarray] = None,
    constraint_grads: Optional[jnp.ndarray] = None,
    move_limit: float = 0.2,
) -> tuple:
    """One MMA update step (Svanberg 1987, simplified).

    Returns a *new* MMAState (no mutation of the input).

    Parameters
    ----------
    x : jnp.ndarray
        Current design variables (any shape, will be flattened).
    grad : jnp.ndarray
        Gradient of the objective (same shape as x).
    mma_state : MMAState
        Optimizer state from previous iteration.
    constraint_vals : jnp.ndarray (m,), optional
        Constraint function values g_j(x).  Feasible when g_j <= 0.
    constraint_grads : jnp.ndarray (m, n), optional
        Jacobian of constraints w.r.t. flattened x.
    move_limit : float
        Maximum move per iteration as fraction of (x_max - x_min).

    Returns
    -------
    x_new : jnp.ndarray (same shape as input x)
    mma_state_new : MMAState (new, immutable)
    """
    orig_shape = x.shape
    x_flat = x.ravel()
    g_flat = grad.ravel()
    s = mma_state
    x_min, x_max = s.x_min, s.x_max
    dx_range = x_max - x_min

    # Asymptote update (Svanberg's heuristic)
    # For iter < 2: initialise at ±0.5 range
    # For iter >= 2: oscillation detection adjusts asymptotes
    osc = (x_flat - s.x_1) * (s.x_1 - s.x_2)
    gamma = jnp.where(osc < 0, 0.7, 1.2)

    low_osc = x_flat - gamma * jnp.abs(s.x_1 - s.low)
    upp_osc = x_flat + gamma * jnp.abs(s.upp - s.x_1)
    low_init = x_flat - 0.5 * dx_range
    upp_init = x_flat + 0.5 * dx_range

    # Select based on iteration count (pure array ops, no Python if)
    use_init = s.iter < 2
    low = jnp.where(use_init, low_init, low_osc)
    upp = jnp.where(use_init, upp_init, upp_osc)

    # Clamp asymptotes to sensible range
    low = jnp.maximum(low, x_flat - 10.0 * dx_range)
    low = jnp.minimum(low, x_flat - 0.01 * dx_range)
    upp = jnp.maximum(upp, x_flat + 0.01 * dx_range)
    upp = jnp.minimum(upp, x_flat + 10.0 * dx_range)

    # Move limits
    ml = move_limit * dx_range
    alpha_mma = jnp.maximum(x_min, x_flat - ml)
    beta_mma = jnp.minimum(x_max, x_flat + ml)

    # MMA subproblem: separable convex approximation
    p = jnp.maximum(g_flat, 0.0) * (upp - x_flat) ** 2
    q = jnp.maximum(-g_flat, 0.0) * (x_flat - low) ** 2

    # Closed-form solution for box-constrained separable subproblem
    sp = jnp.sqrt(p + 1e-12)
    sq = jnp.sqrt(q + 1e-12)
    x_new = (sp * low + sq * upp) / (sp + sq + 1e-30)

    # Apply move limits and box constraints
    x_new = jnp.clip(x_new, alpha_mma, beta_mma)
    x_new = jnp.clip(x_new, x_min, x_max)

    # Handle constraints via iterative dual update (if provided)
    # Solves the Lagrangian dual: max_lam min_x  L(x, lam)
    # with L = f_approx(x) + sum_j lam_j * g_j(x), lam_j >= 0
    if constraint_vals is not None and constraint_grads is not None:
        c_grads = constraint_grads  # (m, n)
        n_dual_iters = 5
        lam_dual = jnp.zeros(constraint_vals.shape[0])

        for _ in range(n_dual_iters):
            # Primal update: x_new given current dual variables
            # Add constraint gradient contribution to objective gradient
            g_aug = g_flat + jnp.sum(lam_dual[:, None] * c_grads, axis=0)
            p_aug = jnp.maximum(g_aug, 0.0) * (upp - x_flat) ** 2
            q_aug = jnp.maximum(-g_aug, 0.0) * (x_flat - low) ** 2
            sp_a = jnp.sqrt(p_aug + 1e-12)
            sq_a = jnp.sqrt(q_aug + 1e-12)
            x_new = (sp_a * low + sq_a * upp) / (sp_a + sq_a + 1e-30)
            x_new = jnp.clip(x_new, alpha_mma, beta_mma)
            x_new = jnp.clip(x_new, x_min, x_max)

            # Dual update: evaluate constraint at current x_new
            c_vals_curr = jnp.dot(c_grads, x_new) - jnp.dot(c_grads, x_flat) + constraint_vals
            # Subgradient step with adaptive step size
            dual_step = 1.0 / (jnp.sum(c_grads ** 2, axis=1) + 1e-10)
            lam_dual = jnp.maximum(0.0, lam_dual + dual_step * c_vals_curr)

    # Return new immutable state
    new_state = MMAState(
        x_1=x_flat, x_2=s.x_1,
        low=low, upp=upp,
        n=s.n, x_min=x_min, x_max=x_max,
        iter=s.iter + 1,
    )

    return x_new.reshape(orig_shape), new_state


# ══════════════════════════════════════════════════════════════════════
#  L-BFGS-B optimizer                                        [Method 6]
# ══════════════════════════════════════════════════════════════════════

def lbfgs_init(memory: int = 10) -> dict:
    """Initialize L-BFGS state (history stored in Python land)."""
    return {'s': [], 'y': [], 'memory': memory,
            'x_prev': None, 'g_prev': None}


def lbfgs_step(
    x: jnp.ndarray,
    grad: jnp.ndarray,
    state: dict,
) -> tuple:
    """L-BFGS search direction via two-loop recursion.

    Returns ``(direction, state)`` where ``x_new = x + lr * direction``.
    The direction includes inverse-Hessian scaling.
    """
    x_flat = x.ravel()
    g_flat = grad.ravel()

    s_list = state['s']
    y_list = state['y']

    # Update history with (s_k, y_k) from previous step
    if state['g_prev'] is not None:
        s_k = x_flat - state['x_prev']
        y_k = g_flat - state['g_prev']
        ys = float(jnp.dot(y_k, s_k))
        if ys > 1e-10:  # curvature condition
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > state['memory']:
                s_list.pop(0)
                y_list.pop(0)

    m = len(s_list)
    if m == 0:
        # Steepest descent, scaled to limit initial step
        g_inf = float(jnp.max(jnp.abs(g_flat)))
        gamma0 = min(1.0, 1.0 / (g_inf + 1e-30))
        d = -gamma0 * g_flat
    else:
        # Two-loop recursion
        q = g_flat.copy()
        alphas = [0.0] * m
        rhos = [0.0] * m
        for i in range(m - 1, -1, -1):
            rhos[i] = 1.0 / (float(jnp.dot(y_list[i], s_list[i])) + 1e-30)
            alphas[i] = rhos[i] * float(jnp.dot(s_list[i], q))
            q = q - alphas[i] * y_list[i]

        # H_0 = gamma * I (Nocedal & Wright scaling)
        gamma = float(jnp.dot(s_list[-1], y_list[-1])) / \
                (float(jnp.dot(y_list[-1], y_list[-1])) + 1e-30)
        r = gamma * q

        for i in range(m):
            beta_i = rhos[i] * float(jnp.dot(y_list[i], r))
            r = r + (alphas[i] - beta_i) * s_list[i]

        d = -r

    new_state = {'s': s_list, 'y': y_list, 'memory': state['memory'],
                 'x_prev': x_flat, 'g_prev': g_flat}
    return d.reshape(x.shape), new_state


# ══════════════════════════════════════════════════════════════════════
#  Unified topology_optimize with all methods
# ══════════════════════════════════════════════════════════════════════

def topology_optimize(
    loss_fn: Callable,
    rho_init: jnp.ndarray,
    grid,
    n_steps: int = 100,
    lr: float = 0.01,
    vol_frac: float = 0.5,
    vol_penalty: float = 10.0,
    optimizer: str = 'adam',
    sobolev_eps: float = 0.0,
    continuation: Optional[List[ContinuationState]] = None,
    alpha_max: float = 1e4,
    q: float = 0.01,
    grad_clip: float = 0.0,
    warm_start_state=None,
    verbose: bool = True,
) -> tuple:
    """Topology optimisation with optional Sobolev + MMA + continuation.

    Parameters
    ----------
    loss_fn : callable
        Without warm-start: ``loss_fn(rho, alpha_max, q) -> scalar``.
        With warm-start: ``loss_fn(rho, alpha_max, q, state_hint)
        -> (scalar, state_out)``.
    rho_init : jnp.ndarray
        Initial design field, values in [0, 1].
    grid : MACGrid
        Grid object (used for Sobolev filter).
    n_steps : int
        Optimisation iterations (used if no continuation schedule).
    lr : float
        Learning rate (used if no continuation schedule).
    vol_frac : float
        Target volume fraction.
    vol_penalty : float
        Quadratic penalty weight for volume constraint.
    optimizer : str
        'adam', 'gd', 'mma', or 'lbfgs'.
    sobolev_eps : float
        Sobolev smoothing scale. 0 = no smoothing.
    continuation : list of ContinuationState, optional
        Multi-stage continuation schedule. If provided, overrides
        n_steps, lr, alpha_max, q, sobolev_eps per stage.
    alpha_max, q : float
        Default RAMP parameters (overridden by continuation).
    grad_clip : float
        If > 0, clip gradient norm to this value.
    warm_start_state : FluidState, optional
        If provided, enables state warm-starting.  The loss_fn must
        accept a 4th ``state_hint`` argument and return ``(loss, state)``.
        The state is carried between iterations, eliminating redundant
        warmup simulation.
    verbose : bool
        Print progress.

    Returns
    -------
    rho_opt : jnp.ndarray
    history : list of dict
    final_state : FluidState or None
        Final flow state (only when warm_start_state is provided).
    """
    # Build stages
    if continuation is not None:
        stages = continuation
    else:
        stages = [ContinuationState(
            stage=0, alpha_max=alpha_max, q=q,
            n_steps=n_steps, lr=lr, sobolev_eps=sobolev_eps,
        )]

    _warm = warm_start_state is not None
    state_hint = warm_start_state

    rho = rho_init.copy()
    history = []
    global_step = 0

    for stage in stages:
        _alpha = stage.alpha_max
        _q = stage.q
        _lr = stage.lr
        _eps = stage.sobolev_eps
        _n = stage.n_steps

        if verbose:
            tag = " [warm-start]" if _warm else ""
            print(f"\n  === Stage {stage.stage}: alpha_max={_alpha:.0f}, "
                  f"q={_q:.4f}, lr={_lr:.4f}, eps={_eps:.1f}, "
                  f"n_steps={_n}{tag} ===")

        # Pass alpha_max and q as traced JAX scalars so that
        # stage transitions don't trigger JIT re-compilation.
        _alpha_t = jnp.float32(_alpha)
        _q_t = jnp.float32(_q)

        if _warm:
            def augmented_loss_ws(rho_in, alpha_t, q_t, sh):
                J, s_out = loss_fn(rho_in, alpha_t, q_t, sh)
                vol = jnp.mean(rho_in)
                J = J + vol_penalty * (vol - vol_frac) ** 2
                return J, s_out
            val_grad_fn = jax.jit(jax.value_and_grad(
                augmented_loss_ws, argnums=0, has_aux=True))
        else:
            def augmented_loss(rho_in, alpha_t, q_t):
                J = loss_fn(rho_in, alpha_t, q_t)
                vol = jnp.mean(rho_in)
                J = J + vol_penalty * (vol - vol_frac) ** 2
                return J
            val_grad_fn = jax.jit(jax.value_and_grad(augmented_loss))

        # JIT-compiled forward-only eval for line search
        if _warm:
            _eval_fn = jax.jit(augmented_loss_ws)
        else:
            _eval_fn = jax.jit(augmented_loss)

        # Init optimizer for this stage (reset at each stage)
        if optimizer == 'adam':
            m, v, t_adam = adam_init(rho)
        elif optimizer == 'mma':
            mma_state = mma_init(rho)
        elif optimizer == 'lbfgs':
            lbfgs_st = lbfgs_init()

        for i in range(_n):
            if _warm:
                (loss_val, state_out), grads = val_grad_fn(
                    rho, _alpha_t, _q_t, state_hint)
                state_hint = state_out
            else:
                loss_val, grads = val_grad_fn(rho, _alpha_t, _q_t)

            # Sobolev gradient smoothing [Method 4]
            if _eps > 0.0 and grid is not None:
                dx_min = min(grid.spacings)
                eps_phys = _eps * dx_min
                grads = sobolev_filter(grads, eps_phys, grid)

            # Gradient clipping
            if grad_clip > 0.0:
                g_norm = jnp.linalg.norm(grads)
                grads = jnp.where(g_norm > grad_clip,
                                  grads * (grad_clip / g_norm), grads)

            # Update
            if optimizer == 'adam':
                delta, m, v, t_adam = adam_update(grads, m, v, t_adam, lr=_lr)
                rho = rho - delta
            elif optimizer == 'mma':
                vol = jnp.mean(rho)
                c_val = jnp.array([vol - vol_frac])
                c_grad = jnp.ones_like(rho).ravel() / rho.size
                c_grad = c_grad[jnp.newaxis, :]
                rho, mma_state = mma_update(
                    rho, grads, mma_state,
                    constraint_vals=c_val, constraint_grads=c_grad,
                    move_limit=0.15,
                )
            elif optimizer == 'lbfgs':
                d, lbfgs_st = lbfgs_step(rho, grads, lbfgs_st)
                # Armijo backtracking line search (start at alpha=1.0)
                alpha_ls = 1.0
                f0 = float(loss_val)
                slope = float(jnp.sum(grads * d))
                c1 = 1e-4  # Armijo constant
                for _bt in range(6):
                    rho_try = jnp.clip(rho + alpha_ls * d, 0.01, 0.99)
                    if _warm:
                        f_try, _ = _eval_fn(rho_try, _alpha_t, _q_t, state_hint)
                    else:
                        f_try = _eval_fn(rho_try, _alpha_t, _q_t)
                    f_try = float(f_try)
                    if f_try <= f0 + c1 * alpha_ls * slope:
                        break
                    alpha_ls *= 0.5
                rho = jnp.clip(rho + alpha_ls * d, 0.01, 0.99)
            else:  # gd
                rho = rho - _lr * grads

            rho = jnp.clip(rho, 0.01, 0.99)

            vol_actual = float(jnp.mean(rho))
            loss_f = float(loss_val)
            history.append({
                'step': global_step, 'stage': stage.stage,
                'loss': loss_f, 'vol_frac': vol_actual,
                'alpha_max': _alpha, 'q': _q,
            })

            if verbose and (i % 5 == 0 or i == _n - 1):
                print(f"    step {global_step:3d}  loss={loss_f:.6f}  "
                      f"vol={vol_actual:.4f}")

            global_step += 1

    return rho, history, state_hint


# ══════════════════════════════════════════════════════════════════════
#  Multi-Resolution Cascade                                 [Method 1]
# ══════════════════════════════════════════════════════════════════════

def multi_resolution_optimize(
    build_loss_fn: Callable,
    grid_shapes: List[tuple],
    domain_size: tuple,
    build_grid_fn: Callable,
    vol_frac: float = 0.5,
    vol_penalty: float = 10.0,
    optimizer: str = 'adam',
    continuation: Optional[List[ContinuationState]] = None,
    alpha_max: float = 1e4,
    q: float = 0.01,
    steps_per_level: int = 30,
    lr: float = 0.01,
    sobolev_eps: float = 2.0,
    grad_clip: float = 0.0,
    dt_fine: float = None,
    nu: float = None,
    verbose: bool = True,
) -> tuple:
    """Coarse-to-fine multi-resolution topology optimisation.

    Starts on a coarse grid and progressively refines, upscaling the
    design field between levels.  At each level:
    - alpha_max is scaled by (dx_fine/dx_level)^2  (consistent Brinkman Da)
    - dt is adapted based on CFL: dt_level = dt_fine * (dx_level/dx_fine)
      when nu is provided (viscous CFL limit), otherwise dt stays constant.

    Parameters
    ----------
    build_loss_fn : callable
        ``build_loss_fn(grid, dt) -> loss_fn(rho, alpha_max, q) -> scalar``.
        Factory that builds a loss function for the given grid and dt.
    grid_shapes : list of tuple
        Grid shapes from coarse to fine, e.g. [(32, 8), (64, 16), (128, 32)].
    domain_size : tuple
        Physical domain size (Lx, Ly, ...).
    build_grid_fn : callable
        ``build_grid_fn(shape, domain_size) -> MACGrid``.
    vol_frac, vol_penalty, optimizer, sobolev_eps, grad_clip
        Passed to ``topology_optimize``.
    continuation : list of ContinuationState, optional
        If provided, used at the FINAL resolution level.
        Coarser levels use constant alpha_max/q with steps_per_level.
    alpha_max : float
        Base alpha_max at the finest resolution.
    q : float
        RAMP parameter.
    steps_per_level : int
        Optimisation steps at each resolution level.
    dt_fine : float, optional
        Time step at the finest resolution.  When provided with ``nu``,
        coarser levels get larger dt proportional to dx_level/dx_fine
        (maintaining the same viscous CFL number).
    nu : float, optional
        Kinematic viscosity.  Used with ``dt_fine`` for CFL-based dt
        adaptation.  Also halves n_steps at coarser levels since larger
        dt covers the same physical time in fewer steps.
    verbose : bool
        Print progress.

    Returns
    -------
    rho_fine : jnp.ndarray
        Optimised design at the finest resolution.
    history : list of dict
        Full optimisation history across all levels.
    """
    n_levels = len(grid_shapes)
    finest_shape = grid_shapes[-1]
    dx_fine = min(L / N for L, N in zip(domain_size, finest_shape))

    # Initialize at coarsest level
    rho = vol_frac * jnp.ones(grid_shapes[0])
    all_history = []

    for level, shape in enumerate(grid_shapes):
        grid = build_grid_fn(shape, domain_size)
        dx_level = min(grid.spacings)
        ratio = dx_level / dx_fine   # > 1 for coarser levels

        # Scale alpha_max by (dx_fine/dx_level)^2 to maintain
        # consistent Brinkman Da number across resolutions
        alpha_scaled = alpha_max / (ratio ** 2)

        # Adapt dt based on CFL when nu is provided
        dt_level = None
        if dt_fine is not None and nu is not None:
            # Viscous CFL: dt * nu / dx^2 = const
            # => dt_level = dt_fine * (dx_level / dx_fine)^2
            dt_level = dt_fine * ratio ** 2

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Multi-res level {level+1}/{n_levels}: {shape}")
            print(f"  dx={dx_level:.4f}, alpha_max={alpha_scaled:.0f}", end="")
            if dt_level is not None:
                print(f", dt={dt_level:.6f}", end="")
            print(f"\n{'='*60}")

        # Build loss function (may depend on dt for simulation-based losses)
        if dt_level is not None:
            loss_fn = build_loss_fn(grid, dt_level)
        else:
            loss_fn = build_loss_fn(grid)

        # Use continuation at the final level if provided
        if level == n_levels - 1 and continuation is not None:
            cont_scaled = []
            for cs in continuation:
                cont_scaled.append(ContinuationState(
                    stage=cs.stage,
                    alpha_max=cs.alpha_max / (ratio ** 2),
                    q=cs.q,
                    n_steps=cs.n_steps,
                    lr=cs.lr,
                    sobolev_eps=cs.sobolev_eps,
                ))
            rho, hist, _ = topology_optimize(
                loss_fn, rho, grid,
                vol_frac=vol_frac, vol_penalty=vol_penalty,
                optimizer=optimizer, continuation=cont_scaled,
                grad_clip=grad_clip, verbose=verbose,
            )
        else:
            rho, hist, _ = topology_optimize(
                loss_fn, rho, grid,
                n_steps=steps_per_level, lr=lr,
                vol_frac=vol_frac, vol_penalty=vol_penalty,
                optimizer=optimizer, sobolev_eps=sobolev_eps,
                alpha_max=alpha_scaled, q=q,
                grad_clip=grad_clip, verbose=verbose,
            )

        all_history.extend(hist)

        # Upscale to next level
        if level < n_levels - 1:
            next_shape = grid_shapes[level + 1]
            rho = upscale_field(rho, next_shape)
            rho = jnp.clip(rho, 0.01, 0.99)
            if verbose:
                print(f"  Upscaled {shape} -> {next_shape}")

    return rho, all_history
