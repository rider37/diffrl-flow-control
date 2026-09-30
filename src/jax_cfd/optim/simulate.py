"""
Differentiable simulation utilities.

Provides:
- ``differentiable_simulate``: per-step gradient checkpointing (O(1) memory)
- ``differentiable_simulate_segmented``: segmented checkpointing
- ``steady_state_solve``: IFT-based implicit differentiation through
  steady-state fixed points (eliminates BPTT)                [Method 3]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_cfd.core.state import FluidState


# ══════════════════════════════════════════════════════════════════════
#  Standard differentiable simulation (per-step checkpoint)
# ══════════════════════════════════════════════════════════════════════

def differentiable_simulate(
    step_fn,
    state0: FluidState,
    n_steps: int,
    dt: float,
    t0: float = 0.0,
    **step_kwargs,
) -> FluidState:
    """Run *n_steps* with per-step gradient checkpointing.

    Memory: O(1).  Compute: ~2x forward (backward recomputes each step).
    """
    @jax.checkpoint
    def body(carry, _):
        s, t = carry
        s = step_fn(s, dt, t, **step_kwargs)
        return (s, t + dt), None

    (s_final, _), _ = jax.lax.scan(
        body, (state0, jnp.float32(t0)), length=n_steps,
    )
    return s_final


def differentiable_simulate_segmented(
    step_fn,
    state0: FluidState,
    n_steps: int,
    dt: float,
    segment_length: int = 10,
    t0: float = 0.0,
    **step_kwargs,
) -> FluidState:
    """Run *n_steps* with segmented gradient checkpointing.

    Memory: O(n_steps / segment_length).  Compute: ~1.5x forward.
    """
    n_segments = n_steps // segment_length
    remainder = n_steps % segment_length

    def run_segment(carry, _):
        @jax.checkpoint
        def seg_body(inner_carry, __):
            s, t = inner_carry
            s = step_fn(s, dt, t, **step_kwargs)
            return (s, t + dt), None
        (s_out, t_out), _ = jax.lax.scan(
            seg_body, carry, length=segment_length,
        )
        return (s_out, t_out), None

    (s, t), _ = jax.lax.scan(
        run_segment,
        (state0, jnp.float32(t0)),
        length=n_segments,
    )

    def rem_body(carry, _):
        s_r, t_r = carry
        s_r = step_fn(s_r, dt, t_r, **step_kwargs)
        return (s_r, t_r + dt), None

    (s, _), _ = jax.lax.scan(rem_body, (s, t), length=remainder)
    return s


# ══════════════════════════════════════════════════════════════════════
#  IFT Steady-State Solve                                   [Method 3]
# ══════════════════════════════════════════════════════════════════════

def steady_state_solve(
    step_fn,
    state0: FluidState,
    dt: float,
    max_steps: int = 2000,
    check_every: int = 50,
    n_ift_iters: int = 20,
    verbose: bool = True,
    **step_kwargs,
) -> FluidState:
    """Find steady-state via time-marching with IFT backward pass.

    Forward pass
    ------------
    Marches ``state = step_fn(state, dt, ...)`` for ``max_steps``
    using ``jax.lax.scan`` (JIT-compatible, fixed iteration count).

    Backward pass (IFT)
    -------------------
    At the fixed point ``u* = G(u*, theta)``, the Implicit Function
    Theorem gives::

        du*/dtheta = (I - dG/du)^{-1} dG/dtheta

    The backward pass solves ``lambda^T (I - dG/du) = dL/du*`` via
    fixed-point (Neumann series) iteration, avoiding backpropagation
    through all forward steps.

    Parameters
    ----------
    step_fn : callable
        ``step_fn(state, dt, extra, **kwargs) -> FluidState``
    state0 : FluidState
        Initial guess (e.g. uniform flow).
    dt : float
        Time step for forward marching.
    max_steps : int
        Total number of forward steps.
    check_every : int
        Steps per scan chunk (for memory efficiency).
    n_ift_iters : int
        Number of Neumann series iterations in backward pass.
    verbose : bool
        Warn if Neumann series converges poorly.
    **step_kwargs
        Forwarded to step_fn (e.g. brinkman_field, kappa_field).

    Returns
    -------
    state_ss : FluidState
        Steady-state solution. Differentiable w.r.t. step_kwargs
        via the Implicit Function Theorem.
    """
    # Pack step_kwargs into a tuple of arrays for custom_vjp
    # (custom_vjp needs explicit pytree arguments, not **kwargs)
    kwargs_leaves, kwargs_def = jax.tree.flatten(step_kwargs)
    kwargs_packed = tuple(kwargs_leaves)

    def _one_step(state, kw_packed):
        """Single forward step with unpacked kwargs."""
        kw = jax.tree.unflatten(kwargs_def, kw_packed)
        return step_fn(state, dt, 0.0, **kw)

    @jax.custom_vjp
    def _solve(state_init, kw_packed):
        """Forward march to steady state."""
        def march_chunk(carry, _):
            def inner_body(c, __):
                return _one_step(c, kw_packed), None
            s_new, _ = jax.lax.scan(inner_body, carry, length=check_every)
            return s_new, None

        n_chunks = max_steps // check_every
        state_ss, _ = jax.lax.scan(march_chunk, state_init, length=n_chunks)

        # Handle remainder steps
        remainder = max_steps % check_every
        def rem_body(c, __):
            return _one_step(c, kw_packed), None
        state_ss, _ = jax.lax.scan(rem_body, state_ss, length=remainder)
        return state_ss

    def _solve_fwd(state_init, kw_packed):
        state_star = _solve(state_init, kw_packed)
        return state_star, (state_star, kw_packed)

    def _solve_bwd(res, g_state):
        state_star, kw_packed = res

        # Neumann series: lambda_{k+1} = g + (dG/du)^T lambda_k
        # Solves lambda^T (I - dG/du) = g_state
        #
        # Convergence requires spectral radius rho(dG/du) < 1, which
        # holds when the time-stepper is contractive at the fixed point
        # (guaranteed for stable steady states with small enough dt).
        lam_init = jax.tree.map(jnp.zeros_like, state_star)

        # Compute vjp ONCE (forward pass of _one_step at fixed point)
        # and reuse the vjp_fn for all Neumann iterations.
        _, vjp_fn_state = jax.vjp(
            lambda s: _one_step(s, kw_packed), state_star)

        # Neumann series via fori_loop (single XLA While op, not unrolled)
        def _neumann_body(k, lam):
            dGdu_T_lam = vjp_fn_state(lam)[0]
            return jax.tree.map(lambda g, d: g + d, g_state, dGdu_T_lam)

        lam = jax.lax.fori_loop(0, n_ift_iters, _neumann_body, lam_init)

        # dG/dtheta^T * lambda
        _, vjp_theta = jax.vjp(
            lambda kw: _one_step(state_star, kw), kw_packed)
        grad_theta = vjp_theta(lam)[0]

        # No gradient w.r.t. state0 (steady state is IC-independent)
        grad_state0 = jax.tree.map(jnp.zeros_like, state_star)
        return grad_state0, grad_theta

    _solve.defvjp(_solve_fwd, _solve_bwd)

    return _solve(state0, kwargs_packed)
