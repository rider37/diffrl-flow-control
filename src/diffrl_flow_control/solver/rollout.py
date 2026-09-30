"""
Differentiable simulation utilities.

Provides ``differentiable_simulate_segmented``: a differentiable rollout with
segmented gradient checkpointing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffrl_flow_control.solver.state import FluidState


# ══════════════════════════════════════════════════════════════════════
#  Standard differentiable simulation (per-step checkpoint)
# ══════════════════════════════════════════════════════════════════════



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

