"""Differentiable pieces of SHAC for the cylinder: the step function that passes
the jet amplitude to the immersed boundary, the probe observation, the
differentiable rollout with the environment's action filter and reward
(interval-mean drag and lift), the actor loss (negative discounted return with a
learned terminal value), the batched critic loss and n-step TD targets.
"""
import jax
import jax.numpy as jnp


# actor and critic networks shared with PPO (ReLU)
from diffrl_flow_control.cylinder.ppo import mlp_forward_relu as mlp_forward, critic_forward


# ═══════════════════════════════════════════════════════════════════════
# Step function wrapper (jet force injection)
# ═══════════════════════════════════════════════════════════════════════

def make_jet_step_fn(base_step_fn):
    """Wrap ``step_fn(state, dt, extra)`` as ``step(state, dt, t, action)``; the jet
    amplitude is passed to the immersed boundary as ``extra``."""
    def shac_step(state, dt, t, action=jnp.float32(0.0)):
        return base_step_fn(state, dt, action)
    return shac_step

def make_get_obs_fn(probe_ix_u, probe_iy_u, probe_ix_v, probe_iy_v):
    """Create a pure velocity observation function (u, v at probe points).

    Returns
    -------
    get_obs : callable(FluidState) -> jnp.ndarray of shape (2*n_probes,)
    """
    def get_obs(state):
        u, v = state.velocity
        u_probes = jax.scipy.ndimage.map_coordinates(
            u, [probe_ix_u, probe_iy_u], order=1, mode='nearest',
        )
        v_probes = jax.scipy.ndimage.map_coordinates(
            v, [probe_ix_v, probe_iy_v], order=1, mode='nearest',
        )
        return jnp.concatenate([u_probes, v_probes])

    return get_obs


# ═══════════════════════════════════════════════════════════════════════
# Environment-matched differentiable rollout
#   * action smoothed per CFD substep with the environment's EMA
#     (Q <- Q + alpha (a - Q)), exactly as CylinderFlowEnv.step
#   * reward uses the action-interval MEAN of Cd and of Cl, exactly as
#     CylinderFlowEnv.step (r = -(mean Cd - Cd0) - w |mean Cl|)
#   * the smoothed action Q is carried across RL steps and across epochs
#     (reset to 0 together with the flow state), as env.Q_actual
# ═══════════════════════════════════════════════════════════════════════

def differentiable_simulate_segmented_env(step_fn, state0, q0, action, n_steps, dt,
                                          segment_length, ema_alpha, q_ref, t0=0.0):
    """Run n_steps substeps with EMA-smoothed action; return
    (state, q_final, cd_mean, cl_mean).  Segmented checkpointing as in
    differentiable_simulate_segmented."""
    n_segments = n_steps // segment_length
    remainder = n_steps % segment_length

    def substep(carry, _):
        s, t, q, cd_sum, cl_sum = carry
        q = q + ema_alpha * (action - q)
        s = step_fn(s, dt, t, action=q)
        cd_sum = cd_sum + s.ibm_forces[0] / q_ref
        cl_sum = cl_sum + s.ibm_forces[1] / q_ref
        return (s, t + dt, q, cd_sum, cl_sum), None

    def run_segment(carry, _):
        # per-substep remat inside each segment (same structure as
        # diffrl_flow_control.solver.rollout.differentiable_simulate_segmented)
        ck_substep = jax.checkpoint(substep)
        seg_out, _ = jax.lax.scan(ck_substep, carry, None, length=segment_length)
        return seg_out, None

    carry0 = (state0, jnp.float32(t0), jnp.float32(q0), jnp.float32(0.0), jnp.float32(0.0))
    carry, _ = jax.lax.scan(run_segment, carry0, None, length=n_segments)
    carry, _ = jax.lax.scan(substep, carry, None, length=remainder)
    s, _, q, cd_sum, cl_sum = carry
    return s, q, cd_sum / n_steps, cl_sum / n_steps


def shac_rollout_env(shac_step_fn, state0, q0, actor_params, get_obs_fn,
                     q_ref, h, action_interval, dt, segment_length, ema_alpha):
    """Like shac_rollout, but reproduces the environment's action smoothing
    and interval-mean force coefficients (see differentiable_simulate_segmented_env)."""

    @jax.checkpoint
    def rl_step(carry, _):
        state, q = carry
        obs = get_obs_fn(state)
        out = mlp_forward(actor_params, obs)
        action_dim = out.shape[-1] // 2
        action = jnp.tanh(out[..., :action_dim]).squeeze()
        state_next, q_next, cd_mean, cl_mean = differentiable_simulate_segmented_env(
            shac_step_fn, state, q, action, action_interval, dt,
            segment_length, ema_alpha, q_ref)
        return (state_next, q_next), (cd_mean, cl_mean, obs, action)

    (state_h, q_h), (cds, cls, obs_history, actions) = jax.lax.scan(
        rl_step, (state0, jnp.float32(q0)), None, length=h)
    return state_h, q_h, cds, cls, obs_history, actions

# ═══════════════════════════════════════════════════════════════════════
# Actor loss (1st-order: gradient through simulator)
# ═══════════════════════════════════════════════════════════════════════

def shac_actor_loss(actor_params, critic_params, state0,
                    shac_step_fn, get_obs_fn, q_ref,
                    h, action_interval, dt, gamma,
                    segment_length=10, cd_baseline=0.0,
                    cl_weight=0.2, action_penalty_coef=0.0,
                    q0=jnp.float32(0.0), ema_alpha=1.0):
    """Actor loss for SHAC: negative discounted return over the
    environment-matched rollout (see shac_rollout_env).

    Reward follows Rabault et al. (2019):
        r_t = -(Cd_t - Cd_baseline) - cl_weight * |Cl_t|

    L = -(Σ_{t=0}^{h-1} γ^t r_t + γ^h V_φ(s_h))

    Differentiated w.r.t. actor_params (argnums=0) via jax.grad.
    The gradient flows through the CFD simulator.

    Returns
    -------
    loss : scalar
    aux : dict with rewards, actions, obs for logging/critic training
    """
    state_h, q_h, cds, cls, obs_history, actions = shac_rollout_env(
        shac_step_fn, state0, q0, actor_params, get_obs_fn,
        q_ref, h, action_interval, dt, segment_length, ema_alpha,
    )

    # Rabault reward on interval means (as CylinderFlowEnv.step):
    #   r = -(mean Cd - Cd0) - cl_weight * |mean Cl|
    rewards = -(cds - cd_baseline) - cl_weight * jnp.abs(cls)

    # Critic value at terminal state (stop gradient — critic is fixed)
    obs_h = get_obs_fn(state_h)
    V_h = jax.lax.stop_gradient(
        critic_forward({'critic': critic_params}, obs_h)
    )

    # Discounted return
    gammas = gamma ** jnp.arange(h)
    discounted_return = jnp.sum(gammas * rewards) + gamma ** h * V_h

    # Actor loss: minimize negative return
    loss = -discounted_return

    # Action regularization (optional, default off)
    if action_penalty_coef > 0:
        loss = loss + action_penalty_coef * jnp.mean(actions ** 2)

    aux = {
        'rewards': rewards,
        'cds': cds,
        'cls': cls,
        'actions': actions,
        'obs_history': obs_history,
        'obs_h': obs_h,
        'state_h': state_h,
        'q_h': q_h,
        'V_h': V_h,
        'return': discounted_return,
    }
    return loss, aux


# ═══════════════════════════════════════════════════════════════════════
# Critic loss (0th-order: standard TD, no simulator gradient)
# ═══════════════════════════════════════════════════════════════════════


def shac_critic_loss_batched(critic_params, obs_batch, targets):
    """Vectorized critic loss for batch of observations."""
    def single_pred(obs):
        return mlp_forward(critic_params, obs).squeeze()

    V_pred = jax.vmap(single_pred)(obs_batch)
    return jnp.mean((V_pred - targets) ** 2)


# ═══════════════════════════════════════════════════════════════════════
# TD target computation
# ═══════════════════════════════════════════════════════════════════════

def compute_td_targets(rewards, V_h, gamma, h):
    """Compute n-step TD targets for each timestep.

    For step t:
        G_t = Σ_{k=0}^{h-1-t} γ^k r_{t+k} + γ^{h-t} V(s_h)

    Parameters
    ----------
    rewards : jnp.ndarray, shape (h,)
    V_h : float
        Critic value at terminal state.
    gamma : float
    h : int

    Returns
    -------
    targets : jnp.ndarray, shape (h,)
    """
    targets = jnp.zeros(h)

    def body(carry, t_rev):
        G = carry
        t = h - 1 - t_rev
        G = rewards[t] + gamma * G
        return G, G

    _, targets_rev = jax.lax.scan(body, V_h, jnp.arange(h))
    targets = targets_rev[::-1]  # reverse to get chronological order

    return targets
