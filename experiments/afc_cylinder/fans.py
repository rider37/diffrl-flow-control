"""
FANS (First-order Analytic Navier-Stokes) for Active Flow Control.

SHAC-style differentiable RL on DiffFlow solver:
  - Actor: 1st-order gradient through CFD simulator (jax.grad)
  - Critic: 0th-order TD learning (standard supervised loss)

The entire pipeline is differentiable:
  obs → policy_θ(obs) → action → jet_force → CFD_step(×50) → next_obs → ...
  ←←←← jax.grad flows through entire chain ←←←←←←←←←←←←←←←←←←←←←←←←←
"""

import jax
import jax.numpy as jnp

from jax_cfd.core.state import FluidState
from jax_cfd.optim.simulate import differentiable_simulate_segmented

# Reuse MLP / optimizer from PPO.
# RELU hidden activation (was mlp_forward_tanh): unifies the cylinder SHAC actor
# with PPO (relu) and with RB SHAC (relu), so the SHAC-vs-PPO comparison differs
# only in the gradient estimator, and training matches the relu evaluation.
from experiments.afc_cylinder.train_ppo import (
    init_mlp, mlp_forward_relu as mlp_forward,
    init_actor_critic, actor_forward, critic_forward,
    init_adam, adam_update,
)


# ═══════════════════════════════════════════════════════════════════════
# Step function wrapper (jet force injection)
# ═══════════════════════════════════════════════════════════════════════

def make_fans_step_fn(base_step_fn, jet_basis, Q_max,
                      jet_method='body_force'):
    """Wrap CFD step_fn to accept ``action`` kwarg for jet forcing.

    Parameters
    ----------
    base_step_fn : callable
        ``step_fn(state, dt, extra)`` from ``create_step_fn``.
    jet_basis : dict
        Pre-computed force fields: ``{'Fx0','Fy0','Fx1','Fy1'}``.
        Ignored when ``jet_method='surface_bc'``.
    Q_max : float
        Maximum jet volume flow rate.
    jet_method : str
        ``'body_force'`` (external impulse) or ``'surface_bc'``
        (action passed as extra → IBM applies jet surface velocity).

    Returns
    -------
    fans_step : callable
        ``fans_step(state, dt, t, action=0.0) -> FluidState``
    """
    if jet_method == 'surface_bc':
        # Action passed as extra → IBM handles jet surface velocity
        def fans_step(state, dt, t, action=jnp.float32(0.0)):
            return base_step_fn(state, dt, action)
        return fans_step

    # Original body-force approach: external velocity impulse
    Fx0, Fy0 = jet_basis['Fx0'], jet_basis['Fy0']
    Fx1, Fy1 = jet_basis['Fx1'], jet_basis['Fy1']

    def fans_step(state, dt, t, action=jnp.float32(0.0)):
        # 1. Base CFD step (RK3 + IBM + pressure)
        state_new = base_step_fn(state, dt, t)

        # 2. Jet forcing (ZNMF: top = +Q, bottom = -Q)
        Q_top = action * Q_max
        Q_bot = -Q_top
        Fx = Q_top * Fx0 + Q_bot * Fx1
        Fy = Q_top * Fy0 + Q_bot * Fy1

        # 3. Apply as velocity impulse
        u_new = state_new.velocity[0] + Fx * dt
        v_new = state_new.velocity[1] + Fy * dt

        return FluidState(
            velocity=(u_new, v_new),
            pressure=state_new.pressure,
            scalars=state_new.scalars,
            ibm_forces=state_new.ibm_forces,
        )

    return fans_step


# ═══════════════════════════════════════════════════════════════════════
# Observation extraction (pure function, must be JIT-friendly)
# ═══════════════════════════════════════════════════════════════════════

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


def make_get_obs_fn_pressure(probe_ix_p, probe_iy_p):
    """Create a pure pressure observation function (p at probe points).

    Used for fair comparison with PPO (which also uses pressure obs by Rabault setup).

    Returns
    -------
    get_obs : callable(FluidState) -> jnp.ndarray of shape (n_probes,)
    """
    def get_obs(state):
        p = state.pressure
        p_probes = jax.scipy.ndimage.map_coordinates(
            p, [probe_ix_p, probe_iy_p], order=1, mode='nearest',
        )
        return p_probes
    return get_obs


# ═══════════════════════════════════════════════════════════════════════
# FANS rollout (differentiable, checkpointed)
# ═══════════════════════════════════════════════════════════════════════

def fans_rollout(fans_step_fn, state0, actor_params, get_obs_fn,
                 q_ref, h, action_interval, dt, segment_length=10):
    """Unroll h RL steps through the differentiable simulator.

    Each RL step:
      1. Extract observation from FluidState
      2. Policy forward: obs → action (deterministic, tanh)
      3. Run action_interval CFD steps with jet forcing
      4. Record Cd and Cl (reward computed in actor_loss)

    Parameters
    ----------
    fans_step_fn : callable
        From ``make_fans_step_fn`` — accepts ``action`` kwarg.
    state0 : FluidState
        Initial fluid state.
    actor_params : list
        Actor network parameters (MLP weights).
    get_obs_fn : callable
        Pure function: FluidState → obs vector.
    q_ref : float
        Reference force (0.5 * rho * U^2 * D).
    h : int
        Truncation horizon (number of RL steps).
    action_interval : int
        CFD steps per RL action.
    dt : float
        CFD timestep.
    segment_length : int
        Checkpoint segment size for memory efficiency.

    Returns
    -------
    state_h : FluidState
        Terminal state after h RL steps.
    cds : jnp.ndarray, shape (h,)
        Per-step drag coefficients.
    cls : jnp.ndarray, shape (h,)
        Per-step lift coefficients.
    obs_history : jnp.ndarray, shape (h, obs_dim)
        Observations at each RL step.
    actions : jnp.ndarray, shape (h,)
        Actions taken at each RL step.
    """

    @jax.checkpoint
    def rl_step(state, _):
        # 1. Observation
        obs = get_obs_fn(state)

        # 2. Policy (deterministic: tanh squash, use mean only)
        out = mlp_forward(actor_params, obs)
        # Actor outputs [mean, log_std] for action_dim=1 → take mean only
        action_dim = out.shape[-1] // 2
        mean = out[..., :action_dim]
        action = jnp.tanh(mean).squeeze()  # scalar ∈ [-1, 1]

        # 3. Differentiable CFD rollout with jet forcing
        state_next = differentiable_simulate_segmented(
            fans_step_fn, state, action_interval, dt,
            segment_length=segment_length,
            action=action,
        )

        # 4. Record force coefficients
        cd = state_next.ibm_forces[0] / q_ref
        cl = state_next.ibm_forces[1] / q_ref

        return state_next, (cd, cl, obs, action)

    state_h, (cds, cls, obs_history, actions) = jax.lax.scan(
        rl_step, state0, None, length=h,
    )

    return state_h, cds, cls, obs_history, actions




# ═══════════════════════════════════════════════════════════════════════
# Environment-matched differentiable rollout (width-matched re-train)
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
        # jax_cfd.optim.simulate.differentiable_simulate_segmented)
        ck_substep = jax.checkpoint(substep)
        seg_out, _ = jax.lax.scan(ck_substep, carry, None, length=segment_length)
        return seg_out, None

    carry0 = (state0, jnp.float32(t0), jnp.float32(q0), jnp.float32(0.0), jnp.float32(0.0))
    carry, _ = jax.lax.scan(run_segment, carry0, None, length=n_segments)
    carry, _ = jax.lax.scan(substep, carry, None, length=remainder)
    s, _, q, cd_sum, cl_sum = carry
    return s, q, cd_sum / n_steps, cl_sum / n_steps


def fans_rollout_env(fans_step_fn, state0, q0, actor_params, get_obs_fn,
                     q_ref, h, action_interval, dt, segment_length, ema_alpha):
    """Like fans_rollout, but reproduces the environment's action smoothing
    and interval-mean force coefficients (see differentiable_simulate_segmented_env)."""

    @jax.checkpoint
    def rl_step(carry, _):
        state, q = carry
        obs = get_obs_fn(state)
        out = mlp_forward(actor_params, obs)
        action_dim = out.shape[-1] // 2
        action = jnp.tanh(out[..., :action_dim]).squeeze()
        state_next, q_next, cd_mean, cl_mean = differentiable_simulate_segmented_env(
            fans_step_fn, state, q, action, action_interval, dt,
            segment_length, ema_alpha, q_ref)
        return (state_next, q_next), (cd_mean, cl_mean, obs, action)

    (state_h, q_h), (cds, cls, obs_history, actions) = jax.lax.scan(
        rl_step, (state0, jnp.float32(q0)), None, length=h)
    return state_h, q_h, cds, cls, obs_history, actions

# ═══════════════════════════════════════════════════════════════════════
# Actor loss (1st-order: gradient through simulator)
# ═══════════════════════════════════════════════════════════════════════

def fans_actor_loss(actor_params, critic_params, state0,
                    fans_step_fn, get_obs_fn, q_ref,
                    h, action_interval, dt, gamma,
                    segment_length=10, cd_baseline=0.0,
                    cl_weight=0.2, action_penalty_coef=0.0,
                    q0=jnp.float32(0.0), ema_alpha=1.0):
    """Actor loss for FANS: negative discounted return.
    (width-matched re-train: environment-matched rollout, see fans_rollout_env)

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
    state_h, q_h, cds, cls, obs_history, actions = fans_rollout_env(
        fans_step_fn, state0, q0, actor_params, get_obs_fn,
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

def fans_critic_loss(critic_params, obs_batch, targets):
    """Critic MSE loss: L = mean((V_φ(obs) - target)²).

    Parameters
    ----------
    critic_params : list
        Critic network parameters.
    obs_batch : jnp.ndarray, shape (n, obs_dim)
        Observations.
    targets : jnp.ndarray, shape (n,)
        TD targets (n-step returns).
    """
    V_pred = jnp.array([
        mlp_forward(critic_params, obs).squeeze()
        for obs in obs_batch  # small batch, no vmap needed
    ]) if obs_batch.ndim == 2 else mlp_forward(critic_params, obs_batch).squeeze()

    return jnp.mean((V_pred - targets) ** 2)


def fans_critic_loss_batched(critic_params, obs_batch, targets):
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


# ═══════════════════════════════════════════════════════════════════════
# Gradient check utility
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Stochastic SHAC: Gaussian policy with reparameterization trick
# ═══════════════════════════════════════════════════════════════════════

def fans_rollout_stochastic(fans_step_fn, state0, actor_params, get_obs_fn,
                            q_ref, h, action_interval, dt, key,
                            segment_length=10):
    """Stochastic SHAC rollout: Gaussian policy with reparam trick.

    Same as fans_rollout but samples actions from N(mean, std) instead of
    using deterministic mean.  The reparameterization trick allows gradients
    to flow through the sampling operation.
    """

    @jax.checkpoint
    def rl_step(carry, _):
        state, rng = carry
        rng, subkey = jax.random.split(rng)

        obs = get_obs_fn(state)

        out = mlp_forward(actor_params, obs)
        action_dim = out.shape[-1] // 2
        mean = out[..., :action_dim]
        log_std = jnp.clip(out[..., action_dim:], -5.0, 2.0)
        std = jnp.exp(log_std)

        noise = jax.random.normal(subkey, mean.shape)
        action_raw = mean + std * noise
        action = jnp.tanh(action_raw).squeeze()

        # Log probability (Gaussian + tanh correction)
        log_prob = jnp.sum(
            -0.5 * ((action_raw - mean) / std) ** 2
            - log_std - 0.5 * jnp.log(2 * jnp.pi),
            axis=-1,
        )
        log_prob = log_prob - jnp.sum(
            jnp.log(1 - jnp.tanh(action_raw) ** 2 + 1e-6), axis=-1,
        )

        # Entropy of Gaussian (before tanh)
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)

        state_next = differentiable_simulate_segmented(
            fans_step_fn, state, action_interval, dt,
            segment_length=segment_length,
            action=action,
        )

        cd = state_next.ibm_forces[0] / q_ref
        cl = state_next.ibm_forces[1] / q_ref

        return (state_next, rng), (cd, cl, obs, action, log_prob, entropy)

    (state_h, _), (cds, cls, obs_history, actions, log_probs, entropies) = \
        jax.lax.scan(rl_step, (state0, key), None, length=h)

    return state_h, cds, cls, obs_history, actions, log_probs, entropies


def fans_actor_loss_stochastic(actor_params, critic_params, state0,
                               fans_step_fn, get_obs_fn, q_ref,
                               h, action_interval, dt, gamma, key,
                               segment_length=10, cd_baseline=0.0,
                               cl_weight=0.2, entropy_coef=0.01):
    """Stochastic SHAC actor loss with entropy regularization.

    L = -(discounted return) - entropy_coef * mean(entropy)
    """
    state_h, cds, cls, obs_history, actions, log_probs, entropies = \
        fans_rollout_stochastic(
            fans_step_fn, state0, actor_params, get_obs_fn,
            q_ref, h, action_interval, dt, key, segment_length,
        )

    rewards = -(cds - cd_baseline) - cl_weight * jnp.abs(cls)

    obs_h = get_obs_fn(state_h)
    V_h = jax.lax.stop_gradient(
        critic_forward({'critic': critic_params}, obs_h)
    )

    gammas = gamma ** jnp.arange(h)
    discounted_return = jnp.sum(gammas * rewards) + gamma ** h * V_h

    loss = -discounted_return - entropy_coef * jnp.mean(entropies)

    aux = {
        'rewards': rewards,
        'cds': cds,
        'cls': cls,
        'actions': actions,
        'obs_history': obs_history,
        'obs_h': obs_h,
        'state_h': state_h,
        'V_h': V_h,
        'return': discounted_return,
        'entropy': jnp.mean(entropies),
        'log_probs': log_probs,
    }
    return loss, aux


# ═══════════════════════════════════════════════════════════════════════
# SAPO: Soft Analytic Policy Optimization (Xing et al., ICLR 2025)
# Entropy inside return + soft critic + automatic alpha tuning
# ═══════════════════════════════════════════════════════════════════════

def fans_rollout_sapo(fans_step_fn, state0, actor_params, get_obs_fn,
                      q_ref, h, action_interval, dt, key, log_alpha,
                      segment_length=10):
    """SAPO rollout: entropy-augmented return with auto alpha."""
    alpha = jnp.exp(log_alpha)

    @jax.checkpoint
    def rl_step(carry, _):
        state, rng = carry
        rng, subkey = jax.random.split(rng)

        obs = get_obs_fn(state)

        out = mlp_forward(actor_params, obs)
        action_dim = out.shape[-1] // 2
        mean = out[..., :action_dim]
        log_std = jnp.clip(out[..., action_dim:], -5.0, 2.0)
        std = jnp.exp(log_std)

        noise = jax.random.normal(subkey, mean.shape)
        action_raw = mean + std * noise
        action = jnp.tanh(action_raw).squeeze()

        log_prob = jnp.sum(
            -0.5 * ((action_raw - mean) / std) ** 2
            - log_std - 0.5 * jnp.log(2 * jnp.pi),
            axis=-1,
        )
        log_prob = log_prob - jnp.sum(
            jnp.log(1 - jnp.tanh(action_raw) ** 2 + 1e-6), axis=-1,
        )
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)

        state_next = differentiable_simulate_segmented(
            fans_step_fn, state, action_interval, dt,
            segment_length=segment_length,
            action=action,
        )

        cd = state_next.ibm_forces[0] / q_ref
        cl = state_next.ibm_forces[1] / q_ref

        return (state_next, rng), (cd, cl, obs, action, log_prob, entropy)

    (state_h, _), (cds, cls, obs_history, actions, log_probs, entropies) = \
        jax.lax.scan(rl_step, (state0, key), None, length=h)

    return state_h, cds, cls, obs_history, actions, log_probs, entropies


def fans_actor_loss_sapo(actor_params, critic_params, state0,
                         fans_step_fn, get_obs_fn, q_ref,
                         h, action_interval, dt, gamma, key, log_alpha,
                         segment_length=10, cd_baseline=0.0, cl_weight=0.2):
    """SAPO actor loss: entropy inside the return, soft critic bootstrap."""
    alpha = jnp.exp(log_alpha)

    state_h, cds, cls, obs_history, actions, log_probs, entropies = \
        fans_rollout_sapo(
            fans_step_fn, state0, actor_params, get_obs_fn,
            q_ref, h, action_interval, dt, key, log_alpha, segment_length,
        )

    # Entropy-augmented reward: r'_t = r_t + alpha * H_t
    rewards_raw = -(cds - cd_baseline) - cl_weight * jnp.abs(cls)
    rewards_augmented = rewards_raw + alpha * entropies

    # Soft critic bootstrap at terminal state
    obs_h = get_obs_fn(state_h)
    V_soft_h = jax.lax.stop_gradient(
        critic_forward({'critic': critic_params}, obs_h)
    )

    gammas = gamma ** jnp.arange(h)
    discounted_return = jnp.sum(gammas * rewards_augmented) + gamma ** h * V_soft_h

    loss = -discounted_return

    aux = {
        'rewards': rewards_raw,
        'rewards_augmented': rewards_augmented,
        'cds': cds,
        'cls': cls,
        'actions': actions,
        'obs_history': obs_history,
        'obs_h': obs_h,
        'state_h': state_h,
        'V_h': V_soft_h,
        'return': discounted_return,
        'entropy': jnp.mean(entropies),
        'log_probs': log_probs,
        'alpha': alpha,
    }
    return loss, aux


def gradient_check(actor_params, critic_params, state0,
                   fans_step_fn, get_obs_fn, q_ref,
                   h, action_interval, dt, gamma,
                   segment_length, cd_baseline,
                   param_idx=0, eps=1e-3):
    """Compare autodiff gradient with finite difference for validation.

    Tests gradient of the first parameter of actor_params[param_idx].
    """
    def loss_fn(a_params):
        l, _ = fans_actor_loss(
            a_params, critic_params, state0,
            fans_step_fn, get_obs_fn, q_ref,
            h, action_interval, dt, gamma,
            segment_length, cd_baseline,
        )
        return l

    # Autodiff
    grad_ad = jax.grad(loss_fn)(actor_params)

    # Finite difference on first element of first weight matrix
    W, b = actor_params[param_idx]
    loss_0 = loss_fn(actor_params)

    W_plus = W.at[0, 0].add(eps)
    params_plus = list(actor_params)
    params_plus[param_idx] = (W_plus, b)
    loss_plus = loss_fn(params_plus)

    grad_fd = (loss_plus - loss_0) / eps
    grad_ad_val = grad_ad[param_idx][0][0, 0]  # W gradient at [0,0]

    return {
        'fd': float(grad_fd),
        'ad': float(grad_ad_val),
        'rel_err': float(jnp.abs(grad_fd - grad_ad_val) / (jnp.abs(grad_fd) + 1e-8)),
    }
