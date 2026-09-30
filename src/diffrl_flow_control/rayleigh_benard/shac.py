"""Differentiable pieces of SHAC for Rayleigh--Benard convection: the map from
the action to the heater temperatures, the step function that passes them to the
bottom-wall boundary, the probe observation, the Nusselt number, the
differentiable rollout (reward 1 - Nu / Nu_base), the actor loss (negative
discounted return with a learned terminal value), the batched critic loss and
n-step TD targets.
"""
import jax
import jax.numpy as jnp

from diffrl_flow_control.solver.rollout import differentiable_simulate_segmented

# actor and critic networks shared with PPO (ReLU)
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward, critic_forward


# ═══════════════════════════════════════════════════════════════════════
# Step function wrapper (action → segment temperature mapping)
# ═══════════════════════════════════════════════════════════════════════

def make_map_action_fn(T_hot, amplitude):
    """Create action → wall temperature mapping (Markmann 2025, Eq. 3)."""
    def map_action(action):
        """Map [-1,1]^n_segments → wall temperatures (zero-mean, clamped)."""
        a_centered = action - jnp.mean(action)
        a_max = jnp.max(jnp.abs(a_centered))
        a_scaled = jnp.where(
            a_max > amplitude,
            a_centered * amplitude / jnp.maximum(a_max, 1e-8),
            a_centered,
        )
        return T_hot + a_scaled
    return map_action


def make_heater_step_fn(base_step_fn, T_hot, amplitude):
    """Wrap CFD step_fn to accept ``action`` kwarg → segment temperatures.

    Parameters
    ----------
    base_step_fn : callable
        ``step_fn(state, dt, extra)`` from ``create_step_fn``.
    T_hot : float
        Base hot-wall temperature.
    amplitude : float
        Maximum temperature perturbation (Markmann 2025: 0.75).

    Returns
    -------
    shac_step : callable
        ``shac_step(state, dt, t, action=None) -> FluidState``
    """
    map_action = make_map_action_fn(T_hot, amplitude)

    def shac_step(state, dt, t, action=None):
        if action is None:
            return base_step_fn(state, dt, None)
        T_segments = map_action(action)
        return base_step_fn(state, dt, T_segments)

    return shac_step


# ═══════════════════════════════════════════════════════════════════════
# Observation extraction (pure function, JIT-friendly)
# ═══════════════════════════════════════════════════════════════════════

def make_rb_get_obs_fn(probe_ix_c, probe_iy_c,
                       probe_ix_u, probe_iy_u,
                       probe_ix_v, probe_iy_v):
    """Create a pure observation function: FluidState → (T, u, v) at probes.

    Returns
    -------
    get_obs : callable(FluidState) -> jnp.ndarray, shape (3 * n_probes,)
    """
    def get_obs(state):
        T = state.scalars['T']
        u, v = state.velocity
        T_probes = jax.scipy.ndimage.map_coordinates(
            T, [probe_ix_c, probe_iy_c], order=1, mode='nearest')
        u_probes = jax.scipy.ndimage.map_coordinates(
            u, [probe_ix_u, probe_iy_u], order=1, mode='nearest')
        v_probes = jax.scipy.ndimage.map_coordinates(
            v, [probe_ix_v, probe_iy_v], order=1, mode='nearest')
        return jnp.concatenate([T_probes, u_probes, v_probes])

    return get_obs


def make_compute_nu_fn(grid, T_hot, T_cold, segment_map=None):
    """Create pure Nusselt number function (wall-gradient method).

    When segment_map is provided, returns a function that accepts
    T_wall_segments to use the actual per-segment wall temperature
    for the bottom-wall gradient computation.

    Parameters
    ----------
    grid : MACGrid
    T_hot : float
    T_cold : float
    segment_map : jnp.ndarray, optional
        (Nx,) array mapping cell index to segment index.

    Returns
    -------
    compute_nu : callable(FluidState, T_wall_segments=None) -> scalar
    """
    dy = grid.spacings[1]
    Ly = grid.domain_size[1]
    dT = T_hot - T_cold

    def compute_nu(state, T_wall_segments=None):
        T = state.scalars['T']
        if T_wall_segments is not None and segment_map is not None:
            T_wall_bot = T_wall_segments[segment_map]  # (Nx,)
        else:
            T_wall_bot = T_hot
        Nu_bot = jnp.mean(2.0 * Ly * (T_wall_bot - T[:, 0]) / (dy * dT))
        Nu_top = jnp.mean(2.0 * Ly * (T[:, -1] - T_cold) / (dy * dT))
        return 0.5 * (Nu_bot + Nu_top)

    return compute_nu


# ═══════════════════════════════════════════════════════════════════════
# SHAC rollout (differentiable, checkpointed)
# ═══════════════════════════════════════════════════════════════════════

def shac_rollout(shac_step_fn, state0, actor_params, get_obs_fn,
                 compute_nu_fn, h, action_interval, dt,
                 segment_length=10, action_dim=12,
                 map_action_fn=None):
    """Unroll h RL steps through the differentiable simulator.

    Each RL step:
      1. Extract observation (T, u, v at probes)
      2. Policy forward: obs → 12-dim action (deterministic, tanh)
      3. Run action_interval CFD steps with segment heating
      4. Record Nusselt number (using actual wall temperatures)

    Parameters
    ----------
    shac_step_fn : callable
        From ``make_heater_step_fn`` — accepts ``action`` kwarg.
    state0 : FluidState
    actor_params : list
        Actor network parameters (MLP weights).
    get_obs_fn : callable
        Pure function: FluidState → obs vector.
    compute_nu_fn : callable
        Pure function: FluidState, T_wall_segments -> Nu scalar.
    h : int
        Truncation horizon (number of RL steps).
    action_interval : int
        CFD steps per RL action.
    dt : float
        CFD timestep.
    segment_length : int
        Checkpoint segment size for memory efficiency.
    action_dim : int
        Action dimension (number of segments).
    map_action_fn : callable, optional
        Maps action → T_wall_segments for correct Nu computation.

    Returns
    -------
    state_h : FluidState
    nus : jnp.ndarray, shape (h,)
    obs_history : jnp.ndarray, shape (h, obs_dim)
    actions : jnp.ndarray, shape (h, action_dim)
    """

    @jax.checkpoint
    def rl_step(state, _):
        # 1. Observation
        obs = get_obs_fn(state)

        # 2. Policy (deterministic: tanh squash, use mean only)
        out = mlp_forward(actor_params, obs)
        mean = out[:action_dim]
        action = jnp.tanh(mean)  # (action_dim,) in [-1, 1]

        # 3. Differentiable CFD rollout with segment heating
        state_next = differentiable_simulate_segmented(
            shac_step_fn, state, action_interval, dt,
            segment_length=segment_length,
            action=action,
        )

        # 4. Record Nusselt number (with actual wall temperatures)
        if map_action_fn is not None:
            T_wall_segs = map_action_fn(action)
            nu = compute_nu_fn(state_next, T_wall_segs)
        else:
            nu = compute_nu_fn(state_next)

        return state_next, (nu, obs, action)

    state_h, (nus, obs_history, actions) = jax.lax.scan(
        rl_step, state0, None, length=h,
    )

    return state_h, nus, obs_history, actions


# ═══════════════════════════════════════════════════════════════════════
# Actor loss (1st-order: gradient through simulator)
# ═══════════════════════════════════════════════════════════════════════

def shac_actor_loss(actor_params, critic_params, state0,
                    shac_step_fn, get_obs_fn, compute_nu_fn,
                    nu_baseline, h, action_interval, dt, gamma,
                    segment_length=10, action_dim=12,
                    action_penalty_coef=0.0,
                    map_action_fn=None):
    """Actor loss for SHAC: negative discounted return.

    Reward (Markmann 2025):
        r_t = 1 - Nu_t / Nu_baseline

    L = -(sum_{t=0}^{h-1} gamma^t r_t + gamma^h V_phi(s_h))

    Differentiated w.r.t. actor_params (argnums=0) via jax.grad.
    The gradient flows through the CFD simulator.

    Returns
    -------
    loss : scalar
    aux : dict with rewards, nus, actions, obs for logging/critic training
    """
    state_h, nus, obs_history, actions = shac_rollout(
        shac_step_fn, state0, actor_params, get_obs_fn,
        compute_nu_fn, h, action_interval, dt,
        segment_length, action_dim,
        map_action_fn=map_action_fn,
    )

    # Markmann reward: R = 1 - Nu / Nu_base
    rewards = 1.0 - nus / nu_baseline

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
        'nus': nus,
        'actions': actions,
        'obs_history': obs_history,
        'obs_h': obs_h,
        'state_h': state_h,
        'V_h': V_h,
        'return': discounted_return,
    }
    return loss, aux


# ═══════════════════════════════════════════════════════════════════════
# Critic loss (0th-order: standard TD, no simulator gradient)
# ═══════════════════════════════════════════════════════════════════════

def shac_critic_loss_batched(critic_params, obs_batch, targets):
    """Vectorized critic MSE loss: L = mean((V_phi(obs) - target)^2)."""
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
        G_t = sum_{k=0}^{h-1-t} gamma^k r_{t+k} + gamma^{h-t} V(s_h)

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
    def body(carry, t_rev):
        G = carry
        t = h - 1 - t_rev
        G = rewards[t] + gamma * G
        return G, G

    _, targets_rev = jax.lax.scan(body, V_h, jnp.arange(h))
    return targets_rev[::-1]
