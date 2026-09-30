"""
PPO for 2D cylinder Active Flow Control (AFC).

Matches Rabault et al. (2019, JFM) PPO setup:
  - Separate actor (2×512, ReLU) and critic (2×32, ReLU)
  - 20 episodes per PPO update, 80 actions per episode
  - 25 epochs with 20% random subsampling per epoch
  - Per-episode GAE (lambda=0.97)
  - 80% soft / 20% hard reset

Usage
-----
    # Full training (Rabault 2019 matching)
    python experiments/afc_cylinder/train_ppo.py

    # Quick test
    python experiments/afc_cylinder/train_ppo.py --test
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jax_cfd.core.operators import vorticity_2d
from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv
from experiments.afc_cylinder.config import (
    ENV_CONFIG, PPO_CONFIG, TEST_ENV_CONFIG, TEST_PPO_CONFIG,
    ENV_CONFIG_RE200, PPO_CONFIG_RE200,
    ENV_CONFIG_RE500, PPO_CONFIG_RE500,
)


# ═══════════════════════════════════════════════════════════════════════
# MLP (pure JAX, no frameworks)
# ═══════════════════════════════════════════════════════════════════════

def init_mlp(key, layer_dims):
    """Initialize MLP with Xavier/Glorot initialization."""
    params = []
    for i in range(len(layer_dims) - 1):
        key, subkey = jax.random.split(key)
        fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        W = jax.random.normal(subkey, (fan_in, fan_out)) * scale
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def mlp_forward_relu(params, x):
    """Forward pass with ReLU activations."""
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def mlp_forward_tanh(params, x):
    """Forward pass with tanh activations."""
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


# Module-level activation dispatch (set once, JAX-trace compatible)
_actor_fwd = mlp_forward_relu
_critic_fwd = mlp_forward_relu


def set_activations(activation='relu'):
    """Set activation function for actor/critic networks."""
    global _actor_fwd, _critic_fwd
    fwd = mlp_forward_relu if activation == 'relu' else mlp_forward_tanh
    _actor_fwd = fwd
    _critic_fwd = fwd


# ═══════════════════════════════════════════════════════════════════════
# Actor-Critic (separate networks, Rabault 2019)
# ═══════════════════════════════════════════════════════════════════════

def init_actor_critic(key, obs_dim, action_dim,
                      actor_hidden_dims, critic_hidden_dims):
    """Initialize separate actor and critic networks.

    Actor: obs → hidden → 2*action_dim (mean + log_std)
    Critic: obs → hidden → 1 (V(s))
    """
    key_a, key_c = jax.random.split(key)

    actor_dims = [obs_dim] + list(actor_hidden_dims) + [2 * action_dim]
    critic_dims = [obs_dim] + list(critic_hidden_dims) + [1]

    actor_params = init_mlp(key_a, actor_dims)
    critic_params = init_mlp(key_c, critic_dims)

    # Initialize log_std bias to -0.5 (moderate initial exploration)
    W_last, b_last = actor_params[-1]
    b_last = b_last.at[action_dim:].set(-0.5)
    actor_params[-1] = (W_last, b_last)

    return {'actor': actor_params, 'critic': critic_params}


def actor_forward(params, obs):
    """Get action distribution parameters (mean, log_std)."""
    out = _actor_fwd(params['actor'], obs)
    action_dim = out.shape[-1] // 2
    mean = out[..., :action_dim]
    log_std = out[..., action_dim:]
    log_std = jnp.clip(log_std, -2.0, 0.5)  # stability
    return mean, log_std


def critic_forward(params, obs):
    """Get value estimate V(s)."""
    return _critic_fwd(params['critic'], obs).squeeze(-1)


def sample_action(params, obs, key):
    """Sample action from Gaussian policy."""
    mean, log_std = actor_forward(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    action_raw = mean + std * noise
    # Squash to [-1, 1] via tanh
    action = jnp.tanh(action_raw)
    # Log probability (with tanh correction)
    log_prob = _gaussian_log_prob(action_raw, mean, log_std)
    log_prob = log_prob - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
    return action, log_prob, mean


def _gaussian_log_prob(x, mean, log_std):
    """Log probability of x under Gaussian(mean, std)."""
    std = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * ((x - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


def evaluate_action(params, obs, action):
    """Compute log_prob and entropy for a given action (for PPO update)."""
    mean, log_std = actor_forward(params, obs)
    # Inverse tanh to get raw action
    action_raw = jnp.arctanh(jnp.clip(action, -0.999, 0.999))
    log_prob = _gaussian_log_prob(action_raw, mean, log_std)
    log_prob = log_prob - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
    # Entropy of Gaussian (before tanh — approximation)
    entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
    return log_prob, entropy


# ═══════════════════════════════════════════════════════════════════════
# Adam Optimizer (pure JAX)
# ═══════════════════════════════════════════════════════════════════════

def init_adam(params, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
    """Initialize Adam optimizer state."""
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return {'m': m, 'v': v, 'step': 0, 'lr': lr,
            'beta1': beta1, 'beta2': beta2, 'eps': eps}


def adam_update(opt_state, params, grads, max_grad_norm=None):
    """Apply one Adam step with optional gradient clipping."""
    if max_grad_norm is not None:
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
        )
        clip_coef = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6))
        grads = jax.tree.map(lambda g: g * clip_coef, grads)

    step = opt_state['step'] + 1
    lr = opt_state['lr']
    b1, b2, eps = opt_state['beta1'], opt_state['beta2'], opt_state['eps']

    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt_state['m'], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2, opt_state['v'], grads)

    m_hat = jax.tree.map(lambda m: m / (1 - b1 ** step), m)
    v_hat = jax.tree.map(lambda v: v / (1 - b2 ** step), v)

    params = jax.tree.map(
        lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps),
        params, m_hat, v_hat,
    )
    return {**opt_state, 'm': m, 'v': v, 'step': step}, params


# ═══════════════════════════════════════════════════════════════════════
# Data saving utilities
# ═══════════════════════════════════════════════════════════════════════

def save_flow_snapshot_ppo(state, grid, label, output_dir):
    """Save flow field arrays (u, v, p, vorticity) as .npz."""
    u, v = np.array(state.velocity[0]), np.array(state.velocity[1])
    p = np.array(state.pressure)
    omega = np.array(vorticity_2d(state.velocity, grid))

    snap_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    xc, yc = grid.cell_centers()
    xu, yu = grid.face_centers(0)
    xv, yv = grid.face_centers(1)

    np.savez_compressed(
        os.path.join(snap_dir, f'flow_update_{label:04d}.npz'),
        u=u, v=v, p=p, vorticity=omega,
        xc=np.array(xc), yc=np.array(yc),
        xu=np.array(xu), yu=np.array(yu),
        xv=np.array(xv), yv=np.array(yv),
        label=label,
    )


def save_flow_image_ppo(state, grid, label, output_dir, cfg_body=None):
    """Save vorticity + pressure contour images with cylinder overlay."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    u, v = state.velocity
    omega = np.array(vorticity_2d((u, v), grid))
    p = np.array(state.pressure)

    xc, yc = grid.cell_centers()
    xc, yc = np.array(xc), np.array(yc)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), dpi=100)

    # Vorticity
    vmax = min(5.0, float(np.percentile(np.abs(omega), 99)))
    ax = axes[0]
    im = ax.pcolormesh(xc, yc, omega.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    if cfg_body:
        theta = np.linspace(0, 2*np.pi, 100)
        cx, cy, R = cfg_body['center'][0], cfg_body['center'][1], cfg_body['radius']
        ax.fill(cx + R*np.cos(theta), cy + R*np.sin(theta), 'white', zorder=5)
        ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), 'k-', lw=1.5, zorder=6)
    ax.set_xlim(0, float(xc[-1]))
    ax.set_ylim(float(yc[0]), float(yc[-1]))
    ax.set_aspect('equal')
    ax.set_title(f'Vorticity  (Update {label})')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Pressure
    ax = axes[1]
    pmax = float(np.percentile(np.abs(p), 98))
    if pmax < 1e-10:
        pmax = 1.0
    im = ax.pcolormesh(xc, yc, p.T, cmap='coolwarm', vmin=-pmax, vmax=pmax,
                       shading='auto')
    if cfg_body:
        ax.fill(cx + R*np.cos(theta), cy + R*np.sin(theta), 'white', zorder=5)
        ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), 'k-', lw=1.5, zorder=6)
    ax.set_xlim(0, float(xc[-1]))
    ax.set_ylim(float(yc[0]), float(yc[-1]))
    ax.set_aspect('equal')
    ax.set_title(f'Pressure  (Update {label})')
    plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'fields_update_{label:04d}.png'),
                bbox_inches='tight')
    plt.close(fig)


def save_ppo_training_curves(history, output_dir):
    """Generate training curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    updates = history['update']

    # Cd mean with envelope
    ax = axes[0, 0]
    ax.plot(updates, history['cd_mean'], 'b-', lw=1.5, label='Cd (mean)')
    ax.axhline(history['cd_baseline'], color='r', ls='--', lw=1, label='Cd baseline')
    if history.get('cd_min'):
        ax.fill_between(updates, history['cd_min'], history['cd_max'],
                        alpha=0.2, color='b', label='Cd (min-max)')
    ax.set_xlabel('Update')
    ax.set_ylabel('Cd')
    ax.set_title('Drag Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Policy loss
    ax = axes[0, 1]
    ax.plot(updates, history['policy_loss'], 'r-', lw=1.5)
    ax.set_xlabel('Update')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss')
    ax.grid(True, alpha=0.3)

    # Value loss
    ax = axes[1, 0]
    ax.plot(updates, history['value_loss'], 'g-', lw=1.5)
    ax.set_xlabel('Update')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss')
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 1]
    ax.plot(updates, history['entropy'], 'k-', lw=1.5)
    ax.set_xlabel('Update')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.grid(True, alpha=0.3)

    # Mean return
    ax = axes[2, 0]
    ax.plot(updates, history['mean_return'], 'purple', lw=1.5)
    ax.set_xlabel('Update')
    ax.set_ylabel('Return')
    ax.set_title('Mean Episode Return')
    ax.grid(True, alpha=0.3)

    # Drag reduction %
    ax = axes[2, 1]
    ax.plot(updates, history['reduction_pct'], 'b-', lw=1.5)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Update')
    ax.set_ylabel('Reduction (%)')
    ax.set_title('Drag Reduction')
    ax.grid(True, alpha=0.3)

    plt.suptitle('PPO Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_ppo_history(history, output_dir):
    """Save history JSON (crash-safe: write to tmp then rename)."""
    hist_path = os.path.join(output_dir, 'history.json')
    tmp_path = hist_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(history, f, indent=2, default=lambda x: float(x))
    os.replace(tmp_path, hist_path)


# ═══════════════════════════════════════════════════════════════════════
# GAE (Generalized Advantage Estimation)
# ═══════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages and returns.

    Parameters
    ----------
    rewards : (T,) array
    values : (T+1,) array — includes bootstrap value at end

    Returns
    -------
    advantages : (T,) array
    returns : (T,) array
    """
    T = rewards.shape[0]

    # Reverse scan
    def body(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        body, 0.0, jnp.arange(T - 1, -1, -1),
    )
    advantages = advantages[::-1]  # reverse back
    returns = advantages + values[:T]
    return advantages, returns


# ═══════════════════════════════════════════════════════════════════════
# PPO Loss
# ═══════════════════════════════════════════════════════════════════════

def ppo_loss(params, obs_batch, action_batch, old_log_probs, advantages,
             returns, clip_eps, vf_coef, ent_coef):
    """Clipped PPO loss = policy_loss + vf_coef * value_loss - ent_coef * entropy."""
    # Evaluate current policy
    log_probs, entropy = jax.vmap(evaluate_action, in_axes=(None, 0, 0))(
        params, obs_batch, action_batch,
    )
    values = jax.vmap(critic_forward, in_axes=(None, 0))(params, obs_batch)

    # Policy loss (clipped surrogate)
    ratio = jnp.exp(log_probs - old_log_probs)
    adv_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    # Value loss
    value_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
    entropy_mean = jnp.mean(entropy)

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    return total_loss, {'policy_loss': policy_loss, 'value_loss': value_loss,
                        'entropy': entropy_mean}


# ═══════════════════════════════════════════════════════════════════════
# Multi-Episode Rollout Collection (Rabault 2019)
# ═══════════════════════════════════════════════════════════════════════

def collect_multi_episode_rollout(env, params, state, n_episodes,
                                  episode_length, key, soft_reset_prob=0.8,
                                  gamma=0.99, gae_lambda=0.97):
    """Collect N episodes of data for PPO update.

    Rabault 2019: 20 episodes × 80 actions = 1600 transitions per update.
    GAE is computed per-episode (not across episode boundaries).
    """
    all_obs, all_actions = [], []
    all_log_probs = []
    all_advantages, all_returns = [], []
    all_cds, all_cls = [], []
    all_actions_scalar = []
    episode_returns = []

    for ep in range(n_episodes):
        # Reset (soft/hard)
        if ep == 0 and state is None:
            state, obs = env.reset()
        else:
            state, obs = env.reset(soft_prob=soft_reset_prob)

        ep_obs, ep_actions, ep_rewards = [], [], []
        ep_log_probs, ep_values = [], []
        ep_cds, ep_cls = [], []

        for t in range(episode_length):
            key, subkey = jax.random.split(key)

            ep_obs.append(obs)
            value = float(critic_forward(params, obs))
            ep_values.append(value)

            action, log_prob, _ = sample_action(params, obs, subkey)
            action_scalar = float(action[0])
            ep_actions.append(action)
            ep_log_probs.append(float(log_prob))

            state, obs, reward, info = env.step_no_scan(state, action_scalar)
            ep_rewards.append(float(reward))
            ep_cds.append(float(info['cd']))
            ep_cls.append(float(info['cl']))
            all_actions_scalar.append(action_scalar)

        # Bootstrap value for GAE
        bootstrap_value = float(critic_forward(params, obs))
        ep_values.append(bootstrap_value)

        # Compute GAE per-episode (critical: don't cross episode boundaries)
        ep_rewards_arr = jnp.array(ep_rewards)
        ep_values_arr = jnp.array(ep_values)
        ep_adv, ep_ret = compute_gae(
            ep_rewards_arr, ep_values_arr,
            gamma=gamma, lam=gae_lambda,
        )

        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        all_log_probs.extend(ep_log_probs)
        all_advantages.extend(ep_adv.tolist())
        all_returns.extend(ep_ret.tolist())
        all_cds.extend(ep_cds)
        all_cls.extend(ep_cls)
        episode_returns.append(float(jnp.sum(ep_rewards_arr)))

        # Update env state for soft reset
        env._current_state = state

        ep_cd_mean = np.mean(ep_cds)
        print(f"    ep {ep+1}/{n_episodes}: "
              f"Cd={ep_cd_mean:.4f}, return={episode_returns[-1]:+.4f}")

    batch = {
        'obs': jnp.stack(all_obs),
        'actions': jnp.stack(all_actions),
        'log_probs': jnp.array(all_log_probs),
        'advantages': jnp.array(all_advantages),
        'returns': jnp.array(all_returns),
        'cds': all_cds,
        'cls': all_cls,
        'actions_scalar': all_actions_scalar,
        'episode_returns': episode_returns,
    }
    return state, batch


# ═══════════════════════════════════════════════════════════════════════
# PPO Update (Rabault 2019: 25 epochs × 20% subsampling)
# ═══════════════════════════════════════════════════════════════════════

def ppo_update(params, opt_state, batch, ppo_cfg):
    """Run PPO update: multiple epochs with random subsampling."""
    n_epochs = ppo_cfg['n_epochs']
    subsample_ratio = ppo_cfg.get('subsample_ratio', 0.2)
    clip_eps = ppo_cfg['clip_eps']
    vf_coef = ppo_cfg['vf_coef']
    ent_coef = ppo_cfg['ent_coef']
    max_grad_norm = ppo_cfg['max_grad_norm']

    obs = batch['obs']
    actions = batch['actions']
    old_log_probs = batch['log_probs']
    advantages = batch['advantages']
    returns = batch['returns']
    T = obs.shape[0]
    subsample_size = max(int(T * subsample_ratio), 1)

    total_loss_sum = 0.0
    info_accum = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    n_updates = 0

    loss_and_grad = jax.value_and_grad(ppo_loss, has_aux=True)

    for epoch in range(n_epochs):
        # Random 20% subsample per epoch
        idx = np.random.choice(T, size=subsample_size, replace=False)

        mb_obs = obs[idx]
        mb_actions = actions[idx]
        mb_old_lp = old_log_probs[idx]
        mb_adv = advantages[idx]
        mb_ret = returns[idx]

        (loss, info), grads = loss_and_grad(
            params, mb_obs, mb_actions, mb_old_lp,
            mb_adv, mb_ret, clip_eps, vf_coef, ent_coef,
        )
        opt_state, params = adam_update(opt_state, params, grads, max_grad_norm)

        total_loss_sum += float(loss)
        for k in info_accum:
            info_accum[k] += float(info[k])
        n_updates += 1

    avg_info = {k: v / max(n_updates, 1) for k, v in info_accum.items()}
    avg_info['total_loss'] = total_loss_sum / max(n_updates, 1)
    return params, opt_state, avg_info


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(env_cfg, ppo_cfg):
    """Run PPO training loop (Rabault 2019 matching)."""
    output_dir = ppo_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    seed = ppo_cfg['seed']
    key = jax.random.PRNGKey(seed)

    # Set activation function
    set_activations(ppo_cfg.get('activation', 'relu'))

    # ── Initialize environment ──
    print("=" * 60)
    print("PPO — 2D Cylinder AFC (Rabault 2019)")
    print("=" * 60)
    env = CylinderFlowEnv(env_cfg)

    key, reset_key = jax.random.split(key)
    state, obs = env.reset(reset_key)

    obs_dim = env.obs_size
    action_dim = env.action_size
    soft_reset_prob = env_cfg.get('soft_reset_prob', 0.8)
    n_updates = ppo_cfg['n_updates']
    episodes_per_update = ppo_cfg['episodes_per_update']
    episode_length = ppo_cfg['episode_length']

    print(f"obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"obs_type={env_cfg.get('obs_type', 'velocity')}, "
          f"probe_layout={env_cfg.get('probe_layout', 'grid')}")
    print(f"Cd_baseline={env.cd_baseline:.4f}")
    print(f"jet_method={env.jet_method}, cl_weight={env.cl_weight}, "
          f"ema_alpha={env.ema_alpha}")
    print(f"Training: {n_updates} updates × {episodes_per_update} episodes "
          f"× {episode_length} actions = "
          f"{n_updates * episodes_per_update} total episodes")

    # ── Initialize policy (separate actor/critic) ──
    key, init_key = jax.random.split(key)
    params = init_actor_critic(
        init_key, obs_dim, action_dim,
        actor_hidden_dims=ppo_cfg['actor_hidden_dims'],
        critic_hidden_dims=ppo_cfg['critic_hidden_dims'],
    )
    opt_state = init_adam(params, lr=ppo_cfg['lr'])

    actor_size = sum(W.size + b.size for W, b in params['actor'])
    critic_size = sum(W.size + b.size for W, b in params['critic'])
    print(f"Actor: {ppo_cfg['actor_hidden_dims']} ({actor_size:,} params)")
    print(f"Critic: {ppo_cfg['critic_hidden_dims']} ({critic_size:,} params)")
    print("-" * 60)

    # ── Training loop ──
    snapshot_every = ppo_cfg.get('snapshot_every', 10)
    cfg_body = {'center': [env_cfg['cx'], env_cfg['cy']], 'radius': env_cfg['R']}

    history = {
        # Metadata
        'cd_baseline': float(env.cd_baseline),
        'env_config': {k: (float(v) if isinstance(v, (int, float)) else
                          [float(x) for x in v] if isinstance(v, (list, tuple)) else str(v))
                       for k, v in env_cfg.items()},
        'ppo_config': {k: (float(v) if isinstance(v, (int, float)) else
                          [float(x) for x in v] if isinstance(v, (list, tuple)) else str(v))
                       for k, v in ppo_cfg.items()},
        # Per-update scalars
        'update': [],
        'total_episodes': [],
        'cd_mean': [], 'cd_min': [], 'cd_max': [],
        'cl_mean': [],
        'mean_return': [],
        'reduction_pct': [],
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'time': [],
        'cumulative_time': [],
        'total_cfd_steps': [],
        # Per-step data (list of lists)
        'cds_per_step': [],
        'cls_per_step': [],
        'actions_per_step': [],
        'episode_returns': [],
    }
    best_cd = float('inf')
    best_update = 0
    t_start = time.time()

    # Save baseline flow snapshot
    save_flow_snapshot_ppo(state, env.grid, label=-1, output_dir=output_dir)
    save_flow_image_ppo(state, env.grid, label=-1, output_dir=output_dir,
                        cfg_body=cfg_body)

    for update in range(n_updates):
        t_update = time.time()

        # Collect multi-episode rollout
        key, rollout_key = jax.random.split(key)
        print(f"\n[Update {update+1}/{n_updates}] "
              f"Collecting {episodes_per_update} episodes...")
        state, batch = collect_multi_episode_rollout(
            env, params, state,
            n_episodes=episodes_per_update,
            episode_length=episode_length,
            key=rollout_key,
            soft_reset_prob=soft_reset_prob,
            gamma=ppo_cfg['gamma'],
            gae_lambda=ppo_cfg['gae_lambda'],
        )

        # PPO update
        params, opt_state, update_info = ppo_update(
            params, opt_state, batch, ppo_cfg,
        )

        # Logging
        cds_arr = np.array(batch['cds'])
        cls_arr = np.array(batch['cls'])
        mean_cd = float(np.mean(cds_arr))
        cd_min = float(np.min(cds_arr))
        cd_max = float(np.max(cds_arr))
        mean_cl = float(np.mean(np.abs(cls_arr)))
        mean_return = float(np.mean(batch['episode_returns']))
        dt_update = time.time() - t_update
        cumulative = time.time() - t_start
        total_episodes = (update + 1) * episodes_per_update
        total_cfd = total_episodes * episode_length * env.action_interval

        reduction = (mean_cd - env.cd_baseline) / env.cd_baseline * 100

        history['update'].append(update + 1)
        history['total_episodes'].append(total_episodes)
        history['cd_mean'].append(mean_cd)
        history['cd_min'].append(cd_min)
        history['cd_max'].append(cd_max)
        history['cl_mean'].append(mean_cl)
        history['mean_return'].append(mean_return)
        history['reduction_pct'].append(reduction)
        history['policy_loss'].append(float(update_info['policy_loss']))
        history['value_loss'].append(float(update_info['value_loss']))
        history['entropy'].append(float(update_info['entropy']))
        history['time'].append(dt_update)
        history['cumulative_time'].append(cumulative)
        history['total_cfd_steps'].append(total_cfd)
        history['cds_per_step'].append(batch['cds'])
        history['cls_per_step'].append(batch['cls'])
        history['actions_per_step'].append(batch['actions_scalar'])
        history['episode_returns'].append(batch['episode_returns'])

        # Best model checkpoint
        if mean_cd < best_cd:
            best_cd = mean_cd
            best_update = update + 1
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            flat_params = {f'{k}_{i}_{j}': np.array(v)
                          for k in params
                          for i, layer in enumerate(params[k])
                          for j, v in enumerate(layer)}
            np.savez(os.path.join(ckpt_dir, 'best_params.npz'),
                     update=update+1, cd=mean_cd, **flat_params)

        print(f"  >> Cd={mean_cd:.4f} ({reduction:+.1f}%), "
              f"|Cl|={mean_cl:.4f}, return={mean_return:+.4f}, "
              f"ent={update_info['entropy']:.3f}, time={dt_update:.1f}s")

        # Save checkpoint
        if (update + 1) % ppo_cfg.get('save_every', 10) == 0:
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'ckpt_{update+1:04d}.npz')
            flat_params = {f'{k}_{i}_{j}': np.array(v)
                          for k in params
                          for i, layer in enumerate(params[k])
                          for j, v in enumerate(layer)}
            np.savez(ckpt_path, **flat_params)

        # Save flow snapshot + image
        if (update + 1) % snapshot_every == 0 or update == n_updates - 1:
            save_flow_snapshot_ppo(state, env.grid, update+1, output_dir)
            save_flow_image_ppo(state, env.grid, update+1, output_dir,
                                cfg_body=cfg_body)
            print(f"  [Saved flow snapshot + image: update {update+1}]")

        # Crash-safe incremental history save
        if (update + 1) % 5 == 0 or update == n_updates - 1:
            _save_ppo_history(history, output_dir)

    # ── Save final results ──
    t_total = time.time() - t_start
    total_cfd_final = n_updates * episodes_per_update * episode_length * env.action_interval
    best_reduction = (best_cd - env.cd_baseline) / env.cd_baseline * 100
    print(f"\n{'=' * 70}")
    print(f"PPO Training complete!")
    print(f"  Cd baseline:     {env.cd_baseline:.4f}")
    print(f"  Best Cd:         {best_cd:.4f}  (update {best_update})")
    print(f"  Drag reduction:  {best_reduction:+.1f}%")
    print(f"  Total episodes:  {n_updates * episodes_per_update}")
    print(f"  Total CFD steps: {total_cfd_final:,}")
    print(f"  Total time:      {t_total:.1f}s ({t_total/3600:.1f}h)")
    print(f"{'=' * 70}")

    # Final saves
    _save_ppo_history(history, output_dir)

    # Final params
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    flat_params = {f'{k}_{i}_{j}': np.array(v)
                  for k in params
                  for i, layer in enumerate(params[k])
                  for j, v in enumerate(layer)}
    np.savez(os.path.join(ckpt_dir, 'final_params.npz'), **flat_params)

    # Training curves
    save_ppo_training_curves(history, output_dir)
    print(f"  Training curves: {output_dir}/training_curves.png")
    print(f"  History:         {output_dir}/history.json")

    return params, history


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO for cylinder AFC')
    parser.add_argument('--test', action='store_true',
                        help='Quick test with reduced grid/iterations')
    parser.add_argument('--re', type=int, default=100,
                        choices=[100, 200, 500],
                        help='Reynolds number (default: 100)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed (appends _s{seed} to output_dir)')
    args = parser.parse_args()

    if args.test:
        print("[TEST MODE] Using reduced config")
        env_cfg, ppo_cfg = TEST_ENV_CONFIG, TEST_PPO_CONFIG
    elif args.re == 200:
        env_cfg, ppo_cfg = ENV_CONFIG_RE200, PPO_CONFIG_RE200
    elif args.re == 500:
        env_cfg, ppo_cfg = ENV_CONFIG_RE500, PPO_CONFIG_RE500
    else:
        env_cfg, ppo_cfg = ENV_CONFIG, PPO_CONFIG

    if args.seed is not None:
        ppo_cfg = {**ppo_cfg, 'seed': args.seed,
                   'output_dir': ppo_cfg['output_dir'] + f'_s{args.seed}'}

    train(env_cfg, ppo_cfg)
