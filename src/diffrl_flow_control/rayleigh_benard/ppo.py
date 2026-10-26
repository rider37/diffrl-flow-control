"""PPO for Rayleigh--Benard convection, following the setup of Markmann et al.
(2025): separate actor and critic networks with ReLU activations, a Gaussian
policy squashed by tanh, several episodes per update from ICs of the training
pool with GAE, and clipped-surrogate updates.  Optional running normalisation of
the observations (``obs_norm``, Section 5.4) is stored with the checkpoints.
Writes history.json, checkpoints/{update_XXXX,best,final}.npz and flow snapshots
to ``ppo_cfg['output_dir']``.
"""
import json
import os
import time


import jax
import jax.numpy as jnp
import numpy as np

from diffrl_flow_control.rayleigh_benard.env import RBControlEnv


# ═══════════════════════════════════════════════════════════════════════
# Running observation normalization (opt-in; obs only, reward untouched)
# ═══════════════════════════════════════════════════════════════════════
# Welford running mean/std applied to observations before the policy/value
# nets, matching standard PPO hygiene (Stable-Baselines3 VecNormalize,
# norm_obs=True / norm_reward=False).  Statistics are updated on raw obs
# and treated as constants (no gradient).  Enabled only when ppo_cfg
# sets 'obs_norm': True; default-off keeps existing runs byte-identical.

class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = float(eps)

    def update(self, x):
        x = np.asarray(x, np.float64)
        delta = x - self.mean
        tot = self.count + 1.0
        self.mean = self.mean + delta / tot
        m2 = self.var * self.count + (delta * delta) * self.count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        n = (np.asarray(x, np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(n, -clip, clip).astype(np.float32)

    def state(self):
        return {'obs_rms_mean': self.mean.astype(np.float64),
                'obs_rms_var': self.var.astype(np.float64),
                'obs_rms_count': np.array(self.count, np.float64)}


# ═══════════════════════════════════════════════════════════════════════
# MLP (pure JAX)
# ═══════════════════════════════════════════════════════════════════════

def init_mlp(key, layer_dims):
    params = []
    for i in range(len(layer_dims) - 1):
        key, subkey = jax.random.split(key)
        fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        W = jax.random.normal(subkey, (fan_in, fan_out)) * scale
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def mlp_forward(params, x):
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


# ═══════════════════════════════════════════════════════════════════════
# Actor-Critic
# ═══════════════════════════════════════════════════════════════════════

def init_actor_critic(key, obs_dim, action_dim,
                      actor_hidden_dims, critic_hidden_dims):
    key_a, key_c = jax.random.split(key)
    actor_dims = [obs_dim] + list(actor_hidden_dims) + [2 * action_dim]
    critic_dims = [obs_dim] + list(critic_hidden_dims) + [1]
    actor_params = init_mlp(key_a, actor_dims)
    critic_params = init_mlp(key_c, critic_dims)
    # Initial log_std = -0.5
    W_last, b_last = actor_params[-1]
    b_last = b_last.at[action_dim:].set(-0.5)
    actor_params[-1] = (W_last, b_last)
    return {'actor': actor_params, 'critic': critic_params}


def actor_forward(params, obs):
    out = mlp_forward(params['actor'], obs)
    action_dim = out.shape[-1] // 2
    mean = out[..., :action_dim]
    log_std = jnp.clip(out[..., action_dim:], -2.0, 0.5)
    return mean, log_std


def critic_forward(params, obs):
    return mlp_forward(params['critic'], obs).squeeze(-1)


def sample_action(params, obs, key):
    mean, log_std = actor_forward(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    action_raw = mean + std * noise
    action = jnp.tanh(action_raw)
    log_prob = _gaussian_log_prob(action_raw, mean, log_std)
    log_prob = log_prob - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
    return action, log_prob, mean, action_raw


def _gaussian_log_prob(x, mean, log_std):
    std = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * ((x - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


def evaluate_action(params, obs, action_raw):
    mean, log_std = actor_forward(params, obs)
    # Log-probability of the stored pre-squash sample, so that the PPO ratio is exactly 1 at
    # theta = theta_old.  (Reconstructing it by inverse tanh of the clipped squashed action
    # differed from the sample for |u| > 3.8 and drove the ratio, and with it the
    # clipped-objective gradient, to zero once the policy saturated.)
    action = jnp.tanh(action_raw)
    log_prob = _gaussian_log_prob(action_raw, mean, log_std)
    log_prob = log_prob - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
    entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
    return log_prob, entropy


# ═══════════════════════════════════════════════════════════════════════
# Adam Optimizer
# ═══════════════════════════════════════════════════════════════════════

def init_adam(params, lr=3e-4):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return {'m': m, 'v': v, 'step': 0, 'lr': lr,
            'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}


def adam_update(opt_state, params, grads, max_grad_norm=None):
    if max_grad_norm is not None:
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        clip_coef = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6))
        grads = jax.tree.map(lambda g: g * clip_coef, grads)
    step = opt_state['step'] + 1
    lr, b1, b2, eps = opt_state['lr'], opt_state['beta1'], opt_state['beta2'], opt_state['eps']
    m = jax.tree.map(lambda m, g: b1*m + (1-b1)*g, opt_state['m'], grads)
    v = jax.tree.map(lambda v, g: b2*v + (1-b2)*g**2, opt_state['v'], grads)
    m_hat = jax.tree.map(lambda m: m / (1-b1**step), m)
    v_hat = jax.tree.map(lambda v: v / (1-b2**step), v)
    params = jax.tree.map(lambda p, m, v: p - lr*m/(jnp.sqrt(v)+eps), params, m_hat, v_hat)
    return {**opt_state, 'm': m, 'v': v, 'step': step}, params


# ═══════════════════════════════════════════════════════════════════════
# Data Saving
# ═══════════════════════════════════════════════════════════════════════

def save_flow_snapshot(state, grid, label, output_dir):
    snap_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    T = np.array(state.scalars['T'])
    u, v = np.array(state.velocity[0]), np.array(state.velocity[1])
    p = np.array(state.pressure)
    xc, yc = grid.cell_centers()
    np.savez_compressed(
        os.path.join(snap_dir, f'flow_update_{label:04d}.npz'),
        T=T, u=u, v=v, p=p,
        xc=np.array(xc), yc=np.array(yc), label=label,
    )


def _save_history(history, output_dir):
    hist_path = os.path.join(output_dir, 'history.json')
    tmp_path = hist_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(history, f, indent=2, default=lambda x: float(x))
    os.replace(tmp_path, hist_path)


# ═══════════════════════════════════════════════════════════════════════
# GAE
# ═══════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    def body(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        return gae, gae
    _, advantages = jax.lax.scan(body, 0.0, jnp.arange(T-1, -1, -1))
    advantages = advantages[::-1]
    returns = advantages + values[:T]
    return advantages, returns


# ═══════════════════════════════════════════════════════════════════════
# PPO Loss
# ═══════════════════════════════════════════════════════════════════════

def ppo_loss(params, obs_batch, action_batch, old_log_probs, advantages,
             returns, clip_eps, vf_coef, ent_coef):
    log_probs, entropy = jax.vmap(evaluate_action, in_axes=(None, 0, 0))(
        params, obs_batch, action_batch)
    values = jax.vmap(critic_forward, in_axes=(None, 0))(params, obs_batch)

    ratio = jnp.exp(log_probs - old_log_probs)
    adv_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    value_loss = jnp.mean((values - returns) ** 2)
    entropy_mean = jnp.mean(entropy)

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    return total_loss, {'policy_loss': policy_loss, 'value_loss': value_loss,
                        'entropy': entropy_mean}


# ═══════════════════════════════════════════════════════════════════════
# Rollout Collection
# ═══════════════════════════════════════════════════════════════════════

def collect_rollout(env, params, state, n_episodes, episode_length, key,
                    soft_reset_prob=0.0, gamma=0.99, gae_lambda=0.95,
                    ic_pool=None, ic_rng=None, obs_rms=None):
    """Collect n_episodes rollouts.

    With ``ic_pool`` each episode starts from a random IC of the pool;
    otherwise from the reference IC with soft/hard resets.

    With ``obs_rms`` the observations are normalised by running statistics
    (updated on the raw observations); without it the raw observations are used.
    """
    def _norm(o):
        if obs_rms is None:
            return o
        obs_rms.update(o)
        return jnp.asarray(obs_rms.normalize(o))

    all_obs, all_actions, all_log_probs = [], [], []
    all_advantages, all_returns = [], []
    all_nus, all_actions_raw = [], []
    episode_returns = []
    episode_ic_seeds = []

    for ep in range(n_episodes):
        if ic_pool is not None:
            # Multi-IC: sample random seed from pool every episode
            ic_seed = int(ic_rng.choice(ic_pool))
            state, obs = env.reset(ic_seed=ic_seed)
            episode_ic_seeds.append(ic_seed)
        elif ep == 0 and state is None:
            state, obs = env.reset()
            episode_ic_seeds.append(None)
        else:
            state, obs = env.reset(soft_prob=soft_reset_prob)
            episode_ic_seeds.append(None)

        obs = _norm(obs)
        ep_obs, ep_actions, ep_rewards = [], [], []
        ep_log_probs, ep_values = [], []
        ep_nus, ep_actions_raw = [], []

        for t in range(episode_length):
            key, subkey = jax.random.split(key)
            ep_obs.append(obs)
            value = float(critic_forward(params, obs))
            ep_values.append(value)

            action, log_prob, _, action_raw = sample_action(params, obs, subkey)
            action_np = np.array(action)
            ep_actions.append(action_raw)          # pre-squash sample, used by evaluate_action
            ep_log_probs.append(float(log_prob))

            state, obs, reward, info = env.step_no_scan(state, action_np)
            obs = _norm(obs)
            ep_rewards.append(float(reward))
            ep_nus.append(float(info['Nu']))
            ep_actions_raw.append(action_np.tolist())

        bootstrap_value = float(critic_forward(params, obs))
        ep_values.append(bootstrap_value)

        ep_rewards_arr = jnp.array(ep_rewards)
        ep_values_arr = jnp.array(ep_values)
        ep_adv, ep_ret = compute_gae(ep_rewards_arr, ep_values_arr,
                                     gamma=gamma, lam=gae_lambda)

        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        all_log_probs.extend(ep_log_probs)
        all_advantages.extend(ep_adv.tolist())
        all_returns.extend(ep_ret.tolist())
        all_nus.extend(ep_nus)
        all_actions_raw.extend(ep_actions_raw)
        episode_returns.append(float(jnp.sum(ep_rewards_arr)))

        env._current_state = state
        ep_nu_mean = np.mean(ep_nus)
        ic_suffix = f" [ic={episode_ic_seeds[-1]}]" if episode_ic_seeds[-1] is not None else ""
        print(f"    ep {ep+1}/{n_episodes}: Nu={ep_nu_mean:.3f}, "
              f"return={episode_returns[-1]:+.4f}{ic_suffix}")

    batch = {
        'obs': jnp.stack(all_obs),
        'actions': jnp.stack(all_actions),
        'log_probs': jnp.array(all_log_probs),
        'advantages': jnp.array(all_advantages),
        'returns': jnp.array(all_returns),
        'nus': all_nus,
        'actions_raw': all_actions_raw,
        'episode_returns': episode_returns,
        'episode_ic_seeds': episode_ic_seeds,
    }
    return state, batch


# ═══════════════════════════════════════════════════════════════════════
# PPO Update
# ═══════════════════════════════════════════════════════════════════════

def ppo_update(params, opt_state, batch, ppo_cfg):
    n_epochs = ppo_cfg['n_epochs']
    subsample_ratio = ppo_cfg.get('subsample_ratio', 1.0)
    clip_eps = ppo_cfg['clip_eps']
    vf_coef, ent_coef = ppo_cfg['vf_coef'], ppo_cfg['ent_coef']
    max_grad_norm = ppo_cfg['max_grad_norm']

    obs, actions = batch['obs'], batch['actions']
    old_log_probs, advantages, returns = batch['log_probs'], batch['advantages'], batch['returns']
    T = obs.shape[0]
    subsample_size = max(int(T * subsample_ratio), 1)

    total_loss_sum = 0.0
    info_accum = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    n_updates = 0
    loss_and_grad = jax.value_and_grad(ppo_loss, has_aux=True)

    for epoch in range(n_epochs):
        idx = np.random.choice(T, size=subsample_size, replace=False)
        (loss, info), grads = loss_and_grad(
            params, obs[idx], actions[idx], old_log_probs[idx],
            advantages[idx], returns[idx], clip_eps, vf_coef, ent_coef)
        opt_state, params = adam_update(opt_state, params, grads, max_grad_norm)
        total_loss_sum += float(loss)
        for k in info_accum:
            info_accum[k] += float(info[k])
        n_updates += 1

    avg_info = {k: v / max(n_updates, 1) for k, v in info_accum.items()}
    return params, opt_state, avg_info


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(env_cfg, ppo_cfg):
    assert ppo_cfg.get('activation', 'relu') == 'relu', 'the actor and critic use ReLU activations'
    output_dir = ppo_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    seed = ppo_cfg['seed']
    key = jax.random.PRNGKey(seed)

    # ── Initialize environment ──
    print("=" * 60)
    print("PPO — 2D Rayleigh-Benard AFC (Markmann 2025)")
    print("=" * 60)
    env = RBControlEnv(env_cfg)

    # ── Multi-IC setup (optional) ──
    ic_pool_seeds = ppo_cfg.get('ic_pool_seeds', None)  # e.g. list(range(15))
    ic_rng = None
    if ic_pool_seeds is not None:
        print(f"[multi-IC] Preloading train pool: {ic_pool_seeds}")
        env.preload_ic_pool(ic_pool_seeds, verbose=True)
        ic_rng = np.random.default_rng(ppo_cfg['seed'])
        # Reset to one pool entry for initial state
        state, obs = env.reset(ic_seed=int(ic_rng.choice(ic_pool_seeds)))
    else:
        key, reset_key = jax.random.split(key)
        state, obs = env.reset(reset_key)

    obs_dim = env.obs_size
    action_dim = env.action_size
    obs_norm = ppo_cfg.get('obs_norm', False)
    obs_rms = RunningMeanStd((obs_dim,)) if obs_norm else None
    if obs_norm:
        print("[obs-norm] running observation normalization ENABLED "
              "(obs only; reward untouched)")
    n_updates = ppo_cfg['n_updates']
    episodes_per_update = ppo_cfg['episodes_per_update']
    episode_length = ppo_cfg['episode_length']

    print(f"obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"Ra={env.Ra:.0f}, Pr={env.Pr}, Nu_baseline={env.nu_baseline:.4f}"
          + (f" (pool mean)" if ic_pool_seeds is not None else ""))
    print(f"Training: {n_updates} updates x {episodes_per_update} episodes "
          f"x {episode_length} actions"
          + (f" [multi-IC: {len(ic_pool_seeds)} train ICs]" if ic_pool_seeds else ""))

    # ── Initialize policy ──
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

    # ── History ──
    snapshot_every = ppo_cfg.get('snapshot_every', 10)
    history = {
        'nu_baseline': float(env.nu_baseline),
        'env_config': {k: (float(v) if isinstance(v, (int, float)) else str(v))
                       for k, v in env_cfg.items()},
        'ppo_config': {k: (float(v) if isinstance(v, (int, float)) else
                          [float(x) for x in v] if isinstance(v, (list, tuple)) else str(v))
                       for k, v in ppo_cfg.items()},
        'update': [], 'total_episodes': [],
        'nu_mean': [], 'nu_min': [], 'nu_max': [],
        'mean_return': [], 'reduction_pct': [],
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'time': [], 'cumulative_time': [], 'total_cfd_steps': [],
        'nus_per_step': [], 'actions_per_step': [], 'episode_returns': [],
    }
    best_nu = float('inf')
    best_update = 0
    t_start = time.time()

    save_flow_snapshot(state, env.grid, label=-1, output_dir=output_dir)

    for update in range(n_updates):
        t_update = time.time()

        key, rollout_key = jax.random.split(key)
        print(f"\n[Update {update+1}/{n_updates}] "
              f"Collecting {episodes_per_update} episodes...")
        state, batch = collect_rollout(
            env, params, state,
            n_episodes=episodes_per_update,
            episode_length=episode_length,
            key=rollout_key,
            gamma=ppo_cfg['gamma'],
            gae_lambda=ppo_cfg['gae_lambda'],
            ic_pool=ic_pool_seeds,
            ic_rng=ic_rng,
            obs_rms=obs_rms,
        )

        params, opt_state, update_info = ppo_update(params, opt_state, batch, ppo_cfg)

        # Logging
        nus_arr = np.array(batch['nus'])
        mean_nu = float(np.mean(nus_arr))
        nu_min, nu_max = float(np.min(nus_arr)), float(np.max(nus_arr))
        mean_return = float(np.mean(batch['episode_returns']))
        dt_update = time.time() - t_update
        cumulative = time.time() - t_start
        total_episodes = (update + 1) * episodes_per_update
        total_cfd = total_episodes * episode_length * env.action_interval
        reduction = (1.0 - mean_nu / env.nu_baseline) * 100

        history['update'].append(update + 1)
        history['total_episodes'].append(total_episodes)
        history['nu_mean'].append(mean_nu)
        history['nu_min'].append(nu_min)
        history['nu_max'].append(nu_max)
        history['mean_return'].append(mean_return)
        history['reduction_pct'].append(reduction)
        history['policy_loss'].append(float(update_info['policy_loss']))
        history['value_loss'].append(float(update_info['value_loss']))
        history['entropy'].append(float(update_info['entropy']))
        history['time'].append(dt_update)
        history['cumulative_time'].append(cumulative)
        history['total_cfd_steps'].append(total_cfd)
        history['nus_per_step'].append(batch['nus'])
        history['actions_per_step'].append(batch['actions_raw'])
        history['episode_returns'].append(batch['episode_returns'])

        if mean_nu < best_nu:
            best_nu = mean_nu
            best_update = update + 1
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            flat_params = {f'{k}_{i}_{j}': np.array(v)
                          for k in params for i, layer in enumerate(params[k])
                          for j, v in enumerate(layer)}
            np.savez(os.path.join(ckpt_dir, 'best.npz'),
                     update=update+1, nu=mean_nu, **flat_params,
                     **(obs_rms.state() if obs_rms else {}))

        print(f"  >> Nu={mean_nu:.3f} ({reduction:+.1f}%), "
              f"return={mean_return:+.4f}, ent={update_info['entropy']:.3f}, "
              f"time={dt_update:.1f}s")

        if (update + 1) % ppo_cfg.get('save_every', 10) == 0:
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            flat_params = {f'{k}_{i}_{j}': np.array(v)
                          for k in params for i, layer in enumerate(params[k])
                          for j, v in enumerate(layer)}
            np.savez(os.path.join(ckpt_dir, f'update_{update+1:04d}.npz'), **flat_params,
                     **(obs_rms.state() if obs_rms else {}))

        if (update + 1) % snapshot_every == 0 or update == n_updates - 1:
            save_flow_snapshot(state, env.grid, update+1, output_dir)

        if (update + 1) % 5 == 0 or update == n_updates - 1:
            _save_history(history, output_dir)

    # ── Final ──
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    flat_params = {f'{k}_{i}_{j}': np.array(v)
                  for k in params for i, layer in enumerate(params[k])
                  for j, v in enumerate(layer)}
    np.savez(os.path.join(ckpt_dir, 'final.npz'), **flat_params,
             **(obs_rms.state() if obs_rms else {}))
    t_total = time.time() - t_start
    best_reduction = (1.0 - best_nu / env.nu_baseline) * 100
    print(f"\n{'='*70}")
    print(f"PPO Training complete!")
    print(f"  Nu baseline:     {env.nu_baseline:.4f}")
    print(f"  Best Nu:         {best_nu:.4f}  (update {best_update})")
    print(f"  Nu reduction:    {best_reduction:+.1f}%")
    print(f"  Total time:      {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"{'='*70}")

    _save_history(history, output_dir)
    return params, history
