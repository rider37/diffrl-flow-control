"""
FANS training for 2D Rayleigh-Benard Active Flow Control.

First-order Analytic Navier-Stokes (FANS) = SHAC on DiffFlow.
Computes exact policy gradients by backpropagating through the CFD simulator.

Saves comprehensive data for post-analysis:
  - Per-epoch: Nu, actor/critic loss, grad norm, actions, rewards
  - Flow snapshots: T, u, v, p fields at configurable intervals
  - Checkpoints: actor/critic parameters
  - History: full JSON log

Usage
-----
    # Full training
    python experiments/rb_control/train_fans.py

    # Quick test (small grid, few epochs)
    python experiments/rb_control/train_fans.py --test
"""

import argparse
import json
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from jax_cfd.envs.rb_control_env import RBControlEnv

from experiments.rb_control.config import (
    ENV_CONFIG, FANS_CONFIG, FANS_V2_CONFIG,
    TEST_ENV_CONFIG, TEST_FANS_CONFIG,
    ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5,
    ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6,
)
from experiments.rb_control.train_ppo import (
    init_actor_critic, critic_forward,
    init_adam, adam_update,
    save_flow_snapshot, save_flow_image,
)
from experiments.rb_control.fans import (
    make_rb_fans_step_fn, make_rb_get_obs_fn, make_compute_nu_fn,
    make_map_action_fn,
    fans_actor_loss, fans_critic_loss_batched,
    compute_td_targets,
)


# ═══════════════════════════════════════════════════════════════════════
# Data saving utilities
# ═══════════════════════════════════════════════════════════════════════

def save_params(params, epoch, output_dir, prefix='fans'):
    """Save actor + critic parameters as .npz."""
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    flat_actor = {f'actor_{i}_{k}': np.array(v)
                  for i, (W, b) in enumerate(params['actor'])
                  for k, v in [('W', W), ('b', b)]}
    flat_critic = {f'critic_{i}_{k}': np.array(v)
                   for i, (W, b) in enumerate(params['critic'])
                   for k, v in [('W', W), ('b', b)]}
    path = os.path.join(ckpt_dir, f'{prefix}_epoch_{epoch:04d}.npz')
    np.savez(path, **flat_actor, **flat_critic)
    return path


def save_training_curves(history, output_dir):
    """Generate training curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    epochs = history['epoch']

    # Nu mean
    ax = axes[0, 0]
    ax.plot(epochs, history['nu_mean'], 'b-', lw=1.5, label='Nu (mean)')
    ax.axhline(history['nu_baseline'], color='r', ls='--', lw=1, label='Nu baseline')
    if history.get('nu_min'):
        ax.fill_between(epochs, history['nu_min'], history['nu_max'],
                        alpha=0.2, color='b')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Nu')
    ax.set_title('Nusselt Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Actor loss
    ax = axes[0, 1]
    ax.plot(epochs, history['actor_loss'], 'r-', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Actor Loss')
    ax.grid(True, alpha=0.3)

    # Critic loss
    ax = axes[1, 0]
    ax.plot(epochs, history['critic_loss'], 'g-', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Critic Loss')
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    ax.semilogy(epochs, history['grad_norm'], 'k-', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|grad|')
    ax.set_title('Actor Gradient Norm')
    ax.grid(True, alpha=0.3)

    # Actions (per-segment mean magnitude)
    ax = axes[2, 0]
    ax.plot(epochs, history['action_mean'], 'b-', lw=1.5, label='|a| mean')
    if history.get('action_std'):
        ax.fill_between(
            epochs,
            np.array(history['action_mean']) - np.array(history['action_std']),
            np.array(history['action_mean']) + np.array(history['action_std']),
            alpha=0.2,
        )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|Action|')
    ax.set_title('Action Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Discounted return
    ax = axes[2, 1]
    ax.plot(epochs, history['return'], 'purple', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Return')
    ax.set_title('Discounted Return')
    ax.grid(True, alpha=0.3)

    plt.suptitle('FANS Training Curves (RB Control)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_history(history, output_dir):
    """Save history JSON (crash-safe: write to tmp then rename)."""
    hist_path = os.path.join(output_dir, 'history.json')
    tmp_path = hist_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(history, f, indent=2, default=lambda x: float(x))
    os.replace(tmp_path, hist_path)


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train(env_cfg, fans_cfg):
    """FANS training loop with comprehensive data logging."""
    seed = fans_cfg['seed']
    key = jax.random.PRNGKey(seed)
    output_dir = fans_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    snapshot_every = fans_cfg.get('snapshot_every', 25)
    reset_every = fans_cfg.get('reset_every', 0)
    action_penalty_coef = fans_cfg.get('action_penalty', 0.0)

    # ── 1. Environment setup (warmup + baseline) ──────────────────────
    print('=' * 70)
    print('FANS — 2D Rayleigh-Benard AFC (Markmann 2025)')
    print('=' * 70)
    env = RBControlEnv(env_cfg)

    # ── Multi-IC setup (optional) ──────────────────────────────────────
    ic_pool_seeds = fans_cfg.get('ic_pool_seeds', None)
    ic_rng = None
    if ic_pool_seeds is not None:
        print(f'[FANS] Preloading train IC pool: {ic_pool_seeds}')
        env.preload_ic_pool(ic_pool_seeds, verbose=True)
        ic_rng = np.random.default_rng(seed)
        # Start from a random pool entry
        state, obs = env.reset(ic_seed=int(ic_rng.choice(ic_pool_seeds)))
    else:
        state, obs = env.reset(key)
    nu_baseline = env.nu_baseline
    print(f'[FANS] Nu_baseline = {nu_baseline:.4f}'
          + (' (pool mean)' if ic_pool_seeds is not None else ''))
    print(f'[FANS] obs_size = {env.obs_size}, action_size = {env.action_size}')

    # Save initial state for periodic reset (legacy single-IC path only)
    state0_reset = jax.tree.map(lambda x: x.copy(), state)

    # Save baseline flow
    save_flow_snapshot(state, env.grid, label=-1, output_dir=output_dir)
    save_flow_image(state, env.grid, label=-1, output_dir=output_dir)

    # ── 2. FANS step function (wraps CFD + action→T_segments) ─────────
    fans_step_fn = make_rb_fans_step_fn(
        env.step_fn, env.T_hot, env.amplitude,
    )

    # ── 3. Pure observation & Nu functions (for JIT) ──────────────────
    get_obs_fn = make_rb_get_obs_fn(
        env.probe_ix_c, env.probe_iy_c,
        env.probe_ix_u, env.probe_iy_u,
        env.probe_ix_v, env.probe_iy_v,
    )
    compute_nu_fn = make_compute_nu_fn(env.grid, env.T_hot, env.T_cold,
                                       segment_map=env.segment_map)
    map_action_fn = make_map_action_fn(env.T_hot, env.amplitude)

    # ── 4. Initialize actor-critic + optimizers ───────────────────────
    key, subkey = jax.random.split(key)
    hidden_dims = fans_cfg['hidden_dims']
    action_dim = env.action_size
    params = init_actor_critic(subkey, env.obs_size, action_dim,
                               hidden_dims, hidden_dims)

    actor_opt = init_adam(params['actor'], lr=fans_cfg['actor_lr'])
    critic_opt = init_adam(params['critic'], lr=fans_cfg['critic_lr'])

    actor_size = sum(W.size + b.size for W, b in params['actor'])
    critic_size = sum(W.size + b.size for W, b in params['critic'])
    print(f'[FANS] Actor: {hidden_dims} ({actor_size:,} params)')
    print(f'[FANS] Critic: {hidden_dims} ({critic_size:,} params)')

    save_params(params, epoch=0, output_dir=output_dir, prefix='fans_init')

    # ── 5. JIT compile actor loss + grad ──────────────────────────────
    h = fans_cfg['horizon']
    action_interval = fans_cfg['action_interval']
    dt = env_cfg['dt']
    gamma = fans_cfg['gamma']
    segment_length = fans_cfg['segment_length']
    max_grad_norm = fans_cfg['max_grad_norm']

    def actor_loss_fn(actor_params, critic_params, state0):
        return fans_actor_loss(
            actor_params, critic_params, state0,
            fans_step_fn, get_obs_fn, compute_nu_fn,
            nu_baseline, h, action_interval, dt, gamma,
            segment_length=segment_length,
            action_dim=action_dim,
            action_penalty_coef=action_penalty_coef,
            map_action_fn=map_action_fn,
        )

    print(f'[FANS] JIT compiling actor loss + grad (this may take a few minutes)...')
    t0 = time.time()
    actor_val_grad_fn = jax.jit(jax.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True))

    # Warmup JIT
    (loss_test, aux_test), grad_test = actor_val_grad_fn(
        params['actor'], params['critic'], state,
    )
    jax.block_until_ready(loss_test)
    jit_time = time.time() - t0
    print(f'[FANS] JIT compiled in {jit_time:.1f}s')

    grad_norm = jnp.sqrt(
        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grad_test))
    )
    print(f'[FANS] Initial: loss={float(loss_test):.4f}, '
          f'grad_norm={float(grad_norm):.6f}')

    # JIT compile critic loss
    critic_val_grad_fn = jax.jit(jax.value_and_grad(
        fans_critic_loss_batched, argnums=0))

    # ── 6. Training loop ──────────────────────────────────────────────
    print(f'\n[FANS] Training: {fans_cfg["n_epochs"]} epochs, h={h}, '
          f'action_interval={action_interval}')
    print(f'[FANS] Grid: {env_cfg["Nx"]}x{env_cfg["Ny"]}, '
          f'segment_length={segment_length}')
    print(f'[FANS] Snapshots every {snapshot_every} epochs')
    if reset_every > 0:
        print(f'[FANS] State reset every {reset_every} epochs (anti-drift)')
    print('=' * 70)

    history = {
        'nu_baseline': float(nu_baseline),
        'env_config': {k: (float(v) if isinstance(v, (int, float)) else str(v))
                       for k, v in env_cfg.items()},
        'fans_config': {k: (float(v) if isinstance(v, (int, float)) else str(v))
                        for k, v in fans_cfg.items()},
        'jit_compile_time': jit_time,

        'epoch': [],
        'actor_loss': [],
        'critic_loss': [],
        'nu_mean': [],
        'nu_min': [],
        'nu_max': [],
        'action_mean': [],
        'action_std': [],
        'action_max': [],
        'grad_norm': [],
        'return': [],
        'V_h': [],
        'time': [],
        'cumulative_time': [],
        'total_cfd_steps': [],
        'reduction_pct': [],

        'rewards_per_step': [],
        'nus_per_step': [],
        'actions_per_step': [],
    }
    best_nu = float('inf')
    best_epoch = 0
    total_cfd_steps = 0

    t_start = time.time()

    for epoch in range(fans_cfg['n_epochs']):
        t_epoch = time.time()

        # ── Actor update (1st-order, through simulator) ──
        (actor_loss, aux), actor_grad = actor_val_grad_fn(
            params['actor'], params['critic'], state,
        )
        jax.block_until_ready(actor_loss)

        actor_grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree.leaves(actor_grad))
        )
        actor_opt, params['actor'] = adam_update(
            actor_opt, params['actor'], actor_grad,
            max_grad_norm=max_grad_norm,
        )

        # Extract info
        rewards = aux['rewards']
        nus = aux['nus']
        actions = aux['actions']
        obs_history = aux['obs_history']
        state_h = aux['state_h']

        nus_np = np.array(nus)
        nu_mean = float(np.mean(nus_np))
        nu_min = float(np.min(nus_np))
        nu_max = float(np.max(nus_np))
        reduction = (1.0 - nu_mean / nu_baseline) * 100

        actions_np = np.array(actions)  # (h, action_dim)
        action_mean = float(np.mean(np.abs(actions_np)))
        action_std = float(np.std(actions_np))
        action_max = float(np.max(np.abs(actions_np)))

        # ── Critic update (0th-order, TD targets) ──
        V_h = float(aux['V_h'])
        td_targets = compute_td_targets(
            jax.lax.stop_gradient(rewards), V_h, gamma, h,
        )

        critic_loss_val = 0.0
        for _ in range(fans_cfg['critic_update_epochs']):
            c_loss, c_grad = critic_val_grad_fn(
                params['critic'], obs_history, td_targets,
            )
            critic_opt, params['critic'] = adam_update(
                critic_opt, params['critic'], c_grad,
            )
            critic_loss_val = float(c_loss)

        # ── Advance state (rolling horizon with periodic reset) ──
        total_cfd_steps += h * action_interval
        if reset_every > 0 and (epoch + 1) % reset_every == 0:
            if ic_pool_seeds is not None:
                # Multi-IC: sample new IC from pool on each reset
                ic_seed = int(ic_rng.choice(ic_pool_seeds))
                state, _ = env.reset(ic_seed=ic_seed)
                print(f'  [State reset to IC seed={ic_seed} at epoch {epoch}]')
            else:
                state = jax.tree.map(lambda x: x.copy(), state0_reset)
                print(f'  [State reset to baseline at epoch {epoch}]')
        else:
            state = jax.lax.stop_gradient(state_h)

        # ── Logging ──
        dt_epoch = time.time() - t_epoch

        history['epoch'].append(epoch)
        history['actor_loss'].append(float(actor_loss))
        history['critic_loss'].append(critic_loss_val)
        history['nu_mean'].append(nu_mean)
        history['nu_min'].append(nu_min)
        history['nu_max'].append(nu_max)
        history['action_mean'].append(action_mean)
        history['action_std'].append(action_std)
        history['action_max'].append(action_max)
        history['grad_norm'].append(float(actor_grad_norm))
        history['return'].append(float(aux['return']))
        history['V_h'].append(V_h)
        history['time'].append(dt_epoch)
        history['cumulative_time'].append(time.time() - t_start)
        history['total_cfd_steps'].append(total_cfd_steps)
        history['reduction_pct'].append(reduction)
        history['rewards_per_step'].append([float(r) for r in rewards])
        history['nus_per_step'].append([float(n) for n in nus_np])
        history['actions_per_step'].append(actions_np.tolist())

        if epoch % fans_cfg['log_every'] == 0:
            print(f'Epoch {epoch:4d} | '
                  f'Nu={nu_mean:.3f} ({reduction:+.1f}%) | '
                  f'actor_L={float(actor_loss):+.4f} | '
                  f'critic_L={critic_loss_val:.4f} | '
                  f'|grad|={float(actor_grad_norm):.2e} | '
                  f'|a|={action_mean:.3f} (max={action_max:.3f}) | '
                  f'{dt_epoch:.1f}s')

        if nu_mean < best_nu:
            best_nu = nu_mean
            best_epoch = epoch
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            flat_actor = {f'actor_{i}_{k}': np.array(v)
                          for i, (W, b) in enumerate(params['actor'])
                          for k, v in [('W', W), ('b', b)]}
            flat_critic = {f'critic_{i}_{k}': np.array(v)
                           for i, (W, b) in enumerate(params['critic'])
                           for k, v in [('W', W), ('b', b)]}
            np.savez(os.path.join(ckpt_dir, 'fans_best.npz'),
                     epoch=epoch, nu=nu_mean, **flat_actor, **flat_critic)

        # ── Save flow snapshots + images ──
        if epoch % snapshot_every == 0 or epoch == fans_cfg['n_epochs'] - 1:
            save_flow_snapshot(state, env.grid, epoch, output_dir)
            save_flow_image(state, env.grid, epoch, output_dir)
            print(f'  [Saved flow snapshot + image: epoch {epoch}]')

        # ── Save checkpoint ──
        if (epoch + 1) % fans_cfg['save_every'] == 0:
            ckpt_path = save_params(params, epoch=epoch+1, output_dir=output_dir)
            print(f'  [Saved checkpoint: {ckpt_path}]')

        # ── Incremental history save (crash-safe) ──
        if (epoch + 1) % 10 == 0 or epoch == fans_cfg['n_epochs'] - 1:
            _save_history(history, output_dir)

    # ── 7. Summary ────────────────────────────────────────────────────
    best_reduction = (1.0 - best_nu / nu_baseline) * 100
    print('=' * 70)
    print(f'[FANS] Training complete.')
    print(f'  Nu baseline:     {nu_baseline:.4f}')
    print(f'  Best Nu:         {best_nu:.4f}  (epoch {best_epoch})')
    print(f'  Nu reduction:    {best_reduction:+.1f}%')
    print(f'  Total CFD steps: {total_cfd_steps:,}')
    print(f'  Total time:      {sum(history["time"]):.1f}s')

    _save_history(history, output_dir)
    save_params(params, epoch=fans_cfg['n_epochs'],
                output_dir=output_dir, prefix='fans_final')
    save_training_curves(history, output_dir)
    print(f'  Training curves: {output_dir}/training_curves.png')
    print(f'  History:         {output_dir}/history.json')

    return params, history


def main():
    parser = argparse.ArgumentParser(
        description='FANS training for RB convection AFC')
    parser.add_argument('--test', action='store_true',
                        help='Quick test with small grid')
    parser.add_argument('--v2', action='store_true',
                        help='Use V2 config (no reset, 200 epochs)')
    parser.add_argument('--ra', type=float, default=None,
                        help='Rayleigh number (1e4, 1e5, 1e6)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed (appends _s{seed} to output_dir)')
    args = parser.parse_args()

    if args.test:
        print('[FANS] Running in TEST mode (small grid, few epochs)')
        env_cfg, fans_cfg = TEST_ENV_CONFIG, TEST_FANS_CONFIG
    elif args.ra is not None and args.ra >= 9e5:
        env_cfg, fans_cfg = ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6
    elif args.ra is not None and args.ra >= 9e4:
        env_cfg, fans_cfg = ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5
    elif args.v2:
        env_cfg, fans_cfg = ENV_CONFIG, FANS_V2_CONFIG
    else:
        env_cfg, fans_cfg = ENV_CONFIG, FANS_CONFIG

    if args.seed is not None:
        fans_cfg = {**fans_cfg, 'seed': args.seed,
                    'output_dir': fans_cfg['output_dir'] + f'_s{args.seed}'}

    train(env_cfg, fans_cfg)


if __name__ == '__main__':
    main()
