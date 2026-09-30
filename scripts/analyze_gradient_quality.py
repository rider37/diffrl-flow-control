"""AD against finite-difference gradients (Appendix F, Fig. F.1).

Cosine similarity between the automatic-differentiation gradient and a central
finite-difference gradient on random policy states for each Ra, and its dependence
on the rollout horizon.
Usage:  python scripts/analyze_gradient_quality.py
Writes results/gradient_quality/cosine_similarity.json and horizon_sweep.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from jax_cfd.envs.rb_control_env import RBControlEnv

from experiments.rb_control.config import (
    ENV_CONFIG, FANS_CONFIG,
    ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5,
    ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6,
)
from experiments.rb_control.fans import (
    make_rb_fans_step_fn, make_rb_get_obs_fn, make_compute_nu_fn,
    make_map_action_fn,
    fans_actor_loss,
)
from experiments.rb_control.train_ppo import (
    init_actor_critic, critic_forward,
    init_adam, adam_update,
)


def setup_env_and_fns(env_cfg, fans_cfg):
    """Initialize environment and FANS functions."""
    key = jax.random.PRNGKey(fans_cfg['seed'])
    env = RBControlEnv(env_cfg)
    state, obs = env.reset(key)
    nu_baseline = env.nu_baseline

    fans_step_fn = make_rb_fans_step_fn(
        env.step_fn, env.T_hot, env.amplitude)
    get_obs_fn = make_rb_get_obs_fn(
        env.probe_ix_c, env.probe_iy_c,
        env.probe_ix_u, env.probe_iy_u,
        env.probe_ix_v, env.probe_iy_v,
    )
    compute_nu_fn = make_compute_nu_fn(
        env.grid, env.T_hot, env.T_cold,
        segment_map=env.segment_map)
    map_action_fn = make_map_action_fn(env.T_hot, env.amplitude)

    return env, state, nu_baseline, fans_step_fn, get_obs_fn, compute_nu_fn, map_action_fn


def compute_ad_gradient(actor_params, critic_params, state,
                        fans_step_fn, get_obs_fn, compute_nu_fn,
                        nu_baseline, h, action_interval, dt, gamma,
                        segment_length, action_dim, map_action_fn):
    """Compute AD gradient of actor loss."""
    def loss_fn(a_params):
        return fans_actor_loss(
            a_params, critic_params, state,
            fans_step_fn, get_obs_fn, compute_nu_fn,
            nu_baseline, h, action_interval, dt, gamma,
            segment_length=segment_length,
            action_dim=action_dim,
            map_action_fn=map_action_fn,
        )

    grad_fn = jax.jit(jax.grad(loss_fn, has_aux=True))
    grad, aux = grad_fn(actor_params)
    jax.block_until_ready(grad)
    return grad, aux


def compute_fd_gradient(actor_params, critic_params, state,
                        fans_step_fn, get_obs_fn, compute_nu_fn,
                        nu_baseline, h, action_interval, dt, gamma,
                        segment_length, action_dim, map_action_fn,
                        eps=1e-4):
    """Compute centered finite-difference gradient."""
    def loss_fn(a_params):
        l, _ = fans_actor_loss(
            a_params, critic_params, state,
            fans_step_fn, get_obs_fn, compute_nu_fn,
            nu_baseline, h, action_interval, dt, gamma,
            segment_length=segment_length,
            action_dim=action_dim,
            map_action_fn=map_action_fn,
        )
        return l

    loss_fn_jit = jax.jit(loss_fn)

    # Flatten params for perturbation
    flat_params, unflatten = ravel_pytree(actor_params)
    flat_params = np.array(flat_params)
    n_params = len(flat_params)

    # Sample subset of parameters for FD (full FD too expensive)
    n_sample = min(200, n_params)
    rng = np.random.RandomState(0)
    indices = rng.choice(n_params, n_sample, replace=False)

    fd_grad = np.zeros(n_params)
    for idx in indices:
        p_plus = flat_params.copy()
        p_plus[idx] += eps
        p_minus = flat_params.copy()
        p_minus[idx] -= eps

        l_plus = float(loss_fn_jit(unflatten(jnp.array(p_plus))))
        l_minus = float(loss_fn_jit(unflatten(jnp.array(p_minus))))
        fd_grad[idx] = (l_plus - l_minus) / (2 * eps)

    return fd_grad, indices


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def analyze_cosine_similarity(env_cfg, fans_cfg, n_states=10):
    """Compare AD vs FD gradient across random states."""
    print(f'\n[Cosine Sim] Ra={env_cfg["Ra"]:.0e}, n_states={n_states}')
    env, state0, nu_baseline, fans_step_fn, get_obs_fn, compute_nu_fn, map_action_fn = \
        setup_env_and_fns(env_cfg, fans_cfg)

    h = fans_cfg['horizon']
    action_interval = fans_cfg['action_interval']
    dt = env_cfg['dt']
    gamma = fans_cfg['gamma']
    segment_length = fans_cfg['segment_length']
    action_dim = env_cfg['n_segments']
    hidden_dims = fans_cfg['hidden_dims']

    results = []
    state = state0

    for i in range(n_states):
        print(f'  State {i+1}/{n_states}...')

        # Fresh random policy for each state
        key = jax.random.PRNGKey(i * 100)
        params = init_actor_critic(key, env.obs_size, action_dim,
                                   hidden_dims, hidden_dims)

        # AD gradient
        t0 = time.time()
        ad_grad, aux = compute_ad_gradient(
            params['actor'], params['critic'], state,
            fans_step_fn, get_obs_fn, compute_nu_fn,
            nu_baseline, h, action_interval, dt, gamma,
            segment_length, action_dim, map_action_fn)
        ad_time = time.time() - t0

        ad_flat, _ = ravel_pytree(ad_grad)
        ad_flat = np.array(ad_flat)

        # FD gradient (sampled)
        t0 = time.time()
        fd_flat, fd_indices = compute_fd_gradient(
            params['actor'], params['critic'], state,
            fans_step_fn, get_obs_fn, compute_nu_fn,
            nu_baseline, h, action_interval, dt, gamma,
            segment_length, action_dim, map_action_fn)
        fd_time = time.time() - t0

        # Cosine similarity on sampled dimensions only
        ad_sampled = ad_flat[fd_indices]
        fd_sampled = fd_flat[fd_indices]
        cos_sim = cosine_sim(ad_sampled, fd_sampled)

        # Relative error (on sampled dims)
        nonzero = np.abs(fd_sampled) > 1e-10
        if np.any(nonzero):
            rel_err = float(np.median(
                np.abs(ad_sampled[nonzero] - fd_sampled[nonzero])
                / np.abs(fd_sampled[nonzero])
            ))
        else:
            rel_err = float('nan')

        results.append({
            'state_idx': i,
            'cosine_similarity': cos_sim,
            'relative_error_median': rel_err,
            'ad_grad_norm': float(np.linalg.norm(ad_flat)),
            'fd_grad_norm': float(np.linalg.norm(fd_sampled)),
            'ad_time': ad_time,
            'fd_time': fd_time,
        })
        print(f'    cos_sim={cos_sim:.4f}, rel_err={rel_err:.4f}, '
              f'AD={ad_time:.1f}s, FD={fd_time:.1f}s')

        # Advance state for diversity
        action = np.zeros(action_dim)
        state, _, _, _ = env.step_no_scan(state, action)

    cos_sims = [r['cosine_similarity'] for r in results]
    print(f'  Mean cosine similarity: {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}')

    return {
        'Ra': float(env_cfg['Ra']),
        'horizon': h,
        'n_states': n_states,
        'results': results,
        'mean_cosine_sim': float(np.mean(cos_sims)),
        'std_cosine_sim': float(np.std(cos_sims)),
    }


def analyze_horizon_sweep(env_cfg, fans_cfg, horizons=[5, 10, 20, 50]):
    """Measure cosine similarity vs rollout horizon."""
    print(f'\n[Horizon Sweep] Ra={env_cfg["Ra"]:.0e}, horizons={horizons}')
    env, state, nu_baseline, fans_step_fn, get_obs_fn, compute_nu_fn, map_action_fn = \
        setup_env_and_fns(env_cfg, fans_cfg)

    action_interval = fans_cfg['action_interval']
    dt = env_cfg['dt']
    gamma = fans_cfg['gamma']
    action_dim = env_cfg['n_segments']
    hidden_dims = fans_cfg['hidden_dims']

    key = jax.random.PRNGKey(42)
    params = init_actor_critic(key, env.obs_size, action_dim,
                               hidden_dims, hidden_dims)

    results = []
    for h in horizons:
        seg_len = min(fans_cfg['segment_length'], h)
        print(f'  h={h}, segment_length={seg_len}...')

        # AD gradient
        ad_grad, _ = compute_ad_gradient(
            params['actor'], params['critic'], state,
            fans_step_fn, get_obs_fn, compute_nu_fn,
            nu_baseline, h, action_interval, dt, gamma,
            seg_len, action_dim, map_action_fn)
        ad_flat, _ = ravel_pytree(ad_grad)
        ad_flat = np.array(ad_flat)

        # FD gradient
        fd_flat, fd_indices = compute_fd_gradient(
            params['actor'], params['critic'], state,
            fans_step_fn, get_obs_fn, compute_nu_fn,
            nu_baseline, h, action_interval, dt, gamma,
            seg_len, action_dim, map_action_fn)

        ad_sampled = ad_flat[fd_indices]
        fd_sampled = fd_flat[fd_indices]
        cos_sim = cosine_sim(ad_sampled, fd_sampled)
        ad_norm = float(np.linalg.norm(ad_flat))

        results.append({
            'horizon': h,
            'cosine_similarity': cos_sim,
            'ad_grad_norm': ad_norm,
        })
        print(f'    cos_sim={cos_sim:.4f}, |grad|={ad_norm:.4e}')

    return {
        'Ra': float(env_cfg['Ra']),
        'results': results,
    }


def make_plots(cosine_results, horizon_results, output_dir):
    """Generate Fig 7: gradient quality plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARN] matplotlib not available, skipping plots')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel (a): Cosine similarity vs Ra ──
    ax = axes[0]
    ra_vals = []
    cos_means = []
    cos_stds = []
    for r in cosine_results:
        ra_vals.append(r['Ra'])
        cos_means.append(r['mean_cosine_sim'])
        cos_stds.append(r['std_cosine_sim'])

    ax.errorbar(range(len(ra_vals)), cos_means, yerr=cos_stds,
                fmt='o-', capsize=5, markersize=8, lw=2, color='#2196F3')
    ax.set_xticks(range(len(ra_vals)))
    ax.set_xticklabels([f'$10^{{{int(np.log10(ra))}}}$' for ra in ra_vals])
    ax.set_xlabel('Ra', fontsize=12)
    ax.set_ylabel('Cosine Similarity (AD vs FD)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('(a) Gradient Accuracy vs Turbulence', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.9, color='gray', ls='--', alpha=0.5, label='0.9 threshold')
    ax.legend()

    # ── Panel (b): Cosine similarity vs horizon ──
    ax = axes[1]
    colors = ['#4CAF50', '#FF9800', '#F44336']
    for i, hr in enumerate(horizon_results):
        ra = hr['Ra']
        hs = [r['horizon'] for r in hr['results']]
        cs = [r['cosine_similarity'] for r in hr['results']]
        label = f'Ra=$10^{{{int(np.log10(ra))}}}$'
        ax.plot(hs, cs, 'o-', color=colors[i % len(colors)],
                label=label, markersize=8, lw=2)

    ax.set_xlabel('Rollout Horizon h', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('(b) Effective Horizon', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_quality.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved: {output_dir}/gradient_quality.png')


def main():
    parser = argparse.ArgumentParser(
        description='Gradient quality analysis (FANS AD vs FD)')
    parser.add_argument('--n-states', type=int, default=10,
                        help='Number of random states for cosine sim')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (3 states, fewer horizons)')
    args = parser.parse_args()

    output_dir = os.path.join(_root, 'results', 'gradient_quality')
    os.makedirs(output_dir, exist_ok=True)

    n_states = 3 if args.quick else args.n_states
    horizons = [5, 10, 20] if args.quick else [5, 10, 20, 50]

    configs = [
        (ENV_CONFIG, FANS_CONFIG, 'Ra=1e4'),
        (ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5, 'Ra=1e5'),
        (ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6, 'Ra=1e6'),
    ]

    # ── 1. Cosine similarity across Ra ──
    print('=' * 70)
    print('ANALYSIS 1: Cosine Similarity (AD vs FD) across Ra')
    print('=' * 70)
    cosine_results = []
    for env_cfg, fans_cfg, label in configs:
        result = analyze_cosine_similarity(env_cfg, fans_cfg, n_states=n_states)
        cosine_results.append(result)

    with open(os.path.join(output_dir, 'cosine_similarity.json'), 'w') as f:
        json.dump(cosine_results, f, indent=2)

    # ── 2. Horizon sweep ──
    print('\n' + '=' * 70)
    print('ANALYSIS 2: Effective Horizon Sweep')
    print('=' * 70)
    horizon_results = []
    for env_cfg, fans_cfg, label in configs:
        result = analyze_horizon_sweep(env_cfg, fans_cfg, horizons=horizons)
        horizon_results.append(result)

    with open(os.path.join(output_dir, 'horizon_sweep.json'), 'w') as f:
        json.dump(horizon_results, f, indent=2)

    # ── 3. Generate Fig 7 ──
    make_plots(cosine_results, horizon_results, output_dir)

    # ── Summary ──
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    for r in cosine_results:
        print(f'  Ra={r["Ra"]:.0e}: cosine_sim = '
              f'{r["mean_cosine_sim"]:.4f} ± {r["std_cosine_sim"]:.4f}')
    print(f'\nResults saved to: {output_dir}/')


if __name__ == '__main__':
    main()
