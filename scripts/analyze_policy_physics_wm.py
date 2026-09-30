"""Wall-temperature patterns of the trained RB policies (Fig. 7).

Mean wall temperature per heater segment of the final SHAC policy (training seed 42)
at each Ra, and the spatial power spectrum of the time-averaged wall pattern.
Usage:  python scripts/analyze_policy_physics_wm.py
Writes results/policy_physics_wm/policy_summary.json (input of figures/replot_fig6_policy.py).
"""

import json
import os
import sys

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src')); sys.path.insert(0, os.path.join(_root, 'scripts'))

import jax
import jax.numpy as jnp

from jax_cfd.envs.rb_control_env import RBControlEnv

from experiments.rb_control.config import (
    ENV_CONFIG, FANS_CONFIG,
    ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5,
    ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6,
)
from experiments.rb_control.fans import make_map_action_fn
from experiments.rb_control.train_ppo import (
    mlp_forward, init_mlp,
)


def load_actor_params(checkpoint_path):
    """Load actor parameters from .npz checkpoint."""
    data = np.load(checkpoint_path)
    # Reconstruct actor params: list of (W, b) tuples
    params = []
    i = 0
    while f'actor_{i}_W' in data:
        W = jnp.array(data[f'actor_{i}_W'])
        b = jnp.array(data[f'actor_{i}_b'])
        params.append((W, b))
        i += 1
    return params


def analyze_rb_policy(env_cfg, fans_cfg, checkpoint_dir, label,
                      n_rollout_steps=100):
    """Analyze learned RB control policy."""
    print(f'\n[Policy Analysis] {label}')

    # Load best actor
    best_path = os.path.join(checkpoint_dir, 'fans_final_epoch_1000.npz')   # final trained policy (paper protocol)
    if not os.path.exists(best_path):
        print(f'  [SKIP] {best_path} not found')
        return None

    actor_params = load_actor_params(best_path)
    map_action_fn = make_map_action_fn(
        env_cfg.get('T_hot', 1.0),
        env_cfg['amplitude'])

    # Setup env
    key = jax.random.PRNGKey(42)
    env = RBControlEnv(env_cfg)
    state, obs = env.reset(key)
    action_dim = env_cfg['n_segments']

    # Collect actions over rollout
    actions_all = []
    T_walls_all = []
    nus = []

    for step in range(n_rollout_steps):
        obs_arr = np.array(env._get_obs_jit(state))
        out = mlp_forward(actor_params, jnp.array(obs_arr))
        action = np.array(jnp.tanh(out[:action_dim]))
        T_wall = np.array(map_action_fn(jnp.array(action)))

        actions_all.append(action)
        T_walls_all.append(T_wall)

        state, _, _, info = env.step_no_scan(state, action)
        nus.append(float(info['Nu']))

    actions_arr = np.array(actions_all)  # (n_steps, n_segments)
    T_walls_arr = np.array(T_walls_all)

    # Mean action pattern
    mean_action = np.mean(actions_arr, axis=0)
    std_action = np.std(actions_arr, axis=0)
    mean_T_wall = np.mean(T_walls_arr, axis=0)

    # Spatial FFT of mean T_wall pattern
    T_anomaly = mean_T_wall - np.mean(mean_T_wall)
    fft_coeff = np.fft.rfft(T_anomaly)
    fft_power = np.abs(fft_coeff) ** 2
    dominant_k = int(np.argmax(fft_power[1:])) + 1  # skip DC

    # Temporal FFT of actions (how often policy changes)
    action_temporal_fft = np.fft.rfft(actions_arr[:, 0])
    temporal_power = np.abs(action_temporal_fft) ** 2

    result = {
        'label': label,
        'Ra': float(env_cfg['Ra']),
        'mean_action': mean_action.tolist(),
        'std_action': std_action.tolist(),
        'mean_T_wall': mean_T_wall.tolist(),
        'dominant_wavenumber': dominant_k,
        'fft_power': fft_power.tolist(),
        'mean_nu': float(np.mean(nus[-50:])),
        'nu_baseline': float(env.nu_baseline),
        'mean_action_magnitude': float(np.mean(np.abs(actions_arr))),
    }

    print(f'  Nu baseline: {env.nu_baseline:.4f}')
    print(f'  Mean Nu (last 50): {result["mean_nu"]:.4f}')
    print(f'  Mean |action|: {result["mean_action_magnitude"]:.3f}')
    print(f'  Dominant wavenumber k={dominant_k}')
    print(f'  Mean T_wall: {mean_T_wall}')

    return result


def analyze_cylinder_policy(checkpoint_dir, re_val, label):
    """Analyze learned cylinder jet policy."""
    print(f'\n[Cylinder Policy] {label}')

    best_path = os.path.join(checkpoint_dir, 'fans_final_epoch_1000.npz')   # final trained policy (paper protocol)
    if not os.path.exists(best_path):
        print(f'  [SKIP] {best_path} not found')
        return None

    # Load history for action patterns
    hist_path = os.path.join(os.path.dirname(checkpoint_dir), 'history.json')
    if not os.path.exists(hist_path):
        hist_path = os.path.join(
            os.path.dirname(checkpoint_dir), '..', 'history.json')

    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

        # Get action patterns from last 50 epochs
        actions = history.get('actions_per_step', [])
        if actions:
            last_actions = np.array(actions[-50:])
            mean_action = np.mean(np.abs(last_actions), axis=0)
            result = {
                'label': label,
                'Re': re_val,
                'mean_action_magnitude': float(np.mean(np.abs(last_actions))),
                'mean_action_pattern': mean_action.tolist() if mean_action.ndim == 1 else float(np.mean(mean_action)),
            }
            print(f'  Mean |action|: {result["mean_action_magnitude"]:.3f}')
            return result

    print(f'  [SKIP] No history found')
    return None


def make_plots(rb_results, cyl_results, output_dir):
    """Generate Fig 8: policy interpretation plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARN] matplotlib not available')
        return

    rb_results = [r for r in rb_results if r is not None]
    n_rb = len(rb_results)

    fig, axes = plt.subplots(2, max(n_rb, 1), figsize=(5 * max(n_rb, 1), 8))
    if n_rb == 1:
        axes = axes[:, np.newaxis]

    # ── Row 1: Mean wall temperature patterns ──
    for i, r in enumerate(rb_results):
        ax = axes[0, i]
        n_seg = len(r['mean_T_wall'])
        x = np.arange(n_seg)
        ax.bar(x, r['mean_T_wall'], color='#FF5722', alpha=0.7,
               edgecolor='black', lw=0.5)
        ax.axhline(1.0, color='gray', ls='--', lw=1, label='$T_{hot}$')
        ax.set_xlabel('Segment', fontsize=11)
        ax.set_ylabel('Mean $T_{wall}$', fontsize=11)
        ra_exp = int(np.log10(r['Ra']))
        ax.set_title(f'Ra=$10^{ra_exp}$, Nu={r["mean_nu"]:.2f} '
                     f'(-{(1 - r["mean_nu"]/r["nu_baseline"])*100:.1f}%)',
                     fontsize=11)
        ax.set_ylim(0, 2.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Row 2: FFT power spectrum ──
    for i, r in enumerate(rb_results):
        ax = axes[1, i]
        power = np.array(r['fft_power'])
        k = np.arange(len(power))
        ax.bar(k[1:], power[1:], color='#2196F3', alpha=0.7,
               edgecolor='black', lw=0.5)
        ax.set_xlabel('Wavenumber k', fontsize=11)
        ax.set_ylabel('FFT Power', fontsize=11)
        ax.set_title(f'Dominant k={r["dominant_wavenumber"]}', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Learned Policy: Wall Temperature Patterns & Spatial FFT',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_physics.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved: {output_dir}/policy_physics.png')


def main():
    output_dir = os.path.join(_root, 'results', 'policy_physics_wm')
    os.makedirs(output_dir, exist_ok=True)

    # ── RB analysis ──
    rb_configs = [
        (ENV_CONFIG, FANS_CONFIG,
         'experiments/rb_control/output_phase3_shac_ra1e4_s42/checkpoints',
         'Ra=1e4'),
        (ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5,
         'experiments/rb_control/output_wm_shac_ra1e5_s42/checkpoints',
         'Ra=1e5'),
        (ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6,
         'experiments/rb_control/output_wm_shac_ra1e6_s42/checkpoints',
         'Ra=1e6'),
    ]

    rb_results = []
    for env_cfg, fans_cfg, ckpt_dir, label in rb_configs:
        ckpt_path = os.path.join(_root, ckpt_dir)
        result = analyze_rb_policy(env_cfg, fans_cfg, ckpt_path, label)
        rb_results.append(result)

    # ── Cylinder analysis ──
    cyl_configs = []   # cylinder panels are not used by the RB policy figure

    cyl_results = []
    for ckpt_dir, re_val, label in cyl_configs:
        ckpt_path = os.path.join(_root, ckpt_dir)
        result = analyze_cylinder_policy(ckpt_path, re_val, label)
        cyl_results.append(result)

    # ── Save results ──
    summary = {
        'rb_results': [r for r in rb_results if r is not None],
        'cylinder_results': [r for r in cyl_results if r is not None],
    }
    with open(os.path.join(output_dir, 'policy_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Plots ──
    make_plots(rb_results, cyl_results, output_dir)

    print(f'\nResults saved to: {output_dir}/')


if __name__ == '__main__':
    main()
