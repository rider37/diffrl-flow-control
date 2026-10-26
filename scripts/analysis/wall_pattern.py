"""Wall-temperature patterns of the trained RB policies (Fig. 9).

Mean wall temperature per heater segment of the final SHAC policy (training seed 42)
at each Ra, and the spatial power spectrum of the time-averaged wall pattern.
Usage:  python scripts/analysis/wall_pattern.py
Writes results/diagnostics/wall_pattern.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

import json

import numpy as np


import jax
import jax.numpy as jnp

from diffrl_flow_control.rayleigh_benard.env import RBControlEnv

from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
SHAC_RA1E4 = load_config('rayleigh_benard', 'shac_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
SHAC_RA1E5 = load_config('rayleigh_benard', 'shac_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
SHAC_RA1E6 = load_config('rayleigh_benard', 'shac_ra1e6')
from diffrl_flow_control.rayleigh_benard.shac import make_map_action_fn
from diffrl_flow_control.rayleigh_benard.ppo import (
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


def analyze_rb_policy(env_cfg, shac_cfg, checkpoint_dir, label,
                      n_rollout_steps=100):
    """Analyze learned RB control policy."""
    print(f'\n[Policy Analysis] {label}')

    # Load best actor
    best_path = os.path.join(checkpoint_dir, 'final.npz')   # final trained policy (paper protocol)
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

    best_path = os.path.join(checkpoint_dir, 'final.npz')   # final trained policy (paper protocol)
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


def main():
    output_dir = os.path.join(ROOT, 'results', 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)

    # ── RB analysis ──
    rb_configs = [
        (ENV_RA1E4, SHAC_RA1E4,
         'runs/rayleigh_benard/shac_ra1e4_s42/checkpoints',
         'Ra=1e4'),
        (ENV_RA1E5, SHAC_RA1E5,
         'runs/rayleigh_benard/shac_ra1e5_s42/checkpoints',
         'Ra=1e5'),
        (ENV_RA1E6, SHAC_RA1E6,
         'runs/rayleigh_benard/shac_ra1e6_s42/checkpoints',
         'Ra=1e6'),
    ]

    rb_results = []
    for env_cfg, shac_cfg, ckpt_dir, label in rb_configs:
        ckpt_path = os.path.join(ROOT, ckpt_dir)
        result = analyze_rb_policy(env_cfg, shac_cfg, ckpt_path, label)
        rb_results.append(result)

    # ── Cylinder analysis ──
    cyl_configs = []   # cylinder panels are not used by the RB policy figure

    cyl_results = []
    for ckpt_dir, re_val, label in cyl_configs:
        ckpt_path = os.path.join(ROOT, ckpt_dir)
        result = analyze_cylinder_policy(ckpt_path, re_val, label)
        cyl_results.append(result)

    # ── Save results ──
    summary = {
        'rb_results': [r for r in rb_results if r is not None],
        'cylinder_results': [r for r in cyl_results if r is not None],
    }
    with open(os.path.join(output_dir, 'wall_pattern.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults saved to: {output_dir}/')


if __name__ == '__main__':
    main()
