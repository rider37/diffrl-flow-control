"""Scales of the Rayleigh-Benard observations on the uncontrolled flow (Table A.2).

Measures the per-channel (T, u, v) distribution of the probe observations seen
by the SHAC and PPO policies at each Ra, showing that they are non-dimensional
and of order one (Section 5.4, observation normalisation).
Writes results/statistics/observation_scales.json.

Output: results/statistics/observation_scales.json  (+ printed table)
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time
import numpy as np


from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic

IC_SEED = 100          # one held-out IC is enough for the distribution
N_SAMPLES = 100        # uncontrolled obs snapshots per Ra


def channel_stats(arr):
    """arr: (n_samples, n_probes) for one channel."""
    a = np.asarray(arr, np.float64).ravel()
    return dict(min=float(a.min()), max=float(a.max()),
                mean=float(a.mean()), std=float(a.std()),
                absmax=float(np.abs(a).max()))


def run_ra(cfg, label):
    env = RBControlEnv(cfg)
    npx, npy = cfg['n_probes_x'], cfg['n_probes_y']
    block = npx * npy
    state = build_ic(env, IC_SEED)
    warmup = cfg['warmup_steps']
    ai = cfg['action_interval']
    print(f"\n[{label}] Ra={cfg['Ra']:.0e} grid {cfg['Nx']}x{cfg['Ny']} "
          f"obs_dim={env.obs_size} (3x{block}); warmup {warmup}...", flush=True)
    t0 = time.time()
    for _ in range(warmup):
        state = env._step_jit(state, env.dt, None)
    obs_samples = []
    for s in range(N_SAMPLES):
        obs_samples.append(np.asarray(env._get_obs_jit(state)))
        for _ in range(ai):
            state = env._step_jit(state, env.dt, None)
    obs = np.stack(obs_samples)  # (N_SAMPLES, obs_dim)
    T_ch, u_ch, v_ch = obs[:, :block], obs[:, block:2*block], obs[:, 2*block:]
    res = {
        'Ra': float(cfg['Ra']), 'grid': [cfg['Nx'], cfg['Ny']],
        'obs_dim': int(env.obs_size), 'n_probes': int(block),
        'n_samples': N_SAMPLES,
        'T': channel_stats(T_ch), 'u': channel_stats(u_ch), 'v': channel_stats(v_ch),
        'all': channel_stats(obs),
    }
    print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)
    for ch in ('T', 'u', 'v'):
        c = res[ch]
        print(f"   {ch}: mean={c['mean']:+.3f} std={c['std']:.3f} "
              f"min={c['min']:+.3f} max={c['max']:+.3f} |max|={c['absmax']:.3f}")
    return res


def main():
    out = {}
    for cfg, label in [(ENV_RA1E4, 'Ra1e4'),
                       (ENV_RA1E5, 'Ra1e5'),
                       (ENV_RA1E6, 'Ra1e6')]:
        out[label] = run_ra({**cfg}, label)
    os.makedirs('results', exist_ok=True)
    path = 'results/statistics/observation_scales.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == '__main__':
    main()
