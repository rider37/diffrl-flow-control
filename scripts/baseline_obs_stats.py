"""Baseline (uncontrolled) observation statistics for RB at each Ra.

Measures the per-channel (T, u, v) distribution of the probe observations
that the PPO/SHAC policy actually sees, on the uncontrolled flow, to
document that observations are already non-dimensional and O(1) (Appendix
support for the obs-normalization discussion).  Pure measurement: no
training, no gradients.

Output: results/baseline_obs_stats.json  (+ printed table)
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.rb_control.config import (
    ENV_CONFIG, ENV_CONFIG_RA1E5, ENV_CONFIG_RA1E6,
)
from jax_cfd.envs.rb_control_env import RBControlEnv
sys.path.insert(0, os.path.dirname(__file__))
from rb_eval_common import build_ic

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
    for cfg, label in [(ENV_CONFIG, 'Ra1e4'),
                       (ENV_CONFIG_RA1E5, 'Ra1e5'),
                       (ENV_CONFIG_RA1E6, 'Ra1e6')]:
        out[label] = run_ra({**cfg}, label)
    os.makedirs('results', exist_ok=True)
    path = 'results/baseline_obs_stats.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == '__main__':
    main()
