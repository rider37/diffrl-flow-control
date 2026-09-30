"""Pre-tanh saturation of the PPO policies at Ra=1e4 and 1e6 (Section 5.4).

For each final PPO checkpoint (five training seeds) the deterministic policy runs on a
held-out IC; at three snapshots the pre-tanh Gaussian mean mu_i of each of the 12 wall
segments is recorded.  Reports the median |mu_i| and the fraction with |mu_i| > 2,
the reference for the observation-normalisation check of
scripts/evaluate/rayleigh_benard_heldout_obsnorm.py.
Writes results/statistics/ppo_saturation.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json
import numpy as np
import jax, jax.numpy as jnp

from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic, load_params_flexible

SEEDS = [42, 123, 456, 517, 1223]
IC = 100
N_SNAP = 3


def ckpt(ra, s):
    return f'runs/rayleigh_benard/ppo_ra{ra}_s{s}/checkpoints/final.npz'


def measure(ra, cfg):
    env = RBControlEnv({**cfg})
    warmup, ep, ad = cfg['warmup_steps'], cfg['episode_actions'], env.action_size
    ai = cfg['action_interval']
    snap_at = [ep // 4, ep // 2, 3 * ep // 4][:N_SNAP]
    mus = []
    for s in SEEDS:
        params, _ = load_params_flexible(ckpt(ra, s))
        actor = params['actor']

        @jax.jit
        def mu_of(obs):
            return mlp_forward(actor, obs)[:ad]  # pre-tanh Gaussian mean

        state = build_ic(env, IC)
        for _ in range(warmup):
            state = env._step_jit(state, env.dt, None)
        for a in range(ep):
            obs = env._get_obs_jit(state)
            mu = np.asarray(mu_of(obs))
            if a in snap_at:
                mus.append(np.abs(mu))
            action = jnp.tanh(jnp.asarray(mu))
            T_segs = env.map_action(action)
            for _ in range(ai):
                state = env._step_jit(state, env.dt, T_segs)
    allmu = np.concatenate(mus)  # (n_seed*n_snap*12,)
    return dict(ra=ra, n=len(SEEDS), median=float(np.median(allmu)),
                frac_gt2=float(np.mean(allmu > 2.0)), n_vals=int(allmu.size))


out = {}
for ra, cfg in [('1e6', ENV_RA1E6), ('1e4', ENV_RA1E4)]:
    r = measure(ra, cfg)
    out[ra] = r
    print(f"Ra={ra}: median|mu|={r['median']:.2f}, frac>2={r['frac_gt2']*100:.0f}% "
          f"(5 seeds x {N_SNAP} snap x 12 seg = {r['n_vals']} values)", flush=True)
os.makedirs('results', exist_ok=True)
json.dump(out, open('results/statistics/ppo_saturation.json', 'w'), indent=2)
print("Saved results/statistics/ppo_saturation.json")
