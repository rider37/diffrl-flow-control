"""AD-FD cosine similarity against the finite-difference step (Appendix F).

The simulator runs in FP32, whose optimal central-difference step is about
eps_machine^(1/3) ~ 5e-3.  The AD gradient is computed once per state and compared
with FD gradients at several steps, to check that the AD-FD agreement does not
depend on the FD step.  Reuses the AD/FD routines of ad_fd_gradients.py.
Writes results/diagnostics/ad_fd_step.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json
import numpy as np


import jax
from jax.flatten_util import ravel_pytree
from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
SHAC_RA1E4 = load_config('rayleigh_benard', 'shac_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
SHAC_RA1E5 = load_config('rayleigh_benard', 'shac_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
SHAC_RA1E6 = load_config('rayleigh_benard', 'shac_ra1e6')
# The AD-FD check uses randomly initialised policies of widths 64, 96 and 128 at
# Ra = 1e4, 1e5, 1e6 (seed 42), as in the article; horizon, action interval and
# segment length are those of the SHAC runs.
SHAC_RA1E4 = dict(SHAC_RA1E4, hidden_dims=[64, 64], seed=42)
SHAC_RA1E5 = dict(SHAC_RA1E5, hidden_dims=[96, 96], seed=42)
SHAC_RA1E6 = dict(SHAC_RA1E6, hidden_dims=[128, 128], seed=42)
from diffrl_flow_control.rayleigh_benard.ppo import init_actor_critic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_fd_gradients import (
    setup_env_and_fns, compute_ad_gradient, compute_fd_gradient, cosine_sim,
)

EPS = [1e-2, 3e-3, 1e-3, 1e-4, 1e-5]      # spans FP32-optimal (~5e-3) to too-small
N_STATES = 5
CONFIGS = [(ENV_RA1E4, SHAC_RA1E4, '1e4'),
           (ENV_RA1E5, SHAC_RA1E5, '1e5'),
           (ENV_RA1E6, SHAC_RA1E6, '1e6')]
OUT = os.path.join(ROOT, 'results', 'diagnostics', 'ad_fd_step.json')

out = []
for env_cfg, shac_cfg, ra in CONFIGS:
    env, state, nu_baseline, shac_step_fn, get_obs_fn, compute_nu_fn, map_action_fn = \
        setup_env_and_fns(env_cfg, shac_cfg)
    h = shac_cfg['horizon']; ai = shac_cfg['action_interval']; dt = env_cfg['dt']
    gamma = shac_cfg['gamma']; seg = shac_cfg['segment_length']
    adim = env_cfg['n_segments']; hid = shac_cfg['hidden_dims']
    cos = {e: [] for e in EPS}
    for i in range(N_STATES):
        key = jax.random.PRNGKey(i * 100)
        params = init_actor_critic(key, env.obs_size, adim, hid, hid)
        ad_grad, _ = compute_ad_gradient(
            params['actor'], params['critic'], state, shac_step_fn, get_obs_fn,
            compute_nu_fn, nu_baseline, h, ai, dt, gamma, seg, adim, map_action_fn)
        ad_flat, _ = ravel_pytree(ad_grad); ad_flat = np.array(ad_flat)
        for e in EPS:
            fd_flat, idx = compute_fd_gradient(
                params['actor'], params['critic'], state, shac_step_fn, get_obs_fn,
                compute_nu_fn, nu_baseline, h, ai, dt, gamma, seg, adim,
                map_action_fn, eps=e)
            cos[e].append(cosine_sim(ad_flat[idx], fd_flat[idx]))
        print(f"  Ra={ra} state {i+1}/{N_STATES}: "
              + " ".join(f"{e:g}={cos[e][-1]:.3f}" for e in EPS), flush=True)
        state, _, _, _ = env.step_no_scan(state, np.zeros(adim))
    row = {'Ra': ra, 'horizon': int(h),
           'eps': {f'{e:g}': {'mean': float(np.mean(cos[e])),
                              'median': float(np.median(cos[e])),
                              'min': float(np.min(cos[e]))} for e in EPS}}
    out.append(row)
    print(f"Ra={ra}: " + "  ".join(f"eps={e:g}:{np.mean(cos[e]):.3f}" for e in EPS),
          flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=2)
print("saved ad_fd_step.json", flush=True)
