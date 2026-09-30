"""AD-FD cosine similarity against the finite-difference step (Appendix F).

The simulator runs in FP32, whose optimal central-difference step is about
eps_machine^(1/3) ~ 5e-3.  The AD gradient is computed once per state and compared
with FD gradients at several steps, to check that the AD-FD agreement does not
depend on the FD step.  Reuses the AD/FD routines of analyze_gradient_quality.py.
Writes results/gradient_quality/eps_sweep.json.
"""
import json, os, sys
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))
sys.path.insert(0, os.path.join(_root, 'scripts'))

import jax
from jax.flatten_util import ravel_pytree
from experiments.rb_control.config import (
    ENV_CONFIG, FANS_CONFIG, ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5,
    ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6,
)
from experiments.rb_control.train_ppo import init_actor_critic
from analyze_gradient_quality import (
    setup_env_and_fns, compute_ad_gradient, compute_fd_gradient, cosine_sim,
)

EPS = [1e-2, 3e-3, 1e-3, 1e-4, 1e-5]      # spans FP32-optimal (~5e-3) to too-small
N_STATES = 5
CONFIGS = [(ENV_CONFIG, FANS_CONFIG, '1e4'),
           (ENV_CONFIG_RA1E5, FANS_CONFIG_RA1E5, '1e5'),
           (ENV_CONFIG_RA1E6, FANS_CONFIG_RA1E6, '1e6')]
OUT = os.path.join(_root, 'results', 'gradient_quality', 'eps_sweep.json')

out = []
for env_cfg, fans_cfg, ra in CONFIGS:
    env, state, nu_baseline, fans_step_fn, get_obs_fn, compute_nu_fn, map_action_fn = \
        setup_env_and_fns(env_cfg, fans_cfg)
    h = fans_cfg['horizon']; ai = fans_cfg['action_interval']; dt = env_cfg['dt']
    gamma = fans_cfg['gamma']; seg = fans_cfg['segment_length']
    adim = env_cfg['n_segments']; hid = fans_cfg['hidden_dims']
    cos = {e: [] for e in EPS}
    for i in range(N_STATES):
        key = jax.random.PRNGKey(i * 100)
        params = init_actor_critic(key, env.obs_size, adim, hid, hid)
        ad_grad, _ = compute_ad_gradient(
            params['actor'], params['critic'], state, fans_step_fn, get_obs_fn,
            compute_nu_fn, nu_baseline, h, ai, dt, gamma, seg, adim, map_action_fn)
        ad_flat, _ = ravel_pytree(ad_grad); ad_flat = np.array(ad_flat)
        for e in EPS:
            fd_flat, idx = compute_fd_gradient(
                params['actor'], params['critic'], state, fans_step_fn, get_obs_fn,
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
print("saved eps_sweep.json", flush=True)
