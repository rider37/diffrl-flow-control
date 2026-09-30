"""Uncontrolled reference of the Rayleigh-Benard held-out evaluation (Table 4).

For each held-out IC: the uncontrolled Nu over (a) the first baseline_steps after
the warm-up and (b) the last half of an uncontrolled run of episode length from the
same equilibrated state, sampled at the action ends like the controlled run; (b) is
the matched denominator of the reductions.
Writes results/evaluation/rayleigh_benard_uncontrolled.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time, numpy as np
from diffrl_flow_control.rayleigh_benard.heldout import build_ic
from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
out = {}
for ra, cfg in (('1e4', ENV_RA1E4), ('1e5', ENV_RA1E5), ('1e6', ENV_RA1E6)):
    env = RBControlEnv({**cfg}); W, B, E, AI = cfg['warmup_steps'], cfg['baseline_steps'], cfg['episode_actions'], cfg['action_interval']
    res = []
    for ic in (100, 101, 102, 103, 104):
        t0 = time.time(); st = build_ic(env, ic)
        for _ in range(W): st = env._step_jit(st, env.dt, None)
        per_step = []; per_act = []
        for a in range(E):
            for k in range(AI):
                st = env._step_jit(st, env.dt, None)
                if a * AI + k < B: per_step.append(float(env._compute_nu_jit(st)))
            per_act.append(float(env._compute_nu_jit(st)))
        per_act = np.array(per_act)
        r = dict(ic=ic, bl_window=float(np.mean(per_step)), last_half=float(per_act[E // 2:].mean()),
                 full=float(per_act.mean()), per_act=per_act.tolist())
        res.append(r); print(ra, ic, round(r['bl_window'], 4), round(r['last_half'], 4), f'{time.time()-t0:.0f}s', flush=True)
    out[ra] = res
json.dump(out, open('results/evaluation/rayleigh_benard_uncontrolled.json', 'w'))
print('done')
