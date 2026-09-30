"""Uncontrolled reference of the cylinder held-out evaluation (Table 2).

For each Re and held-out IC (100-104) the uncontrolled flow runs for one
evaluation episode after the warm-up, and the mean drag coefficient is recorded
over the first 2000 steps (bl2000), the last half of the episode (last_half, the
matched denominator of the drag reductions) and the whole episode (full).
Writes results/evaluation/cylinder_uncontrolled.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time, numpy as np
from diffrl_flow_control.configs import load_config
from diffrl_flow_control.cylinder.env import CylinderFlowEnv
from diffrl_flow_control.cylinder.heldout import build_perturbed_ic, EPISODE_ACTIONS, RE_GRID
out = {}
for Re in sorted(RE_GRID):
    cfg = load_config('cylinder', f'env_re{Re}'); cfg['soft_reset_prob']=0.0; cfg['obs_type']='velocity'
    env = CylinderFlowEnv(cfg); W=int(env.warmup_steps); ai=int(env.action_interval); t0=time.time()
    res = {}
    for ic in (100, 101, 102, 103, 104):
        st = build_perturbed_ic(env, ic)
        for _ in range(W): st = env._step_jit(st, env.dt, 0.0)
        cd = []
        for _ in range(EPISODE_ACTIONS * ai):
            st = env._step_jit(st, env.dt, 0.0); cd.append(float(st.ibm_forces[0]) / env.q_ref)
        cd = np.array(cd); h = EPISODE_ACTIONS // 2 * ai
        res[ic] = dict(bl2000=float(cd[:2000].mean()), last_half=float(cd[h:].mean()), full=float(cd.mean()))
        print(Re, ic, {k: round(v, 4) for k, v in res[ic].items()}, f'{time.time()-t0:.0f}s', flush=True)
    out[Re] = res
json.dump(out, open(ROOT + '/results/evaluation/cylinder_uncontrolled.json', 'w'))
print('done')
