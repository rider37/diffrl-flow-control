import os, sys, json, time, numpy as np
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, PKG+'/scripts'); sys.path.insert(0, PKG); sys.path.insert(0, PKG+'/src'); os.chdir(PKG+'/scripts')
from cases.cylinder_afc import make_env_config
from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv
from cyl_eval_common import build_perturbed_ic, EPISODE_ACTIONS, RE_GRID
out = {}
for Re in sorted(RE_GRID):
    cfg = make_env_config(Re=Re, cells=RE_GRID[Re], ibm_eps_factor=0.75); cfg['jet_method']='surface_bc'; cfg['soft_reset_prob']=0.0; cfg['obs_type']='velocity'
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
json.dump(out, open(PKG + '/results/cyl_heldout_matched_baselines.json', 'w'))
print('done')
