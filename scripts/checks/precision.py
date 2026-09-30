"""Single- versus double-precision check of the held-out reduction (representative cases).

  python scripts/checks/precision.py 32
  python scripts/checks/precision.py 64
For cylinder Re=100 and RB Ra=1e4, the final seed-42 SHAC policy is evaluated on held-out IC 100
with the Table 2/4 protocol (last-half mean against the uncontrolled mean over the same window of an
uncontrolled run from the same equilibrated state) in the requested precision.
Writes results/checks/precision_fp<bits>.json
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json
bits = int(sys.argv[1])
import jax
if bits == 64:
    jax.config.update('jax_enable_x64', True)
import numpy as np, jax.numpy as jnp
out = dict(bits=bits)

# cylinder Re=100
from diffrl_flow_control.configs import load_config
from diffrl_flow_control.cylinder.env import CylinderFlowEnv
from diffrl_flow_control.cylinder.heldout import build_perturbed_ic, load_actor, EPISODE_ACTIONS
from diffrl_flow_control.cylinder.ppo import mlp_forward_relu
cfg = load_config('cylinder', f'env_re{100}')
cfg['soft_reset_prob'] = 0.0; cfg['obs_type'] = 'velocity'
env = CylinderFlowEnv(cfg); ai, ema, E = int(env.action_interval), float(env.ema_alpha), EPISODE_ACTIONS
actor = [tuple(jnp.asarray(w, dtype=jnp.float64 if bits == 64 else jnp.float32) for w in l)
         for l in load_actor(f'{ROOT}/runs/cylinder/shac_re100_s42/checkpoints/final.npz')]
act = jax.jit(lambda o: jnp.tanh(mlp_forward_relu(actor, o)[:1]))
st0 = build_perturbed_ic(env, 100)
for _ in range(int(env.warmup_steps)): st0 = env._step_jit(st0, env.dt, 0.0)
cd = {}
for mode in ('uncontrolled', 'shac'):
    st = st0; Q = 0.0; per = []
    for k in range(E):
        a = float(act(env._get_obs(st))[0]) if mode == 'shac' else 0.0; c = []
        for _ in range(ai):
            Q = Q + ema * (a - Q) if mode == 'shac' else 0.0
            st = env._step_jit(st, env.dt, Q); c.append(float(st.ibm_forces[0]) / env.q_ref)
        per.append(np.mean(c))
    cd[mode] = float(np.mean(per[E // 2:]))
out['cyl_re100'] = dict(cd_unctrl=cd['uncontrolled'], cd_ctrl=cd['shac'],
                        reduction_pct=(cd['uncontrolled'] - cd['shac']) / cd['uncontrolled'] * 100,
                        dtype=str(st.velocity[0].dtype))

# RB Ra=1e4
from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic, load_params_flexible
renv = RBControlEnv(dict(ENV_RA1E4)); rai, rE = int(renv.action_interval), int(ENV_RA1E4['episode_actions'])
ra = load_params_flexible(f'{ROOT}/runs/rayleigh_benard/shac_ra1e4_s42/checkpoints/final.npz')[0]['actor']
ra = [tuple(jnp.asarray(w, dtype=jnp.float64 if bits == 64 else jnp.float32) for w in l) for l in ra]
ract = jax.jit(lambda o: jnp.tanh(mlp_forward(ra, o)[:renv.action_size]))
s0 = build_ic(renv, 100)
for _ in range(int(ENV_RA1E4['warmup_steps'])): s0 = renv._step_jit(s0, renv.dt, None)
nu = {}
for mode in ('uncontrolled', 'shac'):
    s = s0; per = []
    for k in range(rE):
        T = renv.map_action(ract(renv._get_obs(s))) if mode == 'shac' else None
        for _ in range(rai): s = renv._step_jit(s, renv.dt, T)
        per.append(float(renv.compute_nu(s, T)) if T is not None else float(renv.compute_nu(s)))
    nu[mode] = float(np.mean(per[rE // 2:]))
out['rb_ra1e4'] = dict(nu_unctrl=nu['uncontrolled'], nu_ctrl=nu['shac'],
                       reduction_pct=(nu['uncontrolled'] - nu['shac']) / nu['uncontrolled'] * 100,
                       dtype=str(s.scalars['T'].dtype))
json.dump(out, open(f'{ROOT}/results/checks/precision_fp{bits}.json', 'w'), indent=1)
print(json.dumps(out))
