"""Single- versus double-precision check of the held-out reduction (representative cases).

  python scripts/verification/v01_fp64_check.py 32
  python scripts/verification/v01_fp64_check.py 64
For cylinder Re=100 and RB Ra=1e4, the final seed-42 SHAC policy is evaluated on held-out IC 100
with the Table 2/4 protocol (last-half mean against the uncontrolled mean over the same window of an
uncontrolled run from the same equilibrated state) in the requested precision.
Writes results/verification/v01_fp<bits>.json
"""
import os, sys, json
bits = int(sys.argv[1])
import jax
if bits == 64:
    jax.config.update('jax_enable_x64', True)
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
V2 = R
sys.path[:0] = [V2, V2 + '/src', V2 + '/scripts']
import numpy as np, jax.numpy as jnp
out = dict(bits=bits)

# cylinder Re=100
from cases.cylinder_afc import make_env_config
from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv
from cyl_eval_common import build_perturbed_ic, load_actor, EPISODE_ACTIONS
from experiments.afc_cylinder.train_ppo import mlp_forward_relu
cfg = make_env_config(Re=100, cells=(768, 144), ibm_eps_factor=0.75)
cfg['jet_method'] = 'surface_bc'; cfg['soft_reset_prob'] = 0.0; cfg['obs_type'] = 'velocity'
env = CylinderFlowEnv(cfg); ai, ema, E = int(env.action_interval), float(env.ema_alpha), EPISODE_ACTIONS
actor = [tuple(jnp.asarray(w, dtype=jnp.float64 if bits == 64 else jnp.float32) for w in l)
         for l in load_actor(f'{V2}/experiments/afc_cylinder/output_wm_shac_re100_768x144_s42/checkpoints/fans_final_epoch_1000.npz')]
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
from experiments.rb_control.config import ENV_CONFIG
from experiments.rb_control.train_ppo import mlp_forward
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import build_ic, load_params_flexible
renv = RBControlEnv(dict(ENV_CONFIG)); rai, rE = int(renv.action_interval), int(ENV_CONFIG['episode_actions'])
ra = load_params_flexible(f'{R}/experiments/rb_control/output_phase3_shac_ra1e4_s42/checkpoints/fans_final_epoch_1000.npz')[0]['actor']
ra = [tuple(jnp.asarray(w, dtype=jnp.float64 if bits == 64 else jnp.float32) for w in l) for l in ra]
ract = jax.jit(lambda o: jnp.tanh(mlp_forward(ra, o)[:renv.action_size]))
s0 = build_ic(renv, 100)
for _ in range(int(ENV_CONFIG['warmup_steps'])): s0 = renv._step_jit(s0, renv.dt, None)
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
json.dump(out, open(f'{R}/results/verification/v01_fp{bits}.json', 'w'), indent=1)
print(json.dumps(out))
