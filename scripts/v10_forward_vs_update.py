"""Forward-only rollout time over one SHAC training window versus the measured SHAC update time (v03), same GPU.

For cyl200 (h=48 actions x 50 steps) and rb1e6 (h=50 actions x 150 steps) the uncontrolled jitted
step is scanned over h*interval CFD steps from an equilibrated state; the median of 5 timed runs is
compared with the median SHAC update time of v03 on the same device.
Writes results/verification/v10_forward_vs_update.json
"""
import os, sys, json, time
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); V2 = R
sys.path[:0] = [V2, V2 + '/src', V2 + '/scripts']
import numpy as np, jax
out = {}
def timed(fn, x, reps=5):
    jax.block_until_ready(fn(x)); ts = []
    for _ in range(reps):
        t = time.time(); jax.block_until_ready(fn(x)); ts.append(time.time() - t)
    return float(np.median(ts))
# cylinder Re=200
from cases.cylinder_afc import make_env_config
from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv
from cyl_eval_common import build_perturbed_ic
cfg = make_env_config(Re=200, cells=(768, 144), ibm_eps_factor=0.75); cfg['jet_method'] = 'surface_bc'; cfg['soft_reset_prob'] = 0.0
env = CylinderFlowEnv(cfg); s = build_perturbed_ic(env, 100)
for _ in range(2000): s = env._step_jit(s, env.dt, 0.0)
n = 48 * 50
fwd = jax.jit(lambda st: jax.lax.scan(lambda c, _: (env._step_jit(c, env.dt, 0.0), None), st, None, length=n)[0])
sub = jax.jit(lambda st: env._step_jit(st, env.dt, 0.0))
v03 = json.load(open(f'{R}/results/verification/v03_shac_memory_time_cyl200.json'))
t_f = timed(fwd, s); t_u = float(np.median(v03['epoch_times'][1:]))
out['cyl200'] = dict(n_steps=n, substep_ms=timed(sub, s, 50) * 1e3, forward_s=t_f, shac_update_s=t_u, ratio=t_u / t_f, device=jax.devices()[0].device_kind)
# RB Ra=1e6
from experiments.rb_control.config import ENV_CONFIG_RA1E6
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import build_ic
renv = RBControlEnv(dict(ENV_CONFIG_RA1E6)); r = build_ic(renv, 100)
for _ in range(2000): r = renv._step_jit(r, renv.dt, None)
m = 50 * 150
rfwd = jax.jit(lambda st: jax.lax.scan(lambda c, _: (renv._step_jit(c, renv.dt, None), None), st, None, length=m)[0])
rsub = jax.jit(lambda st: renv._step_jit(st, renv.dt, None))
v03r = json.load(open(f'{R}/results/verification/v03_shac_memory_time_rb1e6.json'))
t_f = timed(rfwd, r, 3); t_u = float(np.median(v03r['epoch_times'][1:]))
out['rb1e6'] = dict(n_steps=m, substep_ms=timed(rsub, r, 50) * 1e3, forward_s=t_f, shac_update_s=t_u, ratio=t_u / t_f, device=jax.devices()[0].device_kind)
json.dump(out, open(f'{R}/results/verification/v10_forward_vs_update.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
