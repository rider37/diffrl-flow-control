"""Forward-only rollout time over one SHAC training window versus the measured SHAC update time (checks/shac_cost.py), same GPU.

For cylinder Re = 200 (h=48 actions x 50 steps) and Rayleigh-Benard Ra = 1e6 (h=50 actions x 150 steps) the uncontrolled jitted
step is scanned over h*interval CFD steps from an equilibrated state; the median of 5 timed runs is
compared with the median SHAC update time of checks/shac_cost.py on the same device.
Writes results/checks/forward_vs_update.json
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time
import numpy as np, jax
out = {}
def timed(fn, x, reps=5):
    jax.block_until_ready(fn(x)); ts = []
    for _ in range(reps):
        t = time.time(); jax.block_until_ready(fn(x)); ts.append(time.time() - t)
    return float(np.median(ts))
# cylinder Re=200
from diffrl_flow_control.configs import load_config
from diffrl_flow_control.cylinder.env import CylinderFlowEnv
from diffrl_flow_control.cylinder.heldout import build_perturbed_ic
cfg = load_config('cylinder', f'env_re{200}'); cfg['soft_reset_prob'] = 0.0
env = CylinderFlowEnv(cfg); s = build_perturbed_ic(env, 100)
for _ in range(2000): s = env._step_jit(s, env.dt, 0.0)
n = 48 * 50
fwd = jax.jit(lambda st: jax.lax.scan(lambda c, _: (env._step_jit(c, env.dt, 0.0), None), st, None, length=n)[0])
sub = jax.jit(lambda st: env._step_jit(st, env.dt, 0.0))
cost_cyl = json.load(open(f'{ROOT}/results/checks/shac_cost_cylinder_re200.json'))
t_f = timed(fwd, s); t_u = float(np.median(cost_cyl['epoch_times'][1:]))
out['cylinder_re200'] = dict(n_steps=n, substep_ms=timed(sub, s, 50) * 1e3, forward_s=t_f, shac_update_s=t_u, ratio=t_u / t_f, device=jax.devices()[0].device_kind)
# RB Ra=1e6
from diffrl_flow_control.configs import load_config
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic
renv = RBControlEnv(dict(ENV_RA1E6)); r = build_ic(renv, 100)
for _ in range(2000): r = renv._step_jit(r, renv.dt, None)
m = 50 * 150
rfwd = jax.jit(lambda st: jax.lax.scan(lambda c, _: (renv._step_jit(c, renv.dt, None), None), st, None, length=m)[0])
rsub = jax.jit(lambda st: renv._step_jit(st, renv.dt, None))
cost_rb = json.load(open(f'{ROOT}/results/checks/shac_cost_rayleigh_benard_ra1e6.json'))
t_f = timed(rfwd, r, 3); t_u = float(np.median(cost_rb['epoch_times'][1:]))
out['rayleigh_benard_ra1e6'] = dict(n_steps=m, substep_ms=timed(rsub, r, 50) * 1e3, forward_s=t_f, shac_update_s=t_u, ratio=t_u / t_f, device=jax.devices()[0].device_kind)
json.dump(out, open(f'{ROOT}/results/checks/forward_vs_update.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
