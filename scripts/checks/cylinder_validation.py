"""Uncontrolled cylinder at Re = 100 (Section 2.3): time-mean and maximum drag
coefficient and Strouhal number over 45 000 steps (150 convective time units)
after the environment warm-up, and the jet momentum coefficient C_mu at full
actuation.  Configuration: configs/cylinder/env_re100.json (interface eps = 0.75).
Writes results/checks/cylinder_validation.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time
import numpy as np
import jax
from diffrl_flow_control.configs import load_config
from diffrl_flow_control.cylinder.env import CylinderFlowEnv

cfg = load_config('cylinder', 'env_re100')
cfg['baseline_steps'] = 1000
env = CylinderFlowEnv(cfg)
t0 = time.time(); out = env.reset(jax.random.PRNGKey(0))
print('reset done', round(time.time() - t0, 1), 's', flush=True)
state = env._current_state if getattr(env, '_current_state', None) is not None else (out[0] if isinstance(out, tuple) else out)
dt = env.dt; q = env.q_ref
step = jax.jit(lambda s: env._step_with_jets_pure(s, 0.0, dt))
cds, cls = [], []
N = 45000
for i in range(N):
    state = step(state); cds.append(float(state.ibm_forces[0]) / q); cls.append(float(state.ibm_forces[1]) / q)
cds = np.array(cds); cls = np.array(cls)
f = np.fft.rfftfreq(N, dt); P = np.abs(np.fft.rfft(cls - cls.mean())) ** 2; St = f[np.argmax(P[1:]) + 1]
os.makedirs('results/checks', exist_ok=True)
# jet momentum coefficient at |a| = 1: C_mu = 2 int u_j^2 ds / (U_inf^2 D) along the surface
sig = cfg['jet_width']; Q = cfg['Q_max']; R = cfg['R']; D = 2 * R
th = np.linspace(-np.pi, np.pi, 200001); d = 2 * R * np.sin(np.abs(th) / 2)
un = Q * np.exp(-d ** 2 / (2 * sig ** 2)); Cmu_jet = 2 * np.trapezoid(un ** 2, R * th) / (1.0 ** 2 * D)
res = {'cd_mean': float(cds.mean()), 'cd_max': float(cds.max()), 'cd_min': float(cds.min()), 'cl_max': float(cls.max()),
       'St': float(St), 'Cmu_per_jet': float(Cmu_jet), 'Cmu_pair': float(2 * Cmu_jet), 'steps': N, 'dt': dt, 'cfg': cfg}
json.dump(res, open('results/checks/cylinder_validation.json', 'w'), indent=1)
print(f"Re=100, {N} steps ({N * dt:.1f} t.u.): Cd mean={res['cd_mean']:.4f} max={res['cd_max']:.4f}  St={St:.4f}  "
      f"C_mu per jet={Cmu_jet:.4f}, pair={2 * Cmu_jet:.4f}")
