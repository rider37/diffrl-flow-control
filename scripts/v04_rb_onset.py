"""Verification copy of scripts/validate_rb_critical.py that also saves results/verification/v04_rb_onset.json."""
"""Same-config (bottom-heated) RB validation: the CRITICAL Rayleigh number.

2D RB, no-slip isothermal top/bottom: linear theory gives parameter-free onset
at Ra_c = 1707.76 (Nu->1 below, Nu>1 above). This is the config-specific check
that the RB buoyancy coupling + non-dim are correct (complements de Vahl Davis).

NOTE: the env's default dt=0.025 is tuned for Ra=1e4; at LOW Ra, nu=1/sqrt(Ra/Pr)
grows and the explicit-diffusion limit nu*dt*(1/dx^2+1/dy^2)<~0.5 is violated
(blows up). So we set a per-Ra stable dt here.
"""
import os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src')); sys.path.insert(0, _ROOT)
import numpy as np, jax, jax.numpy as jnp
from jax_cfd.envs.rb_control_env import RBControlEnv

Pr = 0.7
Lx, Ly, Nx, Ny = 2*np.pi, 2.0, 96, 64
dx, dy = Lx/Nx, Ly/Ny
diff_fac = 1.0/dx**2 + 1.0/dy**2
T_END = 500.0       # physical time (near-onset growth is slow -> longer)

RAS = [100, 200, 300, 400, 600]
print(f"{'Ra':>6}{'dt':>9}{'Nu_final':>10}{'steps':>8}{'time':>7}")
print("-" * 42)
results = []
for Ra in RAS:
    nu = 1.0 / np.sqrt(Ra / Pr)
    dt = min(0.02, 0.25 / (nu * diff_fac))      # explicit-diffusion-stable
    env = RBControlEnv({'Ra': float(Ra), 'dt': float(dt)})
    state = env._initial_state()                # perturbed conduction, u=v=0
    nsteps = int(T_END / dt)
    t0 = time.time(); nus = []
    for n in range(nsteps):
        state = env._step_jit(state, dt, None)  # uncontrolled = uniform T_hot
        if (n + 1) % (nsteps // 6) == 0:
            nus.append(float(env.compute_nu(state)))
    nu_final = float(np.mean(nus[-2:]))         # last ~third, smoothed
    print(f"{Ra:>6}{dt:>9.4f}{nu_final:>10.4f}{nsteps:>8}{time.time()-t0:>6.0f}s"
          f"   trace={[f'{x:.2f}' for x in nus]}", flush=True)
    results.append((Ra, nu_final, [float(x) for x in nus], float(dt), int(nsteps)))

print("\n=== onset vs theory Ra_c = 1707.76 ===")
import json
json.dump([dict(Ra=r[0], Nu_final=r[1], Nu_trace=r[2], dt=r[3], steps=r[4]) for r in results],
          open(os.path.join(_ROOT, 'results', 'verification', 'v04_rb_onset.json'), 'w'), indent=1)
below = [r[0] for r in results if r[1] < 1.05]
above = [r[0] for r in results if r[1] >= 1.05]
if below and above:
    print(f"  conduction (Nu~1) up to Ra={max(below)};  convection (Nu>1) from Ra={min(above)}")
    print(f"  -> Ra_c in ({max(below)}, {min(above)})  [theory 1707.76]")
else:
    print(f"  below(Nu~1): {below}   above(Nu>1): {above}  -> inspect trace")
