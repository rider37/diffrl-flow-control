"""Advective/diffusive CFL numbers and post-projection divergence residual at the production settings.

For one configuration (cylinder re100|re200|re400|re500, rayleigh_benard ra1e4|ra1e5|ra1e6): held-out IC 100 is equilibrated
as in the held-out evaluation, then an uncontrolled run and a run with the final seed-42 SHAC policy
each cover one evaluation episode.  At the end of every action interval the script records
  CFL_adv = dt * max(max|u|/dx, max|v|/dy),
  CFL_diff = nu * dt / min(dx, dy)^2 (and kappa for RB),
  mean |div u| and max |div u| over the grid (MAC divergence of the stored face velocities).
Writes results/checks/cfl_divergence_<problem>_<case>.json.  Run from the repository root.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time
import numpy as np, jax, jax.numpy as jnp
from diffrl_flow_control.solver.operators import divergence

problem, tag = sys.argv[1], sys.argv[2]
case = f'{problem}_{tag}'
out = dict(case=case)

def stats(env, st, nu, kappa=None):
    u, v = np.asarray(st.velocity[0]), np.asarray(st.velocity[1])
    dx, dy = env.grid.domain_size[0] / env.grid.shape[0], env.grid.domain_size[1] / env.grid.shape[1]
    div = np.abs(np.asarray(divergence(st.velocity, env.grid)))
    return dict(cfl_adv=float(env.dt * max(np.abs(u).max() / dx, np.abs(v).max() / dy)),
                div_mean=float(div.mean()), div_max=float(div.max()))

if problem == 'cylinder':
    Re = int(tag[2:])
    from diffrl_flow_control.configs import load_config
    from diffrl_flow_control.cylinder.env import CylinderFlowEnv
    from diffrl_flow_control.cylinder.heldout import build_perturbed_ic, load_actor, EPISODE_ACTIONS
    from diffrl_flow_control.cylinder.ppo import mlp_forward_relu
    cfg = load_config('cylinder', f'env_re{Re}')
    cfg['soft_reset_prob'] = 0.0; cfg['obs_type'] = 'velocity'
    env = CylinderFlowEnv(cfg); nu = 1.0 / Re; kappa = None
    ai, ema, E = int(env.action_interval), float(env.ema_alpha), EPISODE_ACTIONS
    actor = load_actor(f'{ROOT}/runs/cylinder/shac_re{Re}_s42/checkpoints/final.npz')
    act = jax.jit(lambda o: jnp.tanh(mlp_forward_relu(actor, o)[:1]))
    st0 = build_perturbed_ic(env, 100)
    for _ in range(int(env.warmup_steps)): st0 = env._step_jit(st0, env.dt, 0.0)
    for mode in ('uncontrolled', 'shac'):
        st = st0; Q = 0.0; rec = []
        for k in range(E):
            a = float(act(env._get_obs(st))[0]) if mode == 'shac' else 0.0
            for _ in range(ai):
                Q = Q + ema * (a - Q) if mode == 'shac' else 0.0
                st = env._step_jit(st, env.dt, Q)
            rec.append(stats(env, st, nu))
        out[mode] = rec
else:
    ra = tag[2:]
    from diffrl_flow_control.configs import load_config
    ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
    ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
    ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
    from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward
    from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
    from diffrl_flow_control.rayleigh_benard.heldout import build_ic, load_params_flexible
    cfg = {'1e4': ENV_RA1E4, '1e5': ENV_RA1E5, '1e6': ENV_RA1E6}[ra]
    env = RBControlEnv(dict(cfg)); nu = float(env.nu); kappa = nu / float(env.Pr)
    ck = (f'{ROOT}/runs/rayleigh_benard/shac_ra1e4_s42/checkpoints/final.npz' if ra == '1e4'
          else f'{ROOT}/runs/rayleigh_benard/shac_ra{ra}_s42/checkpoints/final.npz')
    actor = load_params_flexible(ck)[0]['actor']
    act = jax.jit(lambda o: jnp.tanh(mlp_forward(actor, o)[:env.action_size]))
    ai, E = int(env.action_interval), int(cfg['episode_actions'])
    st0 = build_ic(env, 100)
    for _ in range(int(cfg['warmup_steps'])): st0 = env._step_jit(st0, env.dt, None)
    for mode in ('uncontrolled', 'shac'):
        st = st0; rec = []
        for k in range(E):
            T = env.map_action(act(env._get_obs(st))) if mode == 'shac' else None
            for _ in range(ai): st = env._step_jit(st, env.dt, T)
            rec.append(stats(env, st, nu))
        out[mode] = rec
dx, dy = env.grid.domain_size[0] / env.grid.shape[0], env.grid.domain_size[1] / env.grid.shape[1]
out.update(dt=float(env.dt), dx=float(dx), dy=float(dy), nu=nu, kappa=kappa,
           cfl_diff_nu=float(nu * env.dt / min(dx, dy) ** 2),
           cfl_diff_kappa=(float(kappa * env.dt / min(dx, dy) ** 2) if kappa else None))
for mode in ('uncontrolled', 'shac'):
    r = out[mode]
    out[mode + '_summary'] = dict(cfl_adv_max=max(x['cfl_adv'] for x in r), cfl_adv_min=min(x['cfl_adv'] for x in r),
                                  div_mean_max=max(x['div_mean'] for x in r), div_max_max=max(x['div_max'] for x in r))
json.dump(out, open(f'{ROOT}/results/checks/cfl_divergence_{case}.json', 'w'), indent=1)
print(case, out['uncontrolled_summary'], out['shac_summary'], 'cfl_diff', out['cfl_diff_nu'], out['cfl_diff_kappa'])
