"""Advective/diffusive CFL numbers and post-projection divergence residual at the production settings.

For one configuration (cyl100|cyl200|cyl400|cyl500|rb1e4|rb1e5|rb1e6): held-out IC 100 is equilibrated
as in the held-out evaluation, then an uncontrolled run and a run with the final seed-42 SHAC policy
each cover one evaluation episode.  At the end of every action interval the script records
  CFL_adv = dt * max(max|u|/dx, max|v|/dy),
  CFL_diff = nu * dt / min(dx, dy)^2 (and kappa for RB),
  mean |div u| and max |div u| over the grid (MAC divergence of the stored face velocities).
Writes results/verification/v02_stability_<case>.json.  Run from the repository root.
"""
import os, sys, json, time
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
V2 = R
sys.path[:0] = [V2, V2 + '/src', V2 + '/scripts']
import numpy as np, jax, jax.numpy as jnp
from jax_cfd.core.operators import divergence

case = sys.argv[1]
out = dict(case=case)

def stats(env, st, nu, kappa=None):
    u, v = np.asarray(st.velocity[0]), np.asarray(st.velocity[1])
    dx, dy = env.grid.domain_size[0] / env.grid.shape[0], env.grid.domain_size[1] / env.grid.shape[1]
    div = np.abs(np.asarray(divergence(st.velocity, env.grid)))
    return dict(cfl_adv=float(env.dt * max(np.abs(u).max() / dx, np.abs(v).max() / dy)),
                div_mean=float(div.mean()), div_max=float(div.max()))

if case.startswith('cyl'):
    Re = int(case[3:]); grid = {100: (768, 144), 200: (768, 144), 400: (1280, 240), 500: (1280, 240)}[Re]
    from cases.cylinder_afc import make_env_config
    from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv
    from cyl_eval_common import build_perturbed_ic, load_actor, EPISODE_ACTIONS
    from experiments.afc_cylinder.train_ppo import mlp_forward_relu
    cfg = make_env_config(Re=Re, cells=grid, ibm_eps_factor=0.75)
    cfg['jet_method'] = 'surface_bc'; cfg['soft_reset_prob'] = 0.0; cfg['obs_type'] = 'velocity'
    env = CylinderFlowEnv(cfg); nu = 1.0 / Re; kappa = None
    ai, ema, E = int(env.action_interval), float(env.ema_alpha), EPISODE_ACTIONS
    actor = load_actor(f'{V2}/experiments/afc_cylinder/output_wm_shac_re{Re}_{grid[0]}x{grid[1]}_s42/checkpoints/fans_final_epoch_1000.npz')
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
    ra = case[2:]
    from experiments.rb_control.config import ENV_CONFIG, ENV_CONFIG_RA1E5, ENV_CONFIG_RA1E6
    from experiments.rb_control.train_ppo import mlp_forward
    from jax_cfd.envs.rb_control_env import RBControlEnv
    from rb_eval_common import build_ic, load_params_flexible
    cfg = {'1e4': ENV_CONFIG, '1e5': ENV_CONFIG_RA1E5, '1e6': ENV_CONFIG_RA1E6}[ra]
    env = RBControlEnv(dict(cfg)); nu = float(env.nu); kappa = nu / float(env.Pr)
    ck = (f'{R}/experiments/rb_control/output_phase3_shac_ra1e4_s42/checkpoints/fans_final_epoch_1000.npz' if ra == '1e4'
          else f'{R}/experiments/rb_control/output_wm_shac_ra{ra}_s42/checkpoints/fans_final_epoch_1000.npz')
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
json.dump(out, open(f'{R}/results/verification/v02_stability_{case}.json', 'w'), indent=1)
print(case, out['uncontrolled_summary'], out['shac_summary'], 'cfl_diff', out['cfl_diff_nu'], out['cfl_diff_kappa'])
