"""Temperature fields of Fig. 8.

For each Ra the uncontrolled flow and the final PPO and SHAC policies (training
seed 1223) run from held-out IC seed 100 with the same warm-up and number of control
actions.
Run one Ra per process:  python scripts/analysis/rayleigh_benard_fields.py <1e4|1e5|1e6>
Writes results/figure_data/rayleigh_benard_temperature_ra{Ra}.npz.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import glob
import numpy as np
import jax, jax.numpy as jnp
from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic, load_params_flexible

RA = sys.argv[1] if len(sys.argv) > 1 else '1e4'
env_cfg = {'1e4': ENV_RA1E4, '1e5': ENV_RA1E5, '1e6': ENV_RA1E6}[RA]
env = RBControlEnv(env_cfg)
warmup = int(env_cfg['warmup_steps'])
ep_actions = int(env_cfg.get('episode_actions', {'1e4': 200, '1e5': 250, '1e6': 400}[RA]))
ai = int(env_cfg.get('action_interval', 150))
adim = int(env.action_size)
SETTLE = ep_actions // 4                     # discard initial transient before averaging
print(f'[rb-timeavg] Ra={RA} warmup={warmup} ep_actions={ep_actions} ai={ai} '
      f'settle={SETTLE} grid={env.grid.shape}', flush=True)


def timeavg_T(actor):                        # actor=None -> uncontrolled
    state = build_ic(env, 100)
    for _ in range(warmup):
        state = env._step_jit(state, env.dt, None)
    if actor is not None:
        @jax.jit
        def policy(obs):
            return jnp.tanh(mlp_forward(actor, obs)[:adim])
    Tsum = np.zeros(env.grid.shape, np.float64); n = 0
    for a in range(ep_actions):
        Tseg = env.map_action(policy(env._get_obs_jit(state))) if actor is not None else None
        for _ in range(ai):
            state = env._step_jit(state, env.dt, Tseg)
        if a >= SETTLE:                      # average one frame per action, post-transient
            Tsum += np.asarray(state.scalars['T']); n += 1
    Tfinal = np.asarray(state.scalars['T'])  # well-developed instantaneous frame (same IC+time)
    return Tsum / n, Tfinal


def load_actor(algo):
    return load_params_flexible(f'runs/rayleigh_benard/{algo}_ra{RA}_s1223/checkpoints/final.npz')[0]['actor']


res = {'uncontrolled': timeavg_T(None)}
print('  uncontrolled done', flush=True)
for algo in ['ppo', 'shac']:
    res[algo] = timeavg_T(load_actor(algo))
    print(f'  {algo} done (final T range {res[algo][1].min():.3f}-{res[algo][1].max():.3f})', flush=True)

xc, yc = env.grid.cell_centers()
# save BOTH the well-developed instantaneous final frame (method name) and the
# time-average (method_avg), all from the same IC + same rollout window.
save = {'xc': np.asarray(xc), 'yc': np.asarray(yc)}
for k, (avg, fin) in res.items():
    save[k] = fin; save[f'{k}_avg'] = avg
np.savez(f'results/figure_data/rayleigh_benard_temperature_ra{RA}.npz', **save)
print(f'[rb-timeavg] saved rayleigh_benard_temperature_ra{RA}.npz (instantaneous + avg)', flush=True)
