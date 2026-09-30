"""Temperature fields of Fig. 6.

For each Ra the uncontrolled flow and the final PPO and SHAC policies (training
seed 1223) run from held-out IC seed 100 with the same warm-up and number of control
actions; the fields are saved for figures/make_fig6_temp_grid.py.
Run one Ra per process:  python scripts/render_rb_timeavg_wm.py <1e4|1e5|1e6>
Writes results/figure_data/rb_timeavg_ra{Ra}.npz.
"""
import os, sys, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np
import jax, jax.numpy as jnp
from experiments.rb_control.config import ENV_CONFIG, ENV_CONFIG_RA1E5, ENV_CONFIG_RA1E6
from experiments.rb_control.train_ppo import mlp_forward
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import build_ic, load_params_flexible

RA = sys.argv[1] if len(sys.argv) > 1 else '1e4'
CFG = {'1e4': (ENV_CONFIG, 'phase2'), '1e5': (ENV_CONFIG_RA1E5, 'phase3'),
       '1e6': (ENV_CONFIG_RA1E6, 'phase3')}
env_cfg, phase = CFG[RA]
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
    dirs = [f'experiments/rb_control/output_{ph}_{algo}_ra{RA}_s1223/checkpoints' for ph in ('phase3', 'phase2')]
    if algo == 'shac' and RA in ('1e5', '1e6'):   # width-matched SHAC (64^2)
        dirs = [f'experiments/rb_control/output_wm_shac_ra{RA}_s1223/checkpoints'] + dirs
    for d in dirs:
        cks = ((glob.glob(f'{d}/fans_final_epoch_1000.npz') or glob.glob(f'{d}/fans_epoch_1000.npz'))
               if algo == 'shac' else glob.glob(f'{d}/ckpt_0100.npz'))
        if cks:
            return load_params_flexible(cks[0])[0]['actor']
    raise FileNotFoundError(f'no {algo} checkpoint for ra{RA}')


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
np.savez(f'results/figure_data/rb_timeavg_ra{RA}.npz', **save)
print(f'[rb-timeavg] saved rb_timeavg_ra{RA}.npz (instantaneous + avg)', flush=True)
