"""PPO policy-gradient variance and critic quality (Section 5.2).

At a PPO checkpoint (training seed 1223; final update ckpt_0100 by default) M
independent trajectory collections are drawn, each of the size used in training,
and the PPO actor gradient (GAE, critic and clipped surrogate exactly as in training)
is computed per collection.  Reports the gradient variance V, the signal ||gbar||^2,
the relative variance V/||gbar||^2, the critic R^2 and the episode-return variance.
Run:  python scripts/grad_variance_ppo.py <1e4|1e5|1e6> [M] [pool] [ckpt tag, default 0100]
Writes results/grad_variance/ppo_gradvar_ra{Ra}_ck{tag}.json.
"""
import os, sys, json, time
sys.path.insert(0, '.'); sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np
import jax
import jax.numpy as jnp

from experiments.rb_control.config import (
    ENV_CONFIG, PPO_V2_CONFIG, ENV_CONFIG_RA1E5, PPO_CONFIG_RA1E5,
    ENV_CONFIG_RA1E6, PPO_CONFIG_RA1E6)
from experiments.rb_control.train_ppo import collect_rollout, ppo_loss, critic_forward
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import load_params_flexible

RA = sys.argv[1] if len(sys.argv) > 1 else '1e4'
M = int(sys.argv[2]) if len(sys.argv) > 2 else 16        # independent collections
POOL = int(sys.argv[3]) if len(sys.argv) > 3 else 20     # IC pool size
CKTAG = sys.argv[4] if len(sys.argv) > 4 else '0100'     # checkpoint (0100=final)
CFG = {'1e4': (ENV_CONFIG, PPO_V2_CONFIG, 'phase2'),
       '1e5': (ENV_CONFIG_RA1E5, PPO_CONFIG_RA1E5, 'phase3'),
       '1e6': (ENV_CONFIG_RA1E6, PPO_CONFIG_RA1E6, 'phase3')}
env_cfg, ppo_cfg, phase = CFG[RA]
CKPT = f'experiments/rb_control/output_{phase}_ppo_ra{RA}_s1223/checkpoints/ckpt_{CKTAG}.npz'

env = RBControlEnv(env_cfg)
pool_seeds = list(range(100, 100 + POOL))
env.preload_ic_pool(pool_seeds, verbose=False)
params = load_params_flexible(CKPT)[0]                   # {'actor':[...], 'critic':[...]}
epl = int(ppo_cfg['episode_length']); npe = int(ppo_cfg['episodes_per_update'])
clip = float(ppo_cfg['clip_eps']); vf = float(ppo_cfg['vf_coef']); ent = float(ppo_cfg['ent_coef'])
gam = float(ppo_cfg['gamma']); lam = float(ppo_cfg['gae_lambda'])
print(f'[ppo-gradvar] Ra={RA} ep_len={epl} eps/upd={npe} M={M} pool={POOL} '
      f'clip={clip} obs={env.obs_size}', flush=True)


def loss_fn(p, obs, act, lp, adv, ret):
    return ppo_loss(p, obs, act, lp, adv, ret, clip, vf, ent)[0]
pg = jax.jit(jax.grad(loss_fn))                          # grad w.r.t. all params

# actor-subtree of the gradient = PPO policy gradient (value loss flows only to critic)
flat_actor = lambda g: jnp.concatenate(
    [x.ravel() for (W, b) in g['actor'] for x in (W, b)])


def critic_value_loss(p, obs, ret):
    vals = jax.vmap(critic_forward, in_axes=(None, 0))(p, obs)
    return float(jnp.mean((vals - ret) ** 2)), float(jnp.var(ret))


Sg = 0.0; Sgg = 0.0; n = 0
vls, vret, epret = [], [], []
rng = np.random.default_rng(0)
t0 = time.time()
for m in range(M):
    key = jax.random.PRNGKey(1000 + m)
    _, batch = collect_rollout(env, params, None, npe, epl, key,
                               ic_pool=pool_seeds, ic_rng=rng, gamma=gam, gae_lambda=lam)
    g = pg(params, batch['obs'], batch['actions'], batch['log_probs'],
           batch['advantages'], batch['returns'])
    ga = flat_actor(g); nrm = float(jnp.sum(ga ** 2))
    if not np.isfinite(nrm):
        print(f'  collection {m}: non-finite grad, skip', flush=True); continue
    Sg = Sg + ga; Sgg += nrm; n += 1
    vl, vr = critic_value_loss(params, batch['obs'], batch['returns'])
    vls.append(vl); vret.append(vr); epret.extend(batch['episode_returns'])
    print(f'  coll {m+1}/{M}: |g_actor|^2={nrm:.3e} value_loss={vl:.4f} '
          f'Var[ret]={vr:.4f} [{(time.time()-t0)/60:.1f}m]', flush=True)

gbar = Sg / n; sig = float(jnp.sum(gbar ** 2)); V = Sgg / n - sig
mean_g2 = Sgg / n                                        # robust: mean ||g||^2 (signal+noise)
mean_vl = float(np.mean(vls)); mean_vret = float(np.mean(vret))
out = {'Ra': RA, 'ckpt': CKTAG, 'M': n, 'ep_len': epl, 'eps_per_update': npe,
       'V_PPO': V, 'sig_PPO': sig, 'mean_gradnorm2_PPO': mean_g2,
       'relV_PPO': (V / sig if sig > 1e-30 else float('inf')),
       'mean_value_loss': mean_vl, 'mean_Var_step_return': mean_vret,
       'critic_R2': 1.0 - mean_vl / (mean_vret + 1e-12),
       'Var_episode_return': float(np.var(epret)),
       'mean_episode_return': float(np.mean(epret))}
os.makedirs('results/grad_variance', exist_ok=True)
json.dump(out, open(f'results/grad_variance/ppo_gradvar_ra{RA}_ck{CKTAG}.json', 'w'), indent=2)
print(f'\n=== PPO gradient variance Ra={RA} ===', flush=True)
for k, v in out.items():
    print(f'  {k:22s} {v}', flush=True)
print(f'[ppo-gradvar] saved [{(time.time()-t0)/60:.1f}m]', flush=True)
