"""PPO policy-gradient variance and critic quality (Section 5.2).

At a PPO checkpoint (training seed 1223, final policy) M
independent trajectory collections are drawn, each of the size used in training,
and the PPO actor gradient (GAE, critic and clipped surrogate exactly as in training)
is computed per collection.  Reports the gradient variance V, the signal ||gbar||^2,
the relative variance V/||gbar||^2, the critic R^2 and the episode-return variance.
Run:  python scripts/analysis/ppo_gradient_variance.py <1e4|1e5|1e6> [M] [pool]
Writes results/diagnostics/ppo_gradient_variance_ra{Ra}.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time
import numpy as np
import jax
import jax.numpy as jnp

from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
PPO_RA1E4 = load_config('rayleigh_benard', 'ppo_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
PPO_RA1E5 = load_config('rayleigh_benard', 'ppo_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
PPO_RA1E6 = load_config('rayleigh_benard', 'ppo_ra1e6')
from diffrl_flow_control.rayleigh_benard.ppo import collect_rollout, ppo_loss, critic_forward
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import load_params_flexible

RA = sys.argv[1] if len(sys.argv) > 1 else '1e4'
M = int(sys.argv[2]) if len(sys.argv) > 2 else 16        # independent collections
POOL = int(sys.argv[3]) if len(sys.argv) > 3 else 20     # IC pool size
CFG = {'1e4': (ENV_RA1E4, PPO_RA1E4), '1e5': (ENV_RA1E5, PPO_RA1E5), '1e6': (ENV_RA1E6, PPO_RA1E6)}
env_cfg, ppo_cfg = CFG[RA]
CKPT = f'runs/rayleigh_benard/ppo_ra{RA}_s1223/checkpoints/final.npz'

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
out = {'Ra': RA, 'ckpt': 'final', 'M': n, 'ep_len': epl, 'eps_per_update': npe,
       'V_PPO': V, 'sig_PPO': sig, 'mean_gradnorm2_PPO': mean_g2,
       'relV_PPO': (V / sig if sig > 1e-30 else float('inf')),
       'mean_value_loss': mean_vl, 'mean_Var_step_return': mean_vret,
       'critic_R2': 1.0 - mean_vl / (mean_vret + 1e-12),
       'Var_episode_return': float(np.var(epret)),
       'mean_episode_return': float(np.mean(epret))}
os.makedirs('results/diagnostics', exist_ok=True)
json.dump(out, open(f'results/diagnostics/ppo_gradient_variance_ra{RA}.json', 'w'), indent=2)
print(f'\n=== PPO gradient variance Ra={RA} ===', flush=True)
for k, v in out.items():
    print(f'  {k:22s} {v}', flush=True)
print(f'[ppo-gradvar] saved [{(time.time()-t0)/60:.1f}m]', flush=True)
