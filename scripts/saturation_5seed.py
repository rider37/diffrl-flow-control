"""Pre-tanh saturation of the PPO policies at Ra=1e4 and 1e6 (Section 5.4).

For each final PPO checkpoint (five training seeds) the deterministic policy runs on a
held-out IC; at three snapshots the pre-tanh Gaussian mean mu_i of each of the 12 wall
segments is recorded.  Reports the median |mu_i| and the fraction with |mu_i| > 2,
the reference for the observation-normalisation check of
scripts/v12_obsnorm_ppo_ra1e6_final.py.
Writes results/saturation_5seed.json.
"""
import os, sys, json
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, '.'); sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from experiments.rb_control.config import ENV_CONFIG, ENV_CONFIG_RA1E6
from experiments.rb_control.train_ppo import mlp_forward
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import build_ic, load_params_flexible

SEEDS = [42, 123, 456, 517, 1223]
IC = 100
N_SNAP = 3


def ckpt(ra, s):
    d = 'phase3' if ra == '1e6' else 'phase2'
    return f'experiments/rb_control/output_{d}_ppo_ra{ra}_s{s}/checkpoints/ckpt_0100.npz'


def measure(ra, cfg):
    env = RBControlEnv({**cfg})
    warmup, ep, ad = cfg['warmup_steps'], cfg['episode_actions'], env.action_size
    ai = cfg['action_interval']
    snap_at = [ep // 4, ep // 2, 3 * ep // 4][:N_SNAP]
    mus = []
    for s in SEEDS:
        params, _ = load_params_flexible(ckpt(ra, s))
        actor = params['actor']

        @jax.jit
        def mu_of(obs):
            return mlp_forward(actor, obs)[:ad]  # pre-tanh Gaussian mean

        state = build_ic(env, IC)
        for _ in range(warmup):
            state = env._step_jit(state, env.dt, None)
        for a in range(ep):
            obs = env._get_obs_jit(state)
            mu = np.asarray(mu_of(obs))
            if a in snap_at:
                mus.append(np.abs(mu))
            action = jnp.tanh(jnp.asarray(mu))
            T_segs = env.map_action(action)
            for _ in range(ai):
                state = env._step_jit(state, env.dt, T_segs)
    allmu = np.concatenate(mus)  # (n_seed*n_snap*12,)
    return dict(ra=ra, n=len(SEEDS), median=float(np.median(allmu)),
                frac_gt2=float(np.mean(allmu > 2.0)), n_vals=int(allmu.size))


out = {}
for ra, cfg in [('1e6', ENV_CONFIG_RA1E6), ('1e4', ENV_CONFIG)]:
    r = measure(ra, cfg)
    out[ra] = r
    print(f"Ra={ra}: median|mu|={r['median']:.2f}, frac>2={r['frac_gt2']*100:.0f}% "
          f"(5 seeds x {N_SNAP} snap x 12 seg = {r['n_vals']} values)", flush=True)
os.makedirs('results', exist_ok=True)
json.dump(out, open('results/saturation_5seed.json', 'w'), indent=2)
print("Saved results/saturation_5seed.json")
