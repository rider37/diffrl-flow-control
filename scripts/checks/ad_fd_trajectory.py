"""AD against finite-difference gradients along a trained trajectory (Section 5.3,
Appendix F): Ra = 1e6, final SHAC policy (training seed 42), states sampled along
its controlled trajectory, AD gradient of the truncated return versus centred
finite differences along random parameter directions.
Writes results/checks/ad_fd_trajectory.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import glob, json, time
import numpy as np
import jax, jax.numpy as jnp

from diffrl_flow_control.configs import load_config
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import load_params_flexible as load_params


H = 10
EPS = 1e-3
N_STATES = 10
SEED_DIR = 42


def actor_loss(actor_params, env, state0, h):
    """Sum of negative reward over h actions with deterministic policy."""
    state = state0
    total = 0.0
    for _ in range(h):
        obs = env._get_obs(state)
        a = jnp.tanh(mlp_forward(actor_params, obs))
        for _ in range(int(env.action_interval)):
            state = env._step_jit(state, env.dt, a)
        # Reward proxy: -Nu (segment-averaged) is the SHAC reward;
        # we use the IBM-free RB Nu signal here.
        try:
            nu = env.compute_nu(state)
        except Exception:
            nu = jnp.sum(state.scalars['T'])
        total = total - nu
    return total


def main():
    cfg = dict(ENV_RA1E6)
    env = RBControlEnv(cfg)
    state0, _ = env.reset()

    ckpt_dir = f'runs/rayleigh_benard/shac_ra1e6_s{SEED_DIR}/checkpoints'
    cands = [f'{ckpt_dir}/final.npz']
    if not cands:
        raise SystemExit(f'no ckpt at {ckpt_dir}')
    params, _ = load_params(cands[-1])
    actor = params['actor']

    # Roll to a converged state (use the env's preloaded warmup).
    state = state0

    rng = np.random.default_rng(0)
    cos = []; ads = []; fds = []
    for i in range(N_STATES):
        # advance the rollout 100 steps before each probe
        for _ in range(100):
            obs = env._get_obs(state)
            a = jnp.tanh(mlp_forward(actor, obs))
            for _ in range(int(env.action_interval)):
                state = env._step_jit(state, env.dt, a)

        # Random unit direction in parameter space
        flat = [np.asarray(w) for layer in actor for w in layer]
        shapes = [a.shape for a in flat]
        sizes = [a.size for a in flat]
        total = sum(sizes)
        v_flat_np = rng.standard_normal(total)
        v_flat_np = v_flat_np / np.linalg.norm(v_flat_np)

        def to_struct(flat_np):
            out = []
            i = 0
            for L_idx in range(len(actor)):
                pair = []
                for k in range(2):
                    sz = sizes[L_idx*2 + k]
                    shp = shapes[L_idx*2 + k]
                    pair.append(jnp.asarray(flat_np[i:i+sz].reshape(shp)))
                    i += sz
                out.append(tuple(pair))
            return out

        v_struct = to_struct(v_flat_np)

        def loss_for(theta):
            return actor_loss(theta, env, state, H)

        # AD directional derivative via jvp
        _, ad = jax.jvp(loss_for, (actor,), (v_struct,))
        ad = float(ad)

        # FD: ±eps along v
        plus  = to_struct(np.concatenate([np.asarray(w).reshape(-1)
                                          for layer in actor for w in layer])
                          + EPS * v_flat_np)
        minus = to_struct(np.concatenate([np.asarray(w).reshape(-1)
                                          for layer in actor for w in layer])
                          - EPS * v_flat_np)
        Lp = float(loss_for(plus))
        Lm = float(loss_for(minus))
        fd = (Lp - Lm) / (2 * EPS)

        if abs(fd) > 1e-8 and abs(ad) > 1e-8:
            c = ad * fd / (abs(ad) * abs(fd))
        else:
            c = float('nan')
        print(f'  state {i+1:2d}/{N_STATES}: AD={ad:+.4e} FD={fd:+.4e} '
              f'AD/FD={ad/fd if abs(fd)>0 else float("nan"):+.3f} '
              f'sign-cos={c:+.2f}', flush=True)
        cos.append(c); ads.append(ad); fds.append(fd)
    cos = np.array(cos)
    print(f'\nMedian sign-cosine = {np.nanmedian(cos):+.2f}; '
          f'fraction with AD/FD ratio in [0.7, 1.3] = '
          f'{np.mean((np.abs(cos)>0.7) & (~np.isnan(cos))):.0%}')
    ratio = [a / f if abs(f) > 0 else float('nan') for a, f in zip(ads, fds)]
    out = dict(n_states=int(N_STATES), eps=float(EPS), horizon_actions=int(H), checkpoint=cands[-1],
               ad=[float(x) for x in ads], fd=[float(x) for x in fds], ratio_ad_over_fd=ratio,
               ratio_min=float(np.nanmin(ratio)), ratio_max=float(np.nanmax(ratio)), ratio_median=float(np.nanmedian(ratio)),
               sign_cosines=[float(c) for c in cos.tolist()])
    json.dump(out, open('results/checks/ad_fd_trajectory.json', 'w'), indent=1)
    print('ratio min/median/max', out['ratio_min'], out['ratio_median'], out['ratio_max'])


if __name__ == '__main__':
    main()
