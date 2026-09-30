"""Verification copy of scripts/wm_variants/ad_fd_on_trajectory_ra1e6_wm.py that saves AD, FD and ratios
to results/verification/v06_ad_fd_on_trajectory_ra1e6.json.  Run from the repository root."""
"""On-trajectory AD vs centred finite-difference gradient check at
Ra=1e6, complementing the random-policy probes in the main text.

For each of 10 states drawn from a converged SHAC rollout, we compare
the directional derivative of the short-horizon actor loss in a
random unit direction of the actor-parameter space:
  AD   : <grad_theta L_pi, v>     (via jax.jvp / jax.grad . v)
  FD   : (L_pi(theta + eps v) - L_pi(theta - eps v)) / (2 eps)
and report the cosine similarity AD / FD (= 1 if the gradient is
perfectly accurate in that direction).
"""
import os, sys, glob, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import jax, jax.numpy as jnp

from experiments.rb_control.config import ENV_CONFIG_RA1E6
from experiments.rb_control.train_ppo import mlp_forward
from jax_cfd.envs.rb_control_env import RBControlEnv
from scripts.rb_eval_common import load_params_flexible as load_params


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
    cfg = dict(ENV_CONFIG_RA1E6)
    env = RBControlEnv(cfg)
    state0, _ = env.reset()

    ckpt_dir = (f'experiments/rb_control/'
                f'output_wm_shac_ra1e6_s{SEED_DIR}/checkpoints')   # width-matched SHAC
    cands = sorted(glob.glob(f'{ckpt_dir}/fans_final_*.npz')) or \
            sorted(glob.glob(f'{ckpt_dir}/fans_epoch_*.npz'))
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
    json.dump(out, open('results/verification/v06_ad_fd_on_trajectory_ra1e6.json', 'w'), indent=1)
    print('ratio min/median/max', out['ratio_min'], out['ratio_median'], out['ratio_max'])


if __name__ == '__main__':
    main()
