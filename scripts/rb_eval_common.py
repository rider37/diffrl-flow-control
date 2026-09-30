"""Shared helpers for the Rayleigh-Benard held-out evaluation: held-out initial
condition (IC seed -> perturbed conductive temperature field), checkpoint loader
for PPO and SHAC parameter files, uncontrolled Nu measurement, and the
deterministic controlled rollout (tanh of the actor output, one action per
action interval) returning the Nu series.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.rb_control.train_ppo import mlp_forward
from jax_cfd.envs.rb_control_env import FluidState

TEST_POOL = [100, 101, 102, 103, 104]
SEEDS = [42, 123, 456, 517, 1223]


def build_ic(env, seed):
    rng = np.random.default_rng(seed)
    grid = env.grid
    Nx, Ny = grid.shape
    Lx, Ly = grid.domain_size

    u = jnp.zeros(grid.face_shape(0))
    v = jnp.zeros(grid.face_shape(1))
    p = jnp.zeros(grid.cell_shape())

    xc, yc = grid.cell_centers()
    T_lin = env.T_hot + (env.T_cold - env.T_hot) * yc / Ly
    T = np.broadcast_to(T_lin[None, :], (Nx, Ny)).copy()

    XC, YC = np.meshgrid(xc, yc, indexing='ij')
    amp = env.perturb_amp
    for k in range(1, 7):
        a_k = rng.uniform(0.5, 1.5) * amp
        phi_k = rng.uniform(0.0, 2.0 * np.pi)
        T = T + a_k * np.cos(2.0 * np.pi * k * XC / Lx + phi_k) * np.sin(np.pi * YC / Ly)

    return FluidState(velocity=(u, v), pressure=p, scalars={'T': jnp.asarray(T)}, ibm_forces=None)


def load_params_flexible(ckpt_path):
    d = np.load(ckpt_path)
    files = d.files
    params = {'actor': [], 'critic': []}
    fmt = 'shac' if any(f.endswith('_W') for f in files) else 'ppo'
    for k in ('actor', 'critic'):
        i = 0
        if fmt == 'ppo':
            while f'{k}_{i}_0' in files:
                params[k].append((jnp.asarray(d[f'{k}_{i}_0']), jnp.asarray(d[f'{k}_{i}_1'])))
                i += 1
        else:
            while f'{k}_{i}_W' in files:
                params[k].append((jnp.asarray(d[f'{k}_{i}_W']), jnp.asarray(d[f'{k}_{i}_b'])))
                i += 1
    return params, fmt


def measure_bl(env, seed, warmup, baseline):
    state = build_ic(env, seed)
    for _ in range(warmup):
        state = env._step_jit(state, env.dt, None)
    nus = []
    for _ in range(baseline):
        state = env._step_jit(state, env.dt, None)
        nus.append(float(env._compute_nu_jit(state)))
    return float(np.mean(nus))


def eval_on_ic(env, actor_params, seed, warmup, ep_actions, action_dim):
    state = build_ic(env, seed)
    for _ in range(warmup):
        state = env._step_jit(state, env.dt, None)

    @jax.jit
    def policy(obs):
        out = mlp_forward(actor_params, obs)
        return jnp.tanh(out[:action_dim])

    action_interval = env.cfg.get('action_interval', 150)
    nus = np.empty(ep_actions)
    for a in range(ep_actions):
        obs = env._get_obs_jit(state)
        action = policy(obs)
        T_segs = env.map_action(action)
        for _ in range(action_interval):
            state = env._step_jit(state, env.dt, T_segs)
        nus[a] = float(env.compute_nu(state, T_segs))
    return nus
