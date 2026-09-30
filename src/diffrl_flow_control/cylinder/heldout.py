"""Held-out evaluation protocol of the cylinder: the initial condition of an IC
seed (parabolic inflow plus a random localised cross-flow perturbation behind
the cylinder, equilibrated by the environment warm-up), the actor loader for
SHAC and PPO checkpoints, and the evaluation constants (IC seeds 100-104,
80-action episodes, training seeds, grid per Re).
"""
import numpy as np
import jax.numpy as jnp


from diffrl_flow_control.cylinder.env import FluidState

TEST_POOL = [100, 101, 102, 103, 104]
EPISODE_ACTIONS = 80
SEEDS = [42, 123, 456, 517, 1223]
RE_GRID = {100: (768, 144), 200: (768, 144), 400: (1280, 240), 500: (1280, 240)}


def build_perturbed_ic(env, seed):
    rng = np.random.default_rng(seed)
    cfg = env.cfg; grid = env.grid; U_inf = cfg['U_inf']
    if cfg.get('inlet_type') == 'parabolic':
        Um = cfg.get('Um', 1.5); Ly = grid.domain_size[1]; dy = Ly / grid.shape[1]
        y = jnp.arange(grid.shape[1]) * dy + dy / 2
        u = jnp.broadcast_to((Um * 4.0 * y * (Ly - y) / Ly**2)[None, :], grid.face_shape(0))
    else:
        u = jnp.ones(grid.face_shape(0)) * U_inf
    v = jnp.zeros(grid.face_shape(1))
    xv, yv = grid.face_centers(1); Xv, Yv = jnp.meshgrid(xv, yv, indexing='ij')
    px = cfg['cx'] + rng.uniform(1.0, 3.0) * cfg['R']
    py = cfg['cy'] + rng.uniform(-1.0, 1.0) * cfg['R']
    r2 = ((Xv - px) / cfg['R'])**2 + ((Yv - py) / cfg['R'])**2
    v = v + rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.08) * U_inf * jnp.exp(-r2 / 0.5)
    return FluidState(velocity=(u, v), pressure=jnp.zeros(grid.cell_shape()),
                      scalars={}, ibm_forces=None)


def load_actor(ckpt):
    d = np.load(ckpt)
    w, b = ('W', 'b') if 'actor_0_W' in d.files else ('0', '1')
    return [(jnp.asarray(d[f'actor_{i}_{w}']), jnp.asarray(d[f'actor_{i}_{b}']))
            for i in range(99) if f'actor_{i}_{w}' in d.files]
