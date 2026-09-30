"""The two control environments built from the released configurations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffrl_flow_control.configs import load_config
from diffrl_flow_control.cylinder.env import CylinderFlowEnv
from diffrl_flow_control.cylinder.heldout import build_perturbed_ic
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic


@pytest.mark.parametrize('case', ['re100', 're200', 're400', 're500'])
def test_cylinder_configs(case):
    env = load_config('cylinder', f'env_{case}')
    for alg in ('shac', 'ppo'):
        cfg = load_config('cylinder', f'{alg}_{case}')
        assert cfg['actor_hidden_dims'] == [512, 512] and cfg['critic_hidden_dims'] == [32, 32]
    assert env['ibm_eps_factor'] == 0.75 and env['obs_type'] == 'velocity'


@pytest.mark.parametrize('case', ['ra1e4', 'ra1e5', 'ra1e6'])
def test_rayleigh_benard_configs(case):
    load_config('rayleigh_benard', f'env_{case}')
    for alg in ('shac', 'ppo'):
        cfg = load_config('rayleigh_benard', f'{alg}_{case}')
        dims = cfg.get('actor_hidden_dims', cfg.get('hidden_dims'))
        assert dims == [64, 64]


def test_cylinder_step_and_observation():
    cfg = dict(load_config('cylinder', 'env_re100'), Nx=192, Ny=36)
    env = CylinderFlowEnv(cfg)
    st = build_perturbed_ic(env, 100)
    st = env._step_jit(st, env.dt, 0.5)
    obs = env._get_obs(st)
    assert obs.shape == (302,) and bool(jnp.all(jnp.isfinite(obs)))
    assert np.isfinite(float(st.ibm_forces[0]))


def test_rayleigh_benard_step_action_map_and_nusselt():
    env = RBControlEnv(load_config('rayleigh_benard', 'env_ra1e4'))
    T = env.map_action(jnp.linspace(-1.0, 1.0, env.action_size))
    assert abs(float(jnp.mean(T)) - env.T_hot) < 1e-6          # mean heater temperature held
    assert float(jnp.max(jnp.abs(T - env.T_hot))) <= env.amplitude + 1e-6
    st = build_ic(env, 100)
    st = env._step_jit(st, env.dt, T)
    assert env._get_obs(st).shape == (1152,)
    assert np.isfinite(float(env.compute_nu(st, T)))


def test_rayleigh_benard_is_differentiable():
    env = RBControlEnv(load_config('rayleigh_benard', 'env_ra1e4'))
    st0 = build_ic(env, 100)

    def nu_after(a):
        T = env.map_action(a)
        st = jax.lax.fori_loop(0, 200, lambda i, s: env.step_fn(s, env.dt, T), st0)
        return env.compute_nu(st, T)

    a = 0.3 * jnp.cos(jnp.arange(env.action_size, dtype=jnp.float32))
    g = jax.grad(nu_after)(a)
    k = jnp.arange(env.action_size, dtype=jnp.float32)
    d = jnp.sin(2 * jnp.pi * k / env.action_size)          # zero-mean direction (the mean is removed by map_action)
    d = d / jnp.linalg.norm(d)
    fd = (nu_after(a + 5e-2 * d) - nu_after(a - 5e-2 * d)) / 1e-1
    assert bool(jnp.all(jnp.isfinite(g)))
    np.testing.assert_allclose(float(jnp.dot(g, d)), float(fd), rtol=0.1, atol=1e-6)
