"""Tests for turbulence models."""
import jax
import jax.numpy as jnp
import pytest

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState
from jax_cfd.core.turbulence import get_turbulence_model


def _make_sheared_state(grid):
    """Create a sheared flow field that activates turbulence models."""
    key = jax.random.PRNGKey(0)
    ndim = grid.ndim
    vel = []
    for i in range(ndim):
        vel.append(0.5 * jnp.ones(grid.face_shape(i))
                   + 0.1 * jax.random.normal(key, grid.face_shape(i)))
        key = jax.random.split(key)[0]
    p = jnp.zeros(grid.cell_shape())
    return FluidState(velocity=tuple(vel), pressure=p, scalars={}, ibm_forces=None)


class TestLaminar:
    def test_returns_zero(self, grid_2d, state_2d):
        fn = get_turbulence_model('laminar')
        nu_t = fn(state_2d, grid_2d, {})
        assert jnp.allclose(nu_t, 0.0)

    def test_shape(self, grid_2d, state_2d):
        fn = get_turbulence_model('laminar')
        nu_t = fn(state_2d, grid_2d, {})
        assert nu_t.shape == grid_2d.cell_shape()


class TestSmagorinsky:
    def test_non_negative(self, grid_2d):
        state = _make_sheared_state(grid_2d)
        fn = get_turbulence_model('smagorinsky', Cs=0.1)
        nu_t = fn(state, grid_2d, {})
        assert jnp.all(nu_t >= 0)

    def test_uniform_flow_near_zero(self, grid_2d, state_2d):
        fn = get_turbulence_model('smagorinsky', Cs=0.1)
        nu_t = fn(state_2d, grid_2d, {})
        assert float(jnp.max(nu_t)) < 1e-6

    def test_shape(self, grid_2d):
        state = _make_sheared_state(grid_2d)
        fn = get_turbulence_model('smagorinsky', Cs=0.1)
        nu_t = fn(state, grid_2d, {})
        assert nu_t.shape == grid_2d.cell_shape()


class TestWALE:
    def test_non_negative(self, grid_2d):
        state = _make_sheared_state(grid_2d)
        fn = get_turbulence_model('wale', Cw=0.32)
        nu_t = fn(state, grid_2d, {})
        assert jnp.all(nu_t >= 0)

    def test_shape(self, grid_2d):
        state = _make_sheared_state(grid_2d)
        fn = get_turbulence_model('wale')
        nu_t = fn(state, grid_2d, {})
        assert nu_t.shape == grid_2d.cell_shape()


class TestDynamicSmagorinsky:
    def test_non_negative(self, grid_2d):
        state = _make_sheared_state(grid_2d)
        fn = get_turbulence_model('dynamic_smagorinsky', Cs_max=0.3, periodic_axes=())
        nu_t = fn(state, grid_2d, {})
        assert jnp.all(nu_t >= 0)

    def test_bounded(self, grid_2d):
        state = _make_sheared_state(grid_2d)
        fn = get_turbulence_model('dynamic_smagorinsky', Cs_max=0.3, periodic_axes=())
        nu_t = fn(state, grid_2d, {})
        assert float(jnp.max(nu_t)) < 1.0

    def test_shape(self, grid_2d):
        state = _make_sheared_state(grid_2d)
        fn = get_turbulence_model('dynamic_smagorinsky', Cs_max=0.3, periodic_axes=())
        nu_t = fn(state, grid_2d, {})
        assert nu_t.shape == grid_2d.cell_shape()


class TestFactoryErrors:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown turbulence model"):
            get_turbulence_model('nonexistent_model')
