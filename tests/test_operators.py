"""Tests for core differential operators."""
import jax.numpy as jnp
import pytest

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.operators import divergence, gradient_p, vorticity
from jax_cfd.core.state import FluidState


class TestDivergence:
    def test_uniform_flow_zero_div(self, grid_2d):
        """Uniform velocity field should have zero divergence."""
        u = jnp.ones(grid_2d.face_shape(0))
        v = jnp.zeros(grid_2d.face_shape(1))
        div = divergence((u, v), grid_2d)
        assert jnp.allclose(div, 0.0, atol=1e-10)

    def test_zero_flow_zero_div(self, grid_2d):
        u = jnp.zeros(grid_2d.face_shape(0))
        v = jnp.zeros(grid_2d.face_shape(1))
        div = divergence((u, v), grid_2d)
        assert jnp.allclose(div, 0.0, atol=1e-10)

    def test_3d_uniform_zero_div(self, grid_3d):
        u = jnp.ones(grid_3d.face_shape(0))
        v = jnp.zeros(grid_3d.face_shape(1))
        w = jnp.zeros(grid_3d.face_shape(2))
        div = divergence((u, v, w), grid_3d)
        assert jnp.allclose(div, 0.0, atol=1e-10)

    def test_div_shape(self, grid_2d):
        u = jnp.zeros(grid_2d.face_shape(0))
        v = jnp.zeros(grid_2d.face_shape(1))
        div = divergence((u, v), grid_2d)
        assert div.shape == grid_2d.cell_shape()


class TestGradientP:
    def test_constant_pressure_zero_gradient(self, grid_2d):
        p = jnp.ones(grid_2d.cell_shape()) * 5.0
        gp = gradient_p(p, grid_2d)
        assert len(gp) == 2
        # Interior gradient of constant = 0
        # (boundary points may not be zero due to ghost cells)
        # Check interior only
        assert jnp.allclose(gp[0][1:-1, :], 0.0, atol=1e-10)
        assert jnp.allclose(gp[1][:, 1:-1], 0.0, atol=1e-10)

    def test_gradient_shape(self, grid_2d):
        p = jnp.zeros(grid_2d.cell_shape())
        gp = gradient_p(p, grid_2d)
        assert gp[0].shape == grid_2d.face_shape(0)
        assert gp[1].shape == grid_2d.face_shape(1)


class TestVorticity:
    def test_uniform_flow_zero_vorticity(self, grid_2d):
        u = jnp.ones(grid_2d.face_shape(0))
        v = jnp.zeros(grid_2d.face_shape(1))
        w = vorticity((u, v), grid_2d)
        assert jnp.allclose(w, 0.0, atol=1e-6)

    def test_vorticity_2d_shape(self, grid_2d):
        u = jnp.ones(grid_2d.face_shape(0))
        v = jnp.zeros(grid_2d.face_shape(1))
        w = vorticity((u, v), grid_2d)
        assert w.shape == grid_2d.cell_shape()

    def test_vorticity_3d_returns_tuple(self, grid_3d):
        u = jnp.ones(grid_3d.face_shape(0))
        v = jnp.zeros(grid_3d.face_shape(1))
        w = jnp.zeros(grid_3d.face_shape(2))
        result = vorticity((u, v, w), grid_3d)
        assert isinstance(result, tuple)
        assert len(result) == 3
