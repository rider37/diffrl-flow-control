"""Tests for MACGrid — shapes, spacings, coordinates."""
import jax.numpy as jnp
import pytest

from jax_cfd.core.mac_grid import MACGrid


class TestMACGrid2D:
    def test_shape(self, grid_2d):
        assert grid_2d.shape == (16, 16)

    def test_ndim(self, grid_2d):
        assert grid_2d.ndim == 2

    def test_cell_shape(self, grid_2d):
        assert grid_2d.cell_shape() == (16, 16)

    def test_face_shapes(self, grid_2d):
        assert grid_2d.face_shape(0) == (17, 16)
        assert grid_2d.face_shape(1) == (16, 17)

    def test_spacings_uniform(self, grid_2d):
        dx, dy = grid_2d.spacings
        assert abs(dx - 1.0 / 16) < 1e-10
        assert abs(dy - 1.0 / 16) < 1e-10

    def test_is_uniform(self, grid_2d):
        assert grid_2d.is_uniform

    def test_min_spacing(self, grid_2d):
        expected = 1.0 / 16
        assert abs(grid_2d.min_spacing() - expected) < 1e-10

    def test_cell_centers(self, grid_2d):
        cc = grid_2d.cell_centers()
        assert len(cc) == 2
        assert cc[0].shape == (16,)
        # First cell center should be at dx/2
        dx = 1.0 / 16
        assert abs(float(cc[0][0]) - dx / 2) < 1e-10

    def test_cell_volumes_uniform(self, grid_2d):
        vol = grid_2d.cell_volumes()
        expected = (1.0 / 16) ** 2
        assert jnp.allclose(vol, expected, atol=1e-10)


class TestMACGrid3D:
    def test_shape(self, grid_3d):
        assert grid_3d.shape == (8, 8, 8)

    def test_ndim(self, grid_3d):
        assert grid_3d.ndim == 3

    def test_face_shapes(self, grid_3d):
        assert grid_3d.face_shape(0) == (9, 8, 8)
        assert grid_3d.face_shape(1) == (8, 9, 8)
        assert grid_3d.face_shape(2) == (8, 8, 9)

    def test_cell_volumes(self, grid_3d):
        vol = grid_3d.cell_volumes()
        expected = (1.0 / 8) ** 3
        assert jnp.allclose(vol, expected, atol=1e-10)


class TestStretchedGrid:
    def test_stretched_creation(self):
        g = MACGrid((16, 16), (1.0, 1.0),
                    stretch={'axis': 1, 'type': 'tanh', 'ratio': 3.0})
        assert not g.is_uniform
        assert g.ndim == 2

    def test_stretched_min_lt_uniform(self):
        g_uni = MACGrid((16, 16), (1.0, 1.0))
        g_str = MACGrid((16, 16), (1.0, 1.0),
                        stretch={'axis': 1, 'type': 'tanh', 'ratio': 3.0})
        assert g_str.min_spacing() < g_uni.min_spacing()
