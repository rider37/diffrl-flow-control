"""Tests for the Cahn-Hilliard phase-field module."""
import jax.numpy as jnp
import pytest

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.phase_field import (
    compute_chemical_potential,
    make_ch_scalar_rhs_fn,
    make_surface_tension_fn,
)


@pytest.fixture
def grid():
    return MACGrid((16, 16), (1.0, 1.0))


def _pad_neumann(field, name, grid):
    """Simple Neumann padding for all boundaries."""
    return jnp.pad(field, [(1, 1)] * grid.ndim, mode='edge')


def _zero_advect(phi, vel, grid):
    """Dummy advection: always zero (for testing diffusion-only)."""
    return jnp.zeros_like(phi)


class TestChemicalPotential:
    def test_equilibrium_plus_one(self, grid):
        """phi = +1 everywhere → mu = 1 - 1 - eps^2 * 0 = 0."""
        phi = jnp.ones(grid.cell_shape())
        mu = compute_chemical_potential(
            phi, grid, epsilon=0.01,
            pad_fn=lambda f, ndim: jnp.pad(f, [(1, 1)] * ndim, mode='edge'),
        )
        assert jnp.allclose(mu, 0.0, atol=1e-6)

    def test_equilibrium_minus_one(self, grid):
        """phi = -1 everywhere → mu = -1 + 1 - eps^2 * 0 = 0."""
        phi = -jnp.ones(grid.cell_shape())
        mu = compute_chemical_potential(
            phi, grid, epsilon=0.01,
            pad_fn=lambda f, ndim: jnp.pad(f, [(1, 1)] * ndim, mode='edge'),
        )
        assert jnp.allclose(mu, 0.0, atol=1e-6)

    def test_shape(self, grid):
        phi = jnp.zeros(grid.cell_shape())
        mu = compute_chemical_potential(
            phi, grid, epsilon=0.01,
            pad_fn=lambda f, ndim: jnp.pad(f, [(1, 1)] * ndim, mode='edge'),
        )
        assert mu.shape == grid.cell_shape()


class TestCahnHilliardRHS:
    def test_mass_conservation(self, grid):
        """With zero velocity, CH diffusion should conserve total phi."""
        ch_rhs = make_ch_scalar_rhs_fn(
            grid, mobility=0.01, epsilon=0.05,
            advect_scalar_fn=_zero_advect,
            pad_phi_fn=_pad_neumann,
        )
        # Smooth initial phi: circular droplet
        xc, yc = grid.cell_centers()
        r = jnp.sqrt((xc[:, None] - 0.5) ** 2 + (yc[None, :] - 0.5) ** 2)
        phi = jnp.tanh((0.2 - r) / 0.05)

        vel = tuple(jnp.zeros(grid.face_shape(i)) for i in range(2))
        scalars = {'phi': phi}
        rhs = ch_rhs(scalars, vel, None, 0.01)

        # Integral of RHS should be ~0 (mass conservation with Neumann BC)
        total_rhs = float(jnp.sum(rhs['phi']))
        assert abs(total_rhs) < 1.0


class TestSurfaceTension:
    def test_zero_for_uniform_phi(self, grid):
        """Single phase (phi=1) should produce zero surface tension force."""
        st_fn = make_surface_tension_fn(
            grid, sigma=1.0, epsilon=0.05,
            pad_phi_fn=_pad_neumann,
        )
        phi = jnp.ones(grid.cell_shape())
        vel = tuple(jnp.zeros(grid.face_shape(i)) for i in range(2))
        forces = st_fn({'phi': phi}, vel, grid)
        assert len(forces) == 2
        for f in forces:
            assert jnp.allclose(f, 0.0, atol=1e-10)
